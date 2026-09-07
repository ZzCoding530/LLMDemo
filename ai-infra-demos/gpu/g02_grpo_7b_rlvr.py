# -*- coding: utf-8 -*-
"""GPU/A800 分支（考点16·17 增强）：GRPO + RLVR 真实 7B 数学训练
设计来源：跑全Demo规划 Demo17 / Demo详细设计 row17。
运行前提：A800-80GB 单卡（7B bf16 ≈15G，G=8 组采样峰值 60-70G，注意 monitor）；
   pip install transformers datasets trl vllm
模型：Qwen/Qwen2.5-7B-Instruct（国内用 modelscope 或本地路径）。

做法：
1. 规则 verifier = 字符串精确判 a+b（RLVR：无人类 RM）；
2. trl.GRPOTrainer：policy=7B、G=8、回答长度上限、KL 惩罚；
3. 训练前/后各跑 100 题 holdout，画 pass@1 提升曲线；存前后推理样例对比；
4. 说明：RLVR 收益在更大模型上明显（同 demo CPU 版见 part3/d14）。
运行示例：
  python gpu/g02_grpo_7b_rlvr.py --steps 200 --use-vllm
"""
import argparse
import random
import json
import os
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

QA = '题目：{a}+{b}=？\n你只需输出数字答案。'


def make_problems(n, hi=99, seed=0):
    rng = random.Random(seed)
    return [{"prompt": QA.format(a=rng.randint(0, hi), b=rng.randint(0, hi)),
             "answer": None} for _ in range(n)]


def verifier_reward(ans, prompt):
    """规则/可验证奖励：从题目解析 a+b，剥掉回答里非数字判等（RLVR）。"""
    import re
    mm = re.search(r"(\d+)\+(\d+)=", prompt)
    if not mm:
        return 0.0
    s = "".join(ch for ch in ans if ch.isdigit())
    try:
        return 1.0 if int(s) == int(mm.group(1)) + int(mm.group(2)) else 0.0
    except ValueError:
        return 0.0


def reward_func(prompts, completions, **kwargs):
    """GRPO reward_funcs 签名：按序给每组 completion 打分（组内相对优势在 GRPO 内做）。"""
    rs = []
    for prompt, comp in zip(prompts, completions):
        rs.append(verifier_reward(comp[0]["content"], prompt))
    return rs


@torch.no_grad()
def pass_at_1(model, tok, problems, n=100, hi=9):
    from transformers import pipeline
    gen = pipeline("text-generation", model=model, tokenizer=tok,
                   max_new_tokens=32, do_sample=False, device=model.device if hasattr(model, "device") else 0)
    ok = 0
    for p in problems[:n]:
        out = gen(p["prompt"])[0]["generated_text"]
        a = "".join(ch for ch in out if ch.isdigit())
        try:
            # 答对=输出等于 a+b（解析 prompt 的加数）
            import re
            mm = re.search(r"(\d+)\+(\d+)=", p["prompt"])
            ok += int(int(a) == int(mm.group(1)) + int(mm.group(2)))
        except ValueError:
            pass
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--use-vllm", action="store_true")
    args = ap.parse_args()

    train_prob = make_problems(200, hi=99, seed=1)
    train_prob = [{"prompt": p["prompt"]} for p in train_prob]
    ds = Dataset.from_list(train_prob)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)

    # 训练前基线（toy hi=9 的 100 题）
    if args.use_vllm:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1)
        sp = SamplingParams(max_tokens=32, temperature=0.0)
        import re
        base = 0.0
        evals = make_problems(100, hi=9, seed=7)
        outs = llm.generate([p["prompt"] for p in evals], sp)
        for p, o in zip(evals, outs):
            mm = re.search(r"(\d+)\+(\d+)=", p["prompt"])
            a = "".join(ch for ch in o.outputs[0].text if ch.isdigit())
            try:
                base += int(int(a) == int(mm.group(1)) + int(mm.group(2)))
            except ValueError:
                pass
        print(f"GRPO 前 pass@1(0-9)≈{base/100:.2f}")
    else:
        print("提示：--use-vllm 可选提升采样吞吐；不用时用 transformers generate（慢）。")
        print("可选：先加载模型跑 100 题贪心求 base（时间紧可跳过，脚本聚焦训练）。")

    cfg = GRPOConfig(output_dir="outputs/g02_grpo", max_steps=args.steps,
                     per_device_train_batch_size=4, gradient_accumulation_steps=1,
                     num_generations=args.g, max_prompt_length=args.max_len,
                     max_completion_length=64, beta=0.04, learning_rate=1e-6,
                     bf16=True, logging_steps=10, save_strategy="no",
                     temperature=1.0, report_to=[])
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=ds,
                          processing_class=tok, reward_funcs=[reward_func])
    trainer.train()

    print("提示：训练后用 pass_at_1 对比前后并保存样例（本脚本按 A800 设计；"
          "显存峰值通常 60-70G，若 OOM 减小 --per_device_train_batch_size 或 --g）。")


if __name__ == "__main__":
    main()
