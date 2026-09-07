# -*- coding: utf-8 -*-
"""GPU/A800 分支（考点21 增强）：7B fp16 基线 vs 13B nf4(4bit) 推理；7B AWQ 校准(可选)
设计来源：跑全Demo规划 Demo22 / Demo详细设计 row22。
运行前提：A800-80GB 单卡；pip install transformers bitsandbytes
（AWQ 可选：另需 autoawq + 128 条校准文本；本脚本自动跳过未装依赖的部分。）

做法：
1. fp16 7B 基线 → bitsandbytes nf4 13B → 记录 显存/tokens/s/示例输出；
2. AWQ（可选）：若本地已有 AWQ int4 目录则可加载对比；
3. 三列打表：显存 / 速度 / 质量(主观样例，或自行接困惑度)。
运行示例：
  python gpu/g03_infer_13b_4bit_and_awq_7b.py \
      --model7 Qwen/Qwen2.5-7B-Instruct \
      --model13 Qwen/Qwen2.5-14B-Instruct
"""
import argparse
import time
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompts(n=10):
    return ["用一句话解释：大语言模型的 KV Cache 是什么？"] * n


def gen_and_time(model, tok, prompts, max_new=64):
    model.eval()
    t0 = time.time()
    ntok = 0
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(model.device)
            o = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
            ntok += o.shape[1] - ids["input_ids"].shape[1]
            outs.append(tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    dt = time.time() - t0
    return outs, ntok / dt, torch.cuda.max_memory_allocated() / 1e9


def run(tag, model_name, quant=None):
    print(f"\n>> {tag}: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16 if quant is None else None,
        trust_remote_code=True)
    outs, tps, mem = gen_and_time(model, tok, build_prompts())
    torch.cuda.empty_cache()
    return tag, mem, tps, outs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model7", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model13", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--awq-dir", default=None,
                    help="可选的 AWQ int4 量化目录（用 AutoAWQ 预先量化好）")
    args = ap.parse_args()

    nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    rows = [run("7B-fp16", args.model7)]
    rows.append(run("13B-nf4", args.model13, quant=nf4))

    if args.awq_dir:
        try:
            from awq import AutoAWQForCausalLM
            print("\n>> AWQ int4 加载：", args.awq_dir)
            m = AutoAWQForCausalLM.from_quantized(args.awq_dir, "awq", device="cuda:0")
            tok = AutoTokenizer.from_pretrained(args.model7)
            outs, tps, mem = gen_and_time(m.model, tok, build_prompts())
            rows.append(("7B-AWQ-int4", mem, tps, outs[0]))
        except Exception as e:  # noqa
            print("AWQ 加载失败（未装 autoawq 或目录格式不对），跳过：", e)

    print("\n===== 质量-显存-速度 三列对比表 =====")
    print(f"{'方案':<12} {'显存GB':>7} {'tok/s':>7}  输出前 60 字")
    for tag, mem, tps, ex in rows:
        print(f"{tag:<12} {mem:>7.1f} {tps:>7.0f}  {ex[:60]!r}")
    print("\n提示：13B nf4 权重约 8-9GB；若想量化出自己的 AWQ 7B：")
    print("  python -m awq.entry --model_path <7B> --calib_data wikitext "
          "--export_path outputs/g03_awq --quant_file awq")


if __name__ == "__main__":
    main()
