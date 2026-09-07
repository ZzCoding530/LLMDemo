# -*- coding: utf-8 -*-
"""GPU/A800 分支（考点13 增强）：7B QLoRA 真实微调 —— QKV vs FFN 冻结对比
设计来源：跑全Demo规划 Demo13 / Demo详细设计 row13。
运行前提：A800-80GB 单卡；pip install transformers peft bitsandbytes accelerate datasets
模型源（国内）：可用 modelscope 下载 Qwen/Qwen2.5-7B-Instruct 后改 MODEL 为本地路径。

做法：
1. nf4 量化加载 7B（double_quant + bf16 compute）；
2. 同一份指令数据分别 LoRA 挂在 {q,k,v,o}(attention) 与 {gate,up,down}(FFN)；
3. 记录 显存峰值/loss/时长，同一批 prompt 对比两版生成质量；
4. 说明：全参 7B 需要 8×80G 或 ZeRO（见 part5），单卡只能 QLoRA。

运行示例：
  python gpu/g01_qlora_7b_qkv_vs_ffn.py --data ./my_alpaca_1k.jsonl --steps 300
数据 jsonl 每行: {"instruction":..., "output":...}
"""
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig  # trl>=0.9 风格；老版本用 TrainingArguments+Trainer
from datasets import Dataset

PROMPT = "{instruction}\n\n### 回答："


def load_data(path, n=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({"text": PROMPT.format(instruction=d["instruction"]) + d["output"]})
            if n and len(rows) >= n:
                break
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default="outputs/g01")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    variants = {
        "QKV-attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "FFN": ["gate_proj", "up_proj", "down_proj"],
    }
    data = load_data(args.data)
    records = []
    for name, modules in variants.items():
        print(f"\n########## LoRA on {modules} ##########")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(r=args.r, lora_alpha=args.alpha, target_modules=modules,
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        cfg = SFTConfig(output_dir=f"{args.out}/{name}", per_device_train_batch_size=args.batch,
                        gradient_accumulation_steps=args.grad_acc, max_seq_length=args.seq,
                        num_train_epochs=1, max_steps=args.steps, logging_steps=20,
                        save_strategy="no", optim="paged_adamw_8bit", bf16=True,
                        learning_rate=2e-4, report_to=[])
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=data,
                             processing_class=tok, dataset_text_field="text")
        trainer.train()
        mem = torch.cuda.max_memory_allocated() / 1e9
        records.append((name, mem, trainer.state.log_history[-1].get("loss", 0), time.time() - t0))
        # 推理对比（同一批 prompt）
        model.eval()
        with torch.no_grad():
            for ins in ["写一句关于森林的句子", "1+1 等于几？"]:
                ids = tok(PROMPT.format(instruction=ins), return_tensors="pt").to(model.device)
                out = model.generate(**ids, max_new_tokens=64, do_sample=False)
                print(f"[{name}] {ins}\n  -> {tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)}")
        del model, trainer
        torch.cuda.empty_cache()

    print("\n===== 结果表 =====")
    for name, mem, loss, sec in records:
        print(f"target={name:<14} 显存峰值={mem:.1f}GB  loss≈{loss:.4f}  耗时={sec/60:.1f}min")
    print("提示：真实 7B 全参微调需 ~8×80G（模型状态≈112GB，见 part5/d22 计算器）；单卡走 QLoRA。")


if __name__ == "__main__":
    main()
