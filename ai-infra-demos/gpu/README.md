# GPU 分支运行手册（A800-80GB / AutoDL 单卡）

对应《跑全Demo规划(单卡A800-80G)》里标 **S** 的需要 GPU 大模型项。本沙箱无 GPU，以下脚本**已按逻辑编码并通过语法检查、未在 GPU 实跑**，请在 A800 机器按本手册逐项执行并把结果回填到本文件（作为验收记录）。

## 0. 环境准备（按 跑全Demo规划 A45-47 约定）

- 镜像：AutoDL 官方 `PyTorch 2.5/2.6 + CUDA 12.1/12.4`；A800 = Ampere sm80，装 bnb 时确认版本支持 sm80；

- `pip install transformers datasets accelerate peft bitsandbytes trl vllm autoawq`（分步装，注意版本兼容）；

- 模型建议用 Qwen2.5 系列（国内拉取方便或用 modelscope / 本地缓存目录替换 `--model`），数据/模型缓存放大数据盘(≥100GB)。

## 1. 7B QLoRA：QKV vs FFN（考点13增强，≈1.5-3h，显存 12-16GB）

```bash
# 先准备 500-1000 条指令 jsonl（每行 {"instruction":..., "output":...}，可抽 alpaca 子集）
python gpu/g01_qlora_7b_qkv_vs_ffn.py --model Qwen/Qwen2.5-7B-Instruct \
    --data ./data/alpaca_1k.jsonl --steps 300
```

预期验收：① 打印 trainable ≈ <1%；② 两张 target\_modules 方案的 显存峰值(≈12-16G)/loss/耗时 表；
③ 同 prompt 两版生成样例。回填表：

| 方案                   | 显存峰值GB | loss   | 耗时min  | 生成质量观感 |
| -------------------- | ------ | ------ | ------ | ------ |
| LoRA on q,k,v,o      | <br /> | <br /> | <br /> | <br /> |
| LoRA on gate,up,down | <br /> | <br /> | <br /> | <br /> |

## 2. 7B GRPO + RLVR 数学（考点16·17增强，≈3-5h，显存 60-70GB）

```bash
python gpu/g02_grpo_7b_rlvr.py --model Qwen/Qwen2.5-7B-Instruct --steps 200 --use-vllm
```

预期验收：① 训练前后同 100 题 holdout 的 pass\@1 提升曲线；② 若干训练前后推理样例对比；
③ 显存峰值记录。若 OOM：减 `--per_device_train_batch_size`/`--g`/`--max-len`。

> 说明：脚本内置 `reward_func`(字符串精确判 a+b) 即 RLVR verifier；GRPO 组内 advantage 由 trl 完成（机制讲解见 `part3/d14`）。

## 3. 13B nf4 推理 + 7B AWQ（考点21增强，1-2h，显存 ≤30GB）

```bash
python gpu/g03_infer_13b_4bit_and_awq_7b.py \
    --model7 Qwen/Qwen2.5-7B-Instruct \
    --model13 Qwen/Qwen2.5-14B-Instruct
```

（13B 建议换成 13B 级模型均可；nf4 权重 ≈8-9GB。）
AWQ 可选：先 `python -m awq.entry --model_path <7B> --calib_data wikitext --export_path outputs/g03_awq --quant_file awq`，再 `--awq-dir outputs/g03_awq` 加载对比。
预期验收：三列表 fp16-7B / nf4-13B / AWQ-7B 的 显存+tok/s+样例。

## 验收记录表（执行后回填）

| # | 脚本  | 命令(关键参数) | 显存实测   | 时长     | 关键指标                   | 通过?    |
| - | --- | -------- | ------ | ------ | ---------------------- | ------ |
| 1 | g01 | ...      | <br /> | <br /> | trainable<1%, QKV优于FFN | <br /> |
| 2 | g02 | ...      | <br /> | <br /> | pass\@1 前后提升           | <br /> |
| 3 | g03 | ...      | <br /> | <br /> | 三列表                    | <br /> |

## 常见坑

- bitsandbytes 与 CUDA/显卡架构不匹配 → 报错先查 `nvidia-smi` 与 bnb 版本；

- Qwen tokenizer 需 `trust_remote_code=True`（脚本已带）；

- GRPO 用 vLLM 采样比 transformers 快很多；不用时脚本也兼容（较慢）；

- 多卡并行项（真实 3D/FSDP/RingAllReduce 跨卡）**表格规划明确不跑**，直讲理论。

