# LLM AI-Infra 面试考点代码演示库

> 由《大模型AI岗位面试考点清单.xlsx》中 **AI Infra 岗**（6 Part / 34 考点）的"讲解设计"落地的可运行代码。
> 每个文件 = 1 个自包含 demo：numpy/从零实现或 torch 训 tiny 模型，**CPU 可跑**；GPU 大模型类脚本在 `gpu/`。

## 设计原则（与表格一致）
- AI Infra 是理论岗，绝大多数考点用「从零实现 + 数值验证 + tiny 模型」讲透原理与公式；
- **AB 效果类考点**（GQA、MoE、LoRA、DPO、量化、投机解码、合成数据、评测污染…）跑对比并留图，面试可以说"我测过"；
- 真实多卡（DP/TP/PP、FSDP/ZeRO、RingAllReduce 跨卡）**只讲公式 + CPU 模拟**，不假装跑过多卡；
- 图中文字一律英文（服务器缺中文字体），代码注释/打印为中文（备课材料）。

## 运行环境
Python 3 + `numpy matplotlib torch`（CPU 版即可；沙箱 2 核/3GB 内存全量跑通）。tiny 模型默认参数小，几秒~几分钟；想复现更强效果可加步数/尺寸。

```bash
pip install numpy matplotlib torch
python part1_transformer/d01_multihead_attention.py     # 单文件即可跑
```

## 考点 → 文件索引

| Part | 考点 | 文件 |
|---|---|---|
| P1 Transformer | 1·2 Self-Attention/MHA/√d_k/QKV投影/多头 | `part1_transformer/d01_multihead_attention.py` |
| P1 | 3 LayerNorm 维度 vs BN | `part1_transformer/d02_layernorm_vs_batchnorm.py` |
| P1 | 4 位置编码 sin/cos·RoPE·ALiBi·外推插值 | `part1_transformer/d03_position_encoding.py` |
| P1 | 5 GQA/MQA/滑动窗口 + KV 显存 | `part1_transformer/d04_gqa_mqa_attention.py` |
| P1 | 6 CSA+HCA/Infini-Attention 数值验证 | `part1_transformer/d05_infini_attention.py` |
| P1 | 7 后 Transformer：SSM 递推==卷积/选择性 | `part1_transformer/d06_ssm_recursion.py` |
| P2 预训练Scaling | 8 预训练/SFT/RL 三阶段 | `part2_pretrain_scaling/d07_three_stage_tiny_gpt.py` |
| P2 | 9 Scaling Law 幂律 + 6ND 计算器 | `part2_pretrain_scaling/d08_scaling_law_calculator.py` |
| P2 | 10 MoE top-2 路由 + 负载均衡 | `part2_pretrain_scaling/d09_moe_top2.py` |
| P2 | 11 灾难性遗忘 + 回放/EWC | `part2_pretrain_scaling/d10_catastrophic_forgetting.py` |
| P3 微调对齐 | 12·13 LoRA 从零实现 + QKV vs FFN | `part3_finetune_align/d11_lora_from_scratch.py` |
| P3 | 14 RLHF 三阶段（bandit PG + KL + PPO clip） | `part3_finetune_align/d12_rlhf_bandit_ppo.py` |
| P3 | 15 DPO/ORPO/SimPO 从零实现 | `part3_finetune_align/d13_dpo_family.py` |
| P3 | 16·17 GRPO 组内优势 + RLVR verifier | `part3_finetune_align/d14_grpo_rlvr.py` |
| P3 | 19 对齐税 / Reward Hacking | `part3_finetune_align/d15_reward_hacking.py` |
| P4 推理优化 | 20 KV Cache 增量解码 + PagedAttention | `part4_inference/d16_kv_cache.py` |
| P4 | 21 量化（对称/非对称/逐通道 + 位宽扫描） | `part4_inference/d17_quantization.py` |
| P4 | 22 投机解码（n-gram draft + 拒绝采样） | `part4_inference/d18_speculative_decoding.py` |
| P4 | 24 解码策略 greedy/temp/topk/topp/beam | `part4_inference/d19_sampling_strategies.py` |
| P5 分布式显存 | 25 DP/TP/PP + PP 调度 bubble | `part5_distributed/d20_dp_tp_pp.py` |
| P5 | 26 ZeRO/FSDP 显存记账 | `part5_distributed/d21_zero_memory_accounting.py` |
| P5 | 29 训练显存估算计算器 | `part5_distributed/d22_train_memory_calculator.py` |
| P5 | 27 Ring AllReduce 模拟 + NVLink vs PCIe | `part5_distributed/d23_ring_allreduce_simulation.py` |
| P5 | 28 Flash Attention 分块 softmax 数值 | `part5_distributed/d24_flash_attention_tiled.py` |
| P6 数据工程 | 30 合成数据配比 + 模型崩溃 | `part6_data_engineering/d25_synthetic_data_mix.py` |
| P6 | 31 数据处理 pipeline（清洗/去重/配比） | `part6_data_engineering/d26_data_pipeline.py` |
| P6 | 34 评测 runner + 数据污染演示 | `part6_data_engineering/d27_eval_pollution.py` |
| P3 | 13增强 7B QLoRA QKV-vs-FFN | `gpu/g01_qlora_7b_qkv_vs_ffn.py` |
| P3 | 16·17增强 7B GRPO+RLVR 数学 | `gpu/g02_grpo_7b_rlvr.py` |
| P4 | 21增强 13B nf4 推理 + 7B AWQ | `gpu/g03_infer_13b_4bit_and_awq_7b.py` |

**纯理论（无代码、面试直讲，讲解点见各文件 docstring / 表格 Demo详细设计）**：
考点18 DAPO/URPO/MetroRLHF、考点23 Prefill-Decode 分离与 D=128/G−1、考点32 后训练扩展律、考点33 NSP/多模态/具身。

## 目录结构
- `common/`：`corpus.py`(确定性合成语料)、`tinygpt.py`(可配置 tiny GPT：pos_kind/attn_kind/GQA/MQA)、`train.py`(预训练/SFT-mask 训练循环)、`toylm.py`(part4 缓存小模型)、`viz.py`(存图到 outputs/)
- `part1..part6/`：按考点组织的 demo（文件名含考点编号）
- `gpu/`：A800-80GB 分支（需 GPU 依赖，见 `gpu/README.md`）
- `outputs/`：运行产生的图（自动生成）；`sample_output.jsonl`(d26 输出)

## 常见说明
- 各脚本顶部 docstring 写了「考点 / 讲解思路 / 运行方式」；具体讲稿请另行生成，本库只做代码+讲解骨架；
- tiny 实验用合成语料（seed 固定可复现）；换真实语料：`common/corpus.load_text_file()` 传入 txt 即可；
- 图里标签为英文是刻意为之；表格设计要求的 AB 数值都会 print 到终端，便于贴进备课笔记。
