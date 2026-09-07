# RUNBOOK · 验收记录与复现指南

本文件记录两件事：

1. **沙箱已实测通过**的每个 CPU demo（命令 + 代表性结果），对应《Demo详细设计》的验收标准；
2. **缺硬件(GPU)待测项**的执行记录位（脚本在 `gpu/`，命令与预期见 `gpu/README.md`），请你拉到 A800 后按 `gpu/README.md` 表格回填。

验收口径说明：CPU 沙箱（2 核 / \~3GB 内存）为省时间大多用**减半步数**跑通（表格标注了所用命令）。
在正常 CPU/AutoDL 机器上，把各脚本默认参数或命令里的 steps/尺寸调大即可得到更强曲线。

## 一、沙箱实测通过清单（每项均输出数值与图片，图在 outputs/）

> 通用：`pip install numpy matplotlib torch`(CPU)；结果含 `[fig] saved -> outputs/xxx.png`。

### Part1 · Transformer 与序列架构

| 文件                                  | 实测命令                                                     | 关键输出（沙箱）                                                                                           | <br /> | <br />         |
| ----------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | :----- | :------------- |
| d01\_multihead\_attention.py        | `python part1_transformer/d01_multihead_attention.py`    | d\_k=64 起 不除√d\_k 熵→0/maxP→0.99(one-hot)，除√后熵 0.85+；梯度量级 raw 1e-4→0；MHA 输出形状正确 + 4 头热力图            | <br /> | <br />         |
| d02\_layernorm\_vs\_batchnorm.py    | `python part1_transformer/d02_layernorm_vs_batchnorm.py` | LN 逐样本不受离群影响(diff=0)，BN 被污染(diff=2.03)；tiny 分类器 batch=2 时 LN 优于 BN(0.333 vs 0.560)                 | <br /> | <br />         |
| d03\_position\_encoding.py          | `... d03 ... --steps 120`                                | RoPE 平移不变误差 8.9e-15；外推：sin/cos loss 1.83→2.04，RoPE 0.78→0.81，ALiBi 0.64→0.63                       | <br /> | <br />         |
| d04\_gqa\_mqa\_attention.py         | `... d04 ... --steps 150`                                | 7B 例 32K：MHA 42.9GB→GQA-8 10.7→MQA 1.34GB；tiny val\_loss MHA 1.26 / GQA 1.29 / MQA 1.29            | <br /> | <br />         |
| d05\_infini\_attention.py           | `... d05_infini_attention.py`                            | 长程余弦相似：全量 vs 压缩记忆 0.71 vs 仅段内 0.38；显存 9216→1600 元素                                                 | <br /> | <br />         |
| d06\_ssm\_recursion.py              | `... d06_ssm_recursion.py`                               | 递推==卷积 误差 2e-16；选择性门控 15 步后残留                                                                      | h      | 0.017 vs 1.186 |
| d07\_three\_stage\_tiny\_gpt.py     | `... --pt-steps 200 --sft-steps 150 --rl-rounds 3`       | base=乱续写 / sft=`a amy is near the pond` / rl 更贴 reward；compliance SFT0.21→RL0.75                   | <br /> | <br />         |
| d08\_scaling\_law\_calculator.py    | `... --sizes 32,48,64,96 --steps 150`                    | 自训 loss 1.43→0.99；7B→D=140B tokens；6ND 口算数已修正（7B=5.88e21，70B=5.88e23 FLOPs）                        | <br /> | <br />         |
| d09\_moe\_top2.py                   | `... d09_moe_top2.py --steps 200`                        | Dense val1.27 vs MoE top2/8 val0.86~~0.98；aux 使专家使用由失衡(11k~~45k)趋均衡；激活占比25%                        | <br /> | <br />         |
| d10\_catastrophic\_forgetting.py    | `python ... d10 ...`(默认)                                 | 基线掉 25.9pp / replay 掉 17.8pp / EWC 掉 4.7pp(B 略降，tradeoff 可讲)                                       | <br /> | <br />         |
| d11\_lora\_from\_scratch.py         | `... d11 ... --steps 150`                                | B=0 断言 max\|Δ\|=0；可训练占比 full100% / LoRA-attn3.5% / ffn4.4%；W 梯度=0 验证 True                          | <br /> | <br />         |
| d12\_rlhf\_bandit\_ppo.py           | `... d12 ... --iters 400`                                | λ=0: reward0.70/KL5.65(跑偏)；λ=0.8: reward0.48/KL0.86；PPO clip 数值表完整                                 | <br /> | <br />         |
| d13\_dpo\_family.py                 | `... d13 ... --warm-steps 120 --train-steps 180`         | 偏好差距Δ：DPO+3.84 / SimPO+1.92 / ORPO+0.51；win-rate 0.68\~0.72                                        | <br /> | <br />         |
| d14\_grpo\_rlvr.py                  | `... d14 ... --warm 450 --rounds 12`                     | 打印每组 rewards/组内 advantage；pass\@1 SFT0.44→GRPO0.47；verifier=规则打分(前缀部分给分)                           | <br /> | <br />         |
| d15\_reward\_hacking.py             | `... d15 ... --iters 400`                                | hackable 真质量 首段\~→−0.20（被打洞）；shaped →+0.14（修复）                                                     | <br /> | <br />         |
| d16\_kv\_cache.py                   | `... d16 ... --gen-len 100`                              | 总耗时 120ms vs 35ms；每步 0.93 vs 0.37ms；KV公式：7B/GQA8 在 4K≈0.47GB、32K≈3.76GB；PagedAttention block table | <br /> | <br />         |
| d17\_quantization.py                | `python ... d17 ...`                                     | 位宽8/6/4/3 MSE 升 3 个量级；per-channel < per-tensor；全线性层 int4 后 logits 漂移0.47                           | <br /> | <br />         |
| d18\_speculative\_decoding.py       | `... d18 ... --k 4 --tokens 160`                         | 前向次数 160→57(2.8×)、耗时 168→72ms；接受率α0.46；分布一致性 JS≈0.004                                              | <br /> | <br />         |
| d19\_sampling\_strategies.py        | `... d19 ... --len 60 --samples 3`                       | distinct-2 greedy0.61~~topp0.60~~beam0.67；top-p 截断图；重复率 \~0.2                                      | <br /> | <br />         |
| d20\_dp\_tp\_pp.py                  | `python ... d20 ...`                                     | DP 通信量 14\~27.6GB(随P)；PP bubble P4/m8=27%、P8/m32=18%；调度图利用率 67% 与公式一致                              | <br /> | <br />         |
| d21\_zero\_memory\_accounting.py    | `python ... d21 ...`                                     | 7B 模型状态 112GB；P=8: ZeRO1 38.5 / Z2 26.2 / Z3 14.0 GB/卡                                             | <br /> | <br />         |
| d22\_train\_memory\_calculator.py   | `python ... d22 ...`                                     | 7B+激活 ≈129GB；速查 7B/13B/70B=112/208/1120GB                                                          | <br /> | <br />         |
| d23\_ring\_allreduce\_simulation.py | `python ... d23 ...`                                     | reduce-scatter 各节点=列和一致；all-gather 全等 True；发送量 2(P-1)/P·M=1.5M\@P4                                 | <br /> | <br />         |
| d24\_flash\_attention\_tiled.py     | `python ... d24 ...`                                     | 分块 vs 朴素 误差 3.6e-15；HBM 写入 16384 vs 2048                                                           | <br /> | <br />         |
| d25\_synthetic\_data\_mix.py        | `... d25 ... --steps 150 --collapse 1`                   | real1.36 / 70:30=1.34 / syn100=1.42；自训练 3 代 loss 1.42→2.16→2.49（崩溃）                                | <br /> | <br />         |
| d26\_data\_pipeline.py              | `python ... d26 ...`                                     | 787→784→737 漏斗；输出配比 jsonl                                                                          | <br /> | <br />         |
| d27\_eval\_pollution.py             | `... d27 ... --steps 400`                                | test answer-CE 1.64→0.08(背下) vs fresh 仍≥1.55(没真会) → 泄漏铁证                                           | <br /> | <br />         |

## 二、理论课条目（无代码，面试直讲；讲解要点已在相关文档/表格）

- 考点18 DAPO/URPO/MetroRLHF：GRPO 的工程修补 + loss diff 讲解（可结合 `d14` 的 GRPO 代码读 diff）

- 考点23 Prefill-Decode 分离 + D=128/G−1：见《跑全Demo规划》/《Demo详细设计》row25 的讲解稿要点（无运行）

- 考点32 后训练扩展律：论文图表精读（无运行）

- 考点33 NSP/原生多模态 vs 拼接/具身智能：架构框图讲解（无运行）

## 三、GPU(A800) 待测项 → 见 gpu/README.md

`g01`(7B QLoRA QKV-vs-FFN)、`g02`(7B GRPO+RLVR)、`g03`(13B nf4 + AWQ)。
执行后把 显存/时长/指标 回填 gpu/README.md 表格即完成全部验收。

## 四、复现提示

- 想得到"更强/更典型"曲线：调大对应脚本的 `--steps / --sizes / --rounds`（默认值已平衡时间与效果）；

- 换真实数据：把 `common/corpus.make_plain(...)` 换成 `load_text_file('xx.txt')`（char 级，需先小写+过滤）；

- 完整跑一遍：逐个执行第一部分命令即可，除 d07/d13/d14/d25/d27 外大多 <30s（沙箱 2 核下）。

