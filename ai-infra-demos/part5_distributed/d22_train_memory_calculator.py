# -*- coding: utf-8 -*-
"""考点29 训练显存估算计算器：权重+梯度+优化器状态+激活 分解，7B/13B/70B 一键对照
类型：演示实现（交互/预设计算器脚本；可配合 torch 真实加载 7B 实测校准）

讲解思路（备课用）：
1. 训练显存 = 模型状态 + 激活 + 临时量；
2. 模型状态：参数+梯度+优化器状态，精确到字节/参数（d21 讲透 ZeRO 切法）；
3. 激活（不重计算时按 batch×seq 存）：数量级 ~ n_layer×batch×seq×hidden×12 字节(近似)；
   开启 gradient checkpointing 以 sqrt 换时间，激活可压到 ~sqrt(n_layer) 量级；
4. 面试口算模板：7B bf16+AdamW → 模型状态 ≈ 112GB + 激活 ≈ 数十 GB → 单卡 80G 必然不够。
运行：python d22_train_memory_calculator.py [--params-b 7] [--layer 32] [--hidden 4096] [--seq 4096] [--batch 1]
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def model_states_bytes(N, opt="adam", mixed=True):
    """模型状态每参数字节（×N 后为总字节）：
    bf16 混合精度+AdamW：参数 2B + 梯度 2B + Adam(master 4 + m 4 + v 4)=12B → 16B/参数。"""
    per_param = (2 + 2 + 12) if (opt == "adam" and mixed) else 16
    return per_param * N


def activations_bytes(n_layer, batch, seq, hidden, checkpoint=False, heads=32, L2=None):
    """每层激活量级近似：B·T·(约 12×hidden? 多头打分额外)。
    经验估计：不重计算时训练激活≈ n_layer×batch×seq×(hidden×~34 + ...) 浮动大，
    这里用一个常用粗估：每 token 每层存 ~ (34·hidden + 2·heads·T) 字节？复杂，采用保守公式：
    act_bytes ≈ n_layer·batch·seq·(16·hidden·4B)  fp16。checkpoint 后按层数平方根折算。"""
    per_token_layer = 16 * hidden * 2          # 经验粗估(byte/token/layer)
    base = n_layer * batch * seq * per_token_layer
    if checkpoint:
        base = base / n_layer ** 0.5 * 2        # 激活只存部分层(以重计算换显存)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-b", type=float, default=7)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    print("===== 训练显存计算器（模型状态 + 激活）=====")
    print(f"配置：{args.params_b:.0f}B 参数, {args.layer} 层, hidden={args.hidden}, "
          f"seq={args.seq}, batch={args.batch}, bf16+AdamW")
    N = args.params_b * 1e9
    ms = model_states_bytes(N)                 # baseline 不分片
    act = activations_bytes(args.layer, args.batch, args.seq, args.hidden,
                            checkpoint=False)
    act_ckpt = activations_bytes(args.layer, args.batch, args.seq, args.hidden,
                                 checkpoint=True)
    total = ms + act
    print(f"\n模型状态(不分片) ≈ {ms/1e9:7.1f} GB  "
          f"(参数 {2*N/1e9:.1f} + 梯度 {2*N/1e9:.1f} + Adam {12*N/1e9:.1f})")
    print(f"激活(粗估)         ≈ {act/1e9:7.1f} GB   grad-checkpoint 后 ≈ {act_ckpt/1e9:.1f} GB")
    print(f"合计              ≈ {total/1e9:7.1f} GB")

    print("\n===== 常见口径速查（模型状态/bf16/AdamW, 不含激活）=====")
    for pb in [7, 13, 70]:
        gb = 16 * pb
        print(f"{pb:>3}B → 模型状态≈{gb}GB：单卡80G {'装不下(需ZeRO/量化/卸载)' if gb > 80 else '勉强可放'}")

    print("\n面试结论模板：先报模型状态≈16N(混合精度+Adam)，再加激活(看batch×seq)；"
          "7B 全参训练单卡必然爆 → ZeRO/FSDP 分片或 QLoRA 降精度。")


if __name__ == "__main__":
    main()
