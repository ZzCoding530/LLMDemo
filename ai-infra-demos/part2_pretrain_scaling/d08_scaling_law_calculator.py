# -*- coding: utf-8 -*-
"""考点9 Scaling Law：loss~N 幂律拟合 + Chinchilla 1:20 + 6ND 算力公式
类型：演示实现（自训多尺寸 tiny 模型复现趋势 + 公开结论公式计算器，无 GPU）

讲解思路（备课用）：
1. 经验规律：固定数据量下 loss ≈ a·N^(-α)+c；log-log 上是直线（幂律）；
2. Chinchilla(2022)：compute 最优不是"越大越好"，最优配比约 1 参数:20 token；
   过度训练小模型比欠训练大模型更好 → 引出 compute-optimal；
3. 算力估计公式 6ND：每个 token 前向 ~2N FLOPs、反向 ~4N FLOPs（约 6N），共 6·N·D；
   例：7B×140B token ≈ 5.88e24 FLOPs ≈ 5880 PFLOPS·天（A100 打一天≈5000）→ 常考口算；
4. 涌现能力：部分是"指标非线性/超线性"现象（也有争议：连续指标下并不涌现）。
   本脚本用自训的 4 个不同尺寸模型在相同 token 预算下画 loss-N 曲线验证幂律趋势。
运行：python d08_scaling_law_calculator.py [--sizes 32,48,64,96] [--steps 180]
"""
import argparse
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream, train_lm


# ------------------- 6ND 计算器 -------------------
def flops_estimate(N, D=None, ratio=20):
    """N:参数量; D:tokens; 默认按 Chinchilla 最优比 D≈20N。返回 (D, FLOPs)。"""
    D = D or ratio * N
    return D, 6.0 * N * D


def print_calculator():
    print("===== 6ND 算力计算器（现场口算模板）=====")
    for N in [1e9, 7e9, 13e9, 70e9]:
        D, f = flops_estimate(N)
        print(f"N={N/1e9:>4.0f}B -> D_opt≈{D/1e9:>5.0f}B tokens;  FLOPs≈6ND = {f:.2e}")
    print("例如：7B@140B tokens → 6·7e9·1.4e11 ≈ 5.88e21 FLOPs；"
          "70B@1.4T tokens → 6·7e10·1.4e12 ≈ 5.88e23 FLOPs（常考现场口算）")


# ------------------- 自训多尺寸验证 -------------------
def self_curve(sizes, steps, block=64):
    text = make_plain(1500, seed=31)
    val_text = make_plain(150, seed=32)
    tr = PretrainStream(text)
    va = PretrainStream(val_text, seed=88)
    points = []
    for d in sizes:
        cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=block, n_layer=2,
                        n_head=4, n_embd=d)
        torch.manual_seed(0)
        model = TinyGPT(cfg)
        npar = sum(p.numel() for p in model.parameters())
        _, vh = train_lm(model, tr, steps=steps, bs=16, val_data=va, log=False)
        loss = vh[-1][1]
        points.append((npar, loss))
        print(f"  d_model={d:3d}  params={npar:>8,}  val_loss={loss:.4f}")
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="32,48,64,96")
    ap.add_argument("--steps", type=int, default=180)
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    print_calculator()
    print("\n===== 自训多尺寸 tiny 模型（相同 token 预算）=====")
    points = self_curve(sizes, args.steps)

    import matplotlib.pyplot as plt
    ns = np.array([p[0] for p in points]); ls = np.array([p[1] for p in points])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ns, ls, "o-")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("params N (log)"); axes[0].set_ylabel("val loss (log)")
    axes[0].set_title("self-measured: loss drops as N grows (power law)")
    axes[0].grid(alpha=.3, which="both")
    # 幂律拟合 L = a N^(-α)：log L = log a - α log N
    A = np.polyfit(np.log(ns), np.log(ls), 1)
    xx = np.linspace(ns.min(), ns.max(), 50)
    axes[0].plot(xx, np.exp(A[1]) * xx ** A[0], "--", label=f"fit slope α={-A[0]:.2f}")
    axes[0].legend()

    # 理论示意：不同计算预算 C=6ND 下 loss(N,D) 等高线式的 Pareto 前沿示意
    D_grid = np.logspace(8, 10, 200)
    for N in [3e6, 1e7, 3e7]:
        L = (N / 1e7) ** (-0.5) * (D_grid / 1e9) ** (-0.5)   # 示意 shape
        axes[1].plot(D_grid, L, label=f"N={N/1e6:.0f}M")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("tokens D (log)"); axes[1].set_ylabel("loss (log)")
    axes[1].set_title("illustration: more data & params -> lower loss")
    axes[1].legend(); axes[1].grid(alpha=.3, which="both")
    viz.save(fig, "d08_scaling_law.png")
    print("\n结论：同 token 预算下 loss 随 N 幂律下降（斜率≈α）；compute 预算固定时应按 "
          "Chinchilla 1:20 配比把 N、D 一起做大，而不是单方面堆参数。")


if __name__ == "__main__":
    main()
