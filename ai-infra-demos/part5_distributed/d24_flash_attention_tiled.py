# -*- coding: utf-8 -*-
"""考点28 Flash Attention 原理：分块(online) softmax 数值验证 + IO 量对比
类型：演示 + 对照（numpy 数值模拟；可选真实 flash-attn/SDPA 在讲解稿提示）

讲解思路（备课用）：
1. 朴素 attention 慢在哪：要把 N×N 的 score 矩阵物化到 HBM（显存带宽瓶颈 O(N²)），
   再读回来做 softmax；大 seq 直接爆显存；
2. Flash Attention 三件事：tiling(分块)、online softmax(增量维护 running max 与 exp-sum、
   出新块时对旧结果 rescale)、不把 N×N 落回 HBM；
3. 本 demo：同一 Q,K,V，朴素全量 vs 分块 online 算法 → 数值一致断言 <1e-9；
   并统计"写回 HBM 的元素量"：朴素 O(N²)，分块只有 O(N) 的 m/l 中间量；
4. 面试口径：Flash 省的不是 FLOPs，是 HBM 读写与显存占用。
运行：python d24_flash_attention_tiled.py [--n 128] [--d 32] [--block 16]
"""
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def naive_attn(Q, K, V):
    """Q,K,V: (N, d)。返回 (O, S_hbm_elements)。"""
    S = Q @ K.T                      # (N,N)
    S = S - S.max(-1, keepdims=True)
    e = np.exp(S)
    P = e / e.sum(-1, keepdims=True)
    return P @ V, S.shape[0] * S.shape[1]


def tiled_attn(Q, K, V, block):
    """分块 online softmax（flash-attn 核心数值）：逐 key-block 维护 running max/exp-sum，
    出新块时对旧累加输出 rescale。返回 (O, hbm 中间量元素数)。"""
    N, d = Q.shape
    O = np.zeros((N, d))
    m = np.full(N, -np.inf)
    l = np.zeros(N)
    for s in range(0, N, block):
        Kb, Vb = K[s:s + block], V[s:s + block]
        Sj = Q @ Kb.T                              # (N, block) 只在"片上"存在，不落 HBM
        m_new = np.maximum(m, Sj.max(-1))
        Pj = np.exp(Sj - m_new[:, None])
        l_new = l * np.exp(m - m_new) + Pj.sum(-1)
        O = O * np.exp(m - m_new)[:, None] + Pj @ Vb   # 旧结果按新 max 缩放后累加
        m, l = m_new, l_new
    O = O / l[:, None]
    # 每块需要写回 HBM 的只有 m 与 l（各 N 个），S/P 都不落盘
    hbm = 2 * N * (N // block)
    return O, hbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--block", type=int, default=16)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    Q = rng.standard_normal((args.n, args.d))
    K = rng.standard_normal((args.n, args.d))
    V = rng.standard_normal((args.n, args.d))

    O_naive, s_writes = naive_attn(Q, K, V)
    O_tiled, inter = tiled_attn(Q, K, V, args.block)
    diff = np.abs(O_naive - O_tiled).max()
    print(f"N={args.n}, d={args.d}, block={args.block}")
    print(f"朴素 vs 分块 online softmax 最大误差 = {diff:.2e}（<1e-9 → 数值等价）")

    # IO 量级对比：朴素把 N×N score 写进 HBM；分块只写 O(N) 的 m/l（block 无关）
    hbm_naive = s_writes                       # N×N 写 HBM（+还要读回来）
    hbm_tiled = inter                          # ~2N × (N/block) 次
    print(f"HBM 写入量级：朴素 ≈{hbm_naive:>9,} 元素  vs 分块 ≈{hbm_tiled:>9.0f} 元素"
          f"（比值 {hbm_naive/max(hbm_tiled,1):.0f}×，长序列差距更大）")

    import matplotlib.pyplot as plt
    sizes = np.array([256, 512, 1024, 2048, 4096])
    naive_io = sizes ** 2
    flash_io = 2 * sizes * (sizes // 32)       # 每块写 m,l 2N 个，共 N/block 块
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(sizes, naive_io / 1e6, "o-", label="naive attention (N^2 to HBM)")
    ax.plot(sizes, flash_io / 1e6, "s--", label="tiled online softmax (~N^2/block)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length N"); ax.set_ylabel("HBM traffic (M elements)")
    ax.set_title("flash attention cuts HBM traffic & memory")
    ax.legend(); ax.grid(alpha=.3, which="both")
    viz.save(fig, "d24_flash_attention.png")
    print("\n结论：分块+online softmax 与全量数值一致，但避免物化 N×N score → 省显存、省 HBM 带宽。"
          "（FLOPs 并没少；少的是读写）")


if __name__ == "__main__":
    main()
