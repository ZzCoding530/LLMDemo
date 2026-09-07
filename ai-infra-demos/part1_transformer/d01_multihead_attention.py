# -*- coding: utf-8 -*-
"""考点1·2 Self-Attention / MHA / 为什么除以 √d_k / QKV 投影 / 多头意义
类型：演示 + 轻量对照（纯 numpy / CPU）

讲解思路（备课用，讲稿另出）：
1. Q=Wq·x, K=Wk·x, V=Wv·x 三个投影，attention = softmax(QK^T/√d_k)·V；
2. 除以 √d_k 的原因：d_k 大时点积方差≈d_k，softmax 输入方差大 → 分布迅速 one-hot
   → 熵趋近 0、梯度趋近 0（梯度消失），训练动不了；
3. 数值实验：固定 Q,K 为 N(0,1)，扫描 d_k∈{8..512}，对比 除/不除 √d_k 的
   熵与最大概率曲线、以及 softmax 输出对输入的灵敏度（可视为梯度量级）；
4. 多头 = 切成 h 份并行 attention：每头在不同子空间学不同"关注模式"，热力图直观可见。
运行：python d01_multihead_attention.py [--dk-list 8,16,32,64,128,256,512] [--seq 12]
"""
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def softmax_rows(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def entropy_norm(p):
    """归一化熵：1=均匀分布；0=one-hot。"""
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1) / np.log(p.shape[-1])


def scaled_dot_experiment(dk_list, seed=0):
    """固定 q,k 独立 N(0,1)：scores = q·k，随 d_k 增大方差变大。"""
    rng = np.random.default_rng(seed)
    rows = []
    for dk in dk_list:
        q = rng.standard_normal((1, dk)) * 1.0      # 每维方差 1
        k = rng.standard_normal((dk, 32))           # 32 个 key
        s = q @ k                                   # (1,32) 方差≈dk
        p_raw = softmax_rows(s)
        p_scaled = softmax_rows(s / np.sqrt(dk))
        # 梯度量级代理：softmax 输出对输入单位扰动的平均变化率
        u = rng.standard_normal(s.shape)
        u /= np.linalg.norm(u) + 1e-12
        eps = 1e-3
        deriv_raw = np.linalg.norm(softmax_rows(s + eps * u) - p_raw) / eps
        deriv_scaled = np.linalg.norm(softmax_rows(s / np.sqrt(dk) + eps * u / np.sqrt(dk)) - p_scaled) / eps
        rows.append(dict(dk=dk,
                         ent_raw=float(entropy_norm(p_raw)[0]), ent_sc=float(entropy_norm(p_scaled)[0]),
                         max_raw=float(p_raw.max()), max_sc=float(p_scaled.max()),
                         deriv_raw=deriv_raw, deriv_sc=deriv_scaled))
    return rows


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_head, scale=True):
    """numpy MHA 手撕（面试可默写版本）。x:(B,L,d); W*:(hd*?, d)"""
    B, L, d = x.shape
    dh = d // n_head
    Q = x @ Wq.T            # (B,L,d)
    K = x @ Wk.T
    V = x @ Wv.T
    # 切头： (B, n_head, L, dh)
    def split(z): return z.reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
    Q, K, V = split(Q), split(K), split(V)
    s = Q @ K.transpose(0, 1, 3, 2)          # (B,H,L,L)
    s = s / np.sqrt(dh) if scale else s
    # 因果 mask 可选：这里演示双向即可
    p = softmax_rows(s)
    o = p @ V                                # (B,H,L,dh)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, d)
    return o @ Wo.T, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk-list", default="8,16,32,64,128,256,512")
    ap.add_argument("--seq", type=int, default=12)
    args = ap.parse_args()
    dk_list = [int(x) for x in args.dk_list.split(",")]

    # ---- 实验1：除/不除 √d_k ----
    rows = scaled_dot_experiment(dk_list)
    print(f"{'d_k':>5} | {'熵(除√)':>8} {'熵(不除)':>8} | {'maxP(除√)':>8} "
          f"{'maxP(不除)':>8} | {'梯度量级(除√)':>12} {'(不除)':>10}")
    for r in rows:
        print(f"{r['dk']:>5d} | {r['ent_sc']:8.3f} {r['ent_raw']:8.3f} | {r['max_sc']:8.3f} "
              f"{r['max_raw']:8.3f} | {r['deriv_sc']:12.4f} {r['deriv_raw']:10.4f}")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    dks = [r["dk"] for r in rows]
    axes[0].plot(dks, [r["ent_sc"] for r in rows], "o-", label="scaled /sqrt(d_k)")
    axes[0].plot(dks, [r["ent_raw"] for r in rows], "s--", label="raw QK^T")
    axes[0].set_xscale("log"); axes[0].set_xlabel("d_k"); axes[0].set_ylabel("normalized entropy")
    axes[0].set_title("entropy: scaled stays high (softmax not saturated)")
    axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(dks, [r["max_sc"] for r in rows], "o-", label="scaled")
    axes[1].plot(dks, [r["max_raw"] for r in rows], "s--", label="raw")
    axes[1].set_xscale("log"); axes[1].set_xlabel("d_k"); axes[1].set_ylabel("max softmax prob")
    axes[1].set_title("raw softmax -> one-hot as d_k grows"); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(dks, [r["deriv_sc"] for r in rows], "o-", label="scaled")
    axes[2].plot(dks, [r["deriv_raw"] for r in rows], "s--", label="raw")
    axes[2].set_xscale("log"); axes[2].set_xlabel("d_k"); axes[2].set_ylabel("output sensitivity")
    axes[2].set_title("saturated softmax -> vanishing gradient"); axes[2].legend(); axes[2].grid(alpha=.3)
    viz.save(fig, "d01_sqrt_dk_experiment.png")

    # ---- 实验2：MHA 多头热力图 ----
    d, L, n_head = 16, args.seq, 4
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, L, d))
    # 弱化结构：让某些位置语义相近（同类型词共享粗粒度表示），不同头侧重不同
    Wq = rng.standard_normal((d, d)); Wk = rng.standard_normal((d, d))
    Wv = rng.standard_normal((d, d)); Wo = rng.standard_normal((d, d))
    out, att = multi_head_attention(x, Wq, Wk, Wv, Wo, n_head)
    print(f"\nMHA 输出形状 {out.shape}（应等于输入 (1,{L},{d})）；attention 形状 {att.shape}")
    fig, axes = plt.subplots(1, n_head, figsize=(4 * n_head, 3.4))
    for h in range(n_head):
        im = axes[h].imshow(att[0, h], cmap="viridis", vmin=0, vmax=1)
        axes[h].set_title(f"head {h+1}")
        axes[h].set_xlabel("key position"); axes[h].set_ylabel("query position")
    fig.colorbar(im, ax=axes, fraction=.04)
    fig.suptitle("per-head attention maps (different heads -> different focus patterns)")
    viz.save(fig, "d01_mha_head_maps.png")
    print("结论：除以 √d_k 让 softmax 保持有区分度而非一步 one-hot，梯度才传得动；"
          "多头把 d 维切到 h 个子空间并行，各学各的关注模式。")


if __name__ == "__main__":
    main()
