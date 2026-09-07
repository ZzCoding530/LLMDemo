# -*- coding: utf-8 -*-
"""考点4 位置编码：sin/cos、RoPE、ALiBi、外推与插值
类型：AB 对比实验（numpy 性质验证 + tiny GPT 外推实验，CPU 可跑）

讲解思路（备课用）：
1. sin/cos 绝对位置编码：PE(pos,2i)=sin(pos/10000^{2i/d})，加到输入上；
   优势：周期性让相近位置编码相近；软肋：训练长度外推时表现差；
2. RoPE 旋转位置编码：给 q,k 按位置旋转（q'=R_θ q），使内积 q_i·k_j 只依赖相对位置 i-j，
   天然具备相对位置建模；验证方式：内容相同的 q̃,k̃ 旋转后内积矩阵呈 Toeplitz（只随 |i-j| 变）；
3. ALiBi：不给输入加位置，而是在 attention logits 上减 m·|i-j|（无参数、可外推）；
4. 外推实验：tiny GPT 只在 block=48 训练，分别用 sincos/rope/alibi，到 48/72/96 长度测 loss；
   直接外推会崩 → 引出长度插值(缩放位置)等技巧；
5. Yarn 属于 NTK-aware 插值，只讲思想不实现。
运行：python d03_position_encoding.py [--block 48] [--steps 160]
"""
import argparse
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream, eval_loss


# ---------------- numpy: sin/cos 与 RoPE ----------------

def sincos_pe(L, d):
    pe = np.zeros((L, d))
    pos = np.arange(L)[:, None].astype(float)
    i = np.arange(0, d, 2)
    pe[:, 0::2] = np.sin(pos / 10000 ** (i / d))
    pe[:, 1::2] = np.cos(pos / 10000 ** (i / d))
    return pe


def rope_rotate_vecs(z, pos, base=10000.0):
    """把 (L,d) 的向量按各自 pos 旋转（每两维一组）。"""
    z = z.astype(float)
    zz = z.reshape(len(z), -1, 2)
    d = z.shape[1]
    theta = pos[:, None] / base ** (np.arange(0, d, 2) / d)     # (L, d/2)
    c, s = np.cos(theta), np.sin(theta)
    x0, x1 = zz[..., 0], zz[..., 1]
    out = np.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return out.reshape(len(z), d)


def alibi_bias(L, n_head=4):
    head = np.arange(n_head)[:, None]
    m = 2 ** (-8.0 * head / n_head)
    dist = np.abs(np.arange(L)[None, :] - np.arange(L)[:, None])  # (L,L)
    return -m[..., None, :] * dist[None, ...]                      # (H,L,L)


def rope_relative_property(L=20, d=8, seed=3):
    """内容向量相同、仅位置不同 → 旋转后内积只依赖相对位置。"""
    rng = np.random.default_rng(seed)
    q0 = rng.standard_normal(d); k0 = rng.standard_normal(d)
    pos = np.arange(L)
    q = np.tile(q0, (L, 1)); k = np.tile(k0, (L, 1))
    Q = rope_rotate_vecs(q, pos)
    K = rope_rotate_vecs(k, pos)
    M = Q @ K.T                       # M[i,j] 应只依赖 |i-j|
    # 验证 Toeplitz：M[i,j] vs M[i+t,j+t]
    err = max(abs(M[i, j] - M[i + t, j + t])
              for i in range(L - 1) for j in range(L - 1)
              for t in range(1, min(L - i, L - j)))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-abs(M).max(), vmax=abs(M).max())
    ax.set_title("RoPE: q_i . k_j depends on |i-j| only")
    ax.set_xlabel("key pos j"); ax.set_ylabel("query pos i")
    fig.colorbar(im, ax=ax, fraction=.045)
    viz.save(fig, "d03_rope_relative_heatmap.png")
    print(f"RoPE 相对位置验证：平移不变最大误差 = {err:.2e}（≈0 说明只依赖相对位置）")


def eval_len_loss(model, text, length, block, batches=8, seed=42):
    """在同一语料上按指定长度切窗口测 loss（外推时长度>训练 block）。"""
    data = PretrainStream(text, seed=seed)
    return eval_loss(model, data, batches=batches, bs=6, block=length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--steps", type=int, default=160)
    args = ap.parse_args()
    B = args.block

    # ---- sin/cos 波形示意 + RoPE 相对位置验证 ----
    pe = sincos_pe(100, 16)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    for dim_i in range(4):
        ax.plot(pe[:, dim_i * 2], label=f"dim {dim_i*2}")
    ax.set_title("sin/cos absolute positional encoding (first 4 dims)")
    ax.set_xlabel("position"); ax.legend(); ax.grid(alpha=.3)
    viz.save(fig, "d03_sincos_waves.png")
    rope_relative_property()

    # ---- 外推实验：同架构同数据，仅 PE 不同 ----
    text = make_plain(900, seed=11)
    val_text = make_plain(120, seed=12)
    base_cfg = dict(vocab_size=VOCAB_SIZE, block_size=B, n_layer=2, n_head=4,
                    n_embd=64, dropout=0.0)
    results = {}
    for kind in ["sincos", "rope", "alibi"]:
        print(f"\n>>> 训练 PE={kind} (block={B}, steps={args.steps})")
        torch.manual_seed(0)
        model = TinyGPT(GPTConfig(pos_kind=kind, **base_cfg))
        tr = PretrainStream(text)
        from common.train import train_lm
        train_lm(model, tr, steps=args.steps, bs=12, log=False)
        # 训练长度内 vs 外推
        row = {}
        for L in [B, int(B * 1.5), B * 2]:
            row[L] = eval_len_loss(model, val_text, L, B)
            print(f"  eval length={L:3d}  val_loss={row[L]:.4f}")
        results[kind] = row

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    xs = sorted(results["sincos"].keys())
    for kind, row in results.items():
        ax.plot(xs, [row[x] for x in xs], "o-", label=kind)
    ax.axvline(B, ls="--", color="gray", lw=1)
    ax.text(B, ax.get_ylim()[1] * 0.98, f" train len {B}", color="gray")
    ax.set_xlabel("eval context length"); ax.set_ylabel("val loss")
    ax.set_title("extrapolation: trained at short len, tested longer")
    ax.legend(); ax.grid(alpha=.3)
    viz.save(fig, "d03_extrapolation_loss.png")
    print("\n结论：sin/cos 绝对位置在超训练长度后 loss 快速变差；RoPE 相对位置/ALiBi 更稳。"
          "长度插值 / YaRN(NTK-aware) 是缓解手段。")


if __name__ == "__main__":
    main()
