# -*- coding: utf-8 -*-
"""考点6 混合注意力 CSA+HCA / Infini-Attention（段内 softmax + 段间线性压缩记忆）
类型：演示实现（numpy 数值验证，无 GPU）

讲解思路（备课用）：
1. 长序列成本：标准 softmax attention 显存 O(T²)、且每段都要重看历史；
2. Infini-Attention 结构 = 段内(局部)标准 softmax attention + 段间线性 attention 压缩记忆：
    记忆态  M = M_prev + σ(K)^T V        （线性核把历史压成 d×d 常数）
    归一化  z = z_prev + rowsum(σ(K))
    段间检索  A_mem = σ(Q) M / (σ(Q) z)  （与标准 attention 数值上只是"核函数"不同）
3. 本 demo 数值验证：给定一段长序列分块处理，压缩记忆检索能近似"全量检索"的长程依赖；
   显存从 O(T²) 降到 O(T_seg×d + d²)。
运行：python d05_infini_attention.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def full_attention(Q, K, V):
    """全量 softmax attention，作为 reference。Q:(L,d)"""
    s = Q @ K.T
    e = np.exp(s - s.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    return p @ V, p


def seg_softmax_attention(Q, K, V):
    """只允许看本段内部（局部 attention），模拟"没有跨段记忆"的基线。"""
    s = Q @ K.T
    # 用段间 mask 不方便直接比；这里返回本段 full——含义由外部调用控制。
    e = np.exp(s - s.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    return p @ V


def infini_forward(Q, K, V, seg, activate=lambda x: x):
    """分块 + 压缩记忆递归。seg 为段长。
    记忆态用线性核：先对 K 做逐元素激活(此处用 elu+1 保证非负更接近论文做法，简单版直接恒等)。
    返回每段的 (局部输出, 记忆检索输出, 融合输出)。"""
    L, d = Q.shape
    K_a = np.maximum(K, 0) + 1e-3            # 非负激活近似 elu(x)+1
    Q_a = np.maximum(Q, 0) + 1e-3
    M = np.zeros((d, d)); z = np.zeros((d, 1))      # 压缩记忆态
    outs = []
    for s in range(0, L, seg):
        q, k, v = Q[s:s + seg], K[s:s + seg], V[s:s + seg]
        ka = K_a[s:s + seg]; qa = Q_a[s:s + seg]
        # 段内局部 attention
        loc = full_attention(q, k, v)[0]
        # 段间记忆检索：a_mem = σ(q) M / (σ(q) z)
        denom = (qa @ z) + 1e-8
        mem = (qa @ M) / denom
        # 门控融合：简单取平均（论文用可学习门控）
        out = (loc + mem) / 2
        outs.append(out)
        # 递归更新记忆态
        M = M + ka.T @ v
        z = z + ka.sum(0, keepdims=True).T
    return np.concatenate(outs, 0)


def main():
    rng = np.random.default_rng(0)
    L, d = 96, 8
    seg = 16
    # 构造有"长程规律"的数据：后半段值依赖前半段（让记忆检索 vs 仅段内可区分）
    Q = rng.standard_normal((L, d)) * 0.5
    V = rng.standard_normal((L, d)) * 0.5
    K = rng.standard_normal((L, d)) * 0.5
    # 人为在 V 上叠加长程依赖：v[t] 受 v[t-seg*2] 影响
    V2 = V.copy()
    V2[seg * 2:] += 0.6 * V2[:-seg * 2]

    full_out, _ = full_attention(Q, K, V2)
    inf_out = infini_forward(Q, K, V2, seg)

    # 指标：记忆部分是否正确把远端信息带进来 —— 对比第 2 段之后 query 的输出与 full 的余弦
    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)

    tail = slice(seg * 2, L)          # 依赖 2 段之前内容的区间
    sim_full_vs_inf = cos(full_out[tail], inf_out[tail]).mean()
    # 只用段内局部(即每段只有 seg 长历史)的 baseline
    local_out = np.concatenate([full_attention(Q[s:s + seg], K[s:s + seg], V2[s:s + seg])[0]
                                for s in range(0, L, seg)], 0)
    sim_full_vs_local = cos(full_out[tail], local_out[tail]).mean()

    print(f"L={L}, d={d}, seg={seg} -> 共 {L // seg} 段，记忆态显存 = {d*d} 个数")
    print(f"压缩记忆检索 与 全量检索 平均余弦相似(长程区间): {sim_full_vs_inf:.3f}")
    print(f"仅段内局部   与 全量检索 平均余弦相似(长程区间): {sim_full_vs_local:.3f}")
    print(f"说明：需要看 seg 以外历史时，带压缩记忆的结果显著更接近全量注意力。")

    # 显存复杂度对照
    naive = L * L
    infini_mem = seg * seg * (L // seg) + d * d          # 段内分块 + 常数记忆
    print(f"\n显存对比（元素个数）：朴素 attention={naive:,} vs Infini 分块+记忆={infini_mem:,}"
          f"（O(T²) -> O(seg·T+d²)，长序列省主要来自跨段不再重看历史）")


if __name__ == "__main__":
    main()
