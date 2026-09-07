# -*- coding: utf-8 -*-
"""考点25 DP / TP / PP / 3D 并行：切分方式、通信量、PP 流水线 bubble（模拟脚本）
类型：演示实现（Python 模拟 + 图；真实多卡训练不跑，直讲原理——与表格【不跑】一致）

讲解思路（备课用）：
1. DP 切数据：每卡一份完整模型，各训不同 batch，梯度 AllReduce 求平均。
   通信量/步 ≈ 2×模型字节×(P-1)/P（ring allreduce，见 d23）；
2. TP 切单层：把一个 Linear 按列切成 P 份，各卡只算自己的分片，前/后各一次 AllReduce；
   每层都通信 → 需要机内高带宽(NVLink)，所以 TP 同机；
3. PP 切层：模型按层分成 P 段，数据切成 micro-batch 流水线填满各段；
   流水线有"气泡"：bubble 比例 ≈ (P-1)/(P-1+m)，m 个 micro-batch；
4. 3D = 先 TP(卡内) × PP(机间) × DP(副本)；示例 4×2×4=32 卡。
运行：python d20_dp_tp_pp.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def dp_comm_bytes(P, model_bytes):
    return 2 * model_bytes * (P - 1) / P


def pp_bubble_ratio(P, m):
    return (P - 1) / (P - 1 + m)


def draw_schedule(P, m):
    """画 m 个 micro-batch 在 P 段流水线上的调度（横=时间步，竖=PP rank）。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, m + P), 3.6))
    cells = np.zeros((P, m + P - 1), dtype=int)
    for s in range(m + P - 1):
        for p in range(P):
            mb = s - p
            if 0 <= mb < m:
                cells[p, s] = mb + 1          # 有活干
    for p in range(P):
        for s in range(m + P - 1):
            v = cells[p, s]
            c = "#4C72B0" if v else "#dddddd"
            ax.add_patch(plt.Rectangle((s, P - 1 - p), 1, 1, color=c))
    ax.set_xlim(0, m + P - 1); ax.set_ylim(0, P)
    ax.set_yticks(np.arange(P) + 0.5, [f"stage {i+1}" for i in range(P - 1, -1, -1)])
    ax.set_xticks(np.arange(0, m + P))
    ax.set_title(f"PP pipeline schedule: P={P} stages, m={m} micro-batches")
    ax.set_xlabel("time step")
    viz.save(fig, "d20_pp_schedule.png")
    busy = (cells > 0).sum()
    total = P * (m + P - 1)
    return busy / total


def main():
    print("===== 通信量公式（每步）=====")
    print(f"{'P卡':>4} {'DP allreduce MB(7B bf16≈14GB)':>26}")
    for P in [2, 4, 8, 32, 64]:
        gb = dp_comm_bytes(P, 14e9) / 1e9
        print(f"{P:>4} {gb:>22.1f} GB")

    print("\n===== PP 流水线 bubble =====")
    for P in [4, 8]:
        for m in [4, 8, 16, 32]:
            print(f"P={P}, m={m:>2}: bubble={pp_bubble_ratio(P, m)*100:5.1f}%", end="  ")
        print()

    util = draw_schedule(4, 6)
    print(f"\n调度图(存 outputs/d20_pp_schedule.png)，实际利用率≈{util*100:.0f}%"
          f"（bubble {100-util*100:.0f}%，与公式一致）")

    print("""
===== 3D 组合（举例 32 卡 = TP4 × PP2 × DP4）=====
- TP=4：同一机内 4 卡切每层（需 NVLink），4 卡组成"1 个 TP 组"；
- PP=2：8 卡(2 个 TP 组×4?) 不对——正确：TP×PP×DP 的维度要乘到总卡数；
  例：32 卡 = 2(机)×4(卡/机)=8 卡先做 TP4×PP2 → 得到 4 个完整模型副本 → DP4。
- 规则口诀：TP 切单层(同机带宽) > PP 切层(机间少通信) > DP 切数据(最粗粒度)。
""")
    print("结论：真实 3D 并行需多卡集群训练；面试只考概念+公式。")


if __name__ == "__main__":
    main()
