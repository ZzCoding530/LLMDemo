# -*- coding: utf-8 -*-
"""考点27 Ring AllReduce（ReduceScatter + AllGather）时序模拟 + NVLink vs PCIe
类型：演示实现（Python 模拟分块传递，验证每卡发送量 2(P-1)/P·M；真实跨卡不跑）

讲解思路（备课用）：
1. 为什么不是"主节点广播"：AllReduce 若让 0 号收全部再广播，0 号带宽瓶颈 O(M)；
   ring 把数据切成 P 块沿环传，每卡每步只传 M/P → 通信量均摊且与 P 无关地降到 O(M)；
2. 两阶段：reduce-scatter(P-1 步，边传边加) → all-gather(P-1 步把结果传回每卡)；
   每卡总发送 = 2·(P-1)/P·M；
3. 时间 = 带宽延迟两部分：TP 要用 NVLink(约 400-900GB/s 卡间直连)；
   PP/DP 跨机走 IB/PCIe(慢一个数量级)——决定"TP 同机、PP 跨机"。
运行：python d23_ring_allreduce_simulation.py [--p 4] [--chunks 4]
"""
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def simulate_ring(P, seed=0):
    """经典 ring allreduce：数据切 P 块。
    reduce-scatter：第 s 步，node i 把 chunk c=(i-s-1) mod P 发给 (i+1) mod P，接收方累加。
    结束后每个 node 恰好持有某一块的全局和；all-gather 反向转一圈让每卡拿全所有块。"""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 10, size=(P, P))          # node i, chunk c 的本地值
    print("\n模拟 ring allreduce（P=%d）。初始 x[node,chunk]=" % P)
    print(x)

    # ---- phase1: reduce-scatter ----
    cur = x.astype(float).copy()
    for s in range(P - 1):
        nxt = cur.copy()
        for i in range(P):
            c_send = (i - s - 1) % P
            recver = (i + 1) % P
            nxt[recver, c_send] += cur[i, c_send]   # 把累加和继续往下传
        cur = nxt
    diag = [cur[i, i] for i in range(P)]                 # node i 保留 chunk i 的全局和
    col_sum = x.sum(axis=0)
    got = all(abs(cur[i, i] - col_sum[i]) < 1e-9 for i in range(P))
    print("reduce-scatter 后各 node 持有（node i → chunk i 的全局和）：",
          [f"node{i}={cur[i, i]:.0f}" for i in range(P)], " 与列和一致:", got)

    # ---- phase2: all-gather（chunk c 每步向前传一格，node c+s 持有，每卡留副本）----
    final = np.zeros_like(cur)
    for i in range(P):
        final[i, i] = cur[i, i]
    for s in range(P - 1):
        nxtf = final.copy()
        for j in range(P):
            c = (j - s) % P                   # 第 s 步 node j 持有的"最新"chunk
            nxtf[(j + 1) % P, c] = final[j, c]
        final = nxtf
    ok = np.allclose(final, np.tile(col_sum, (P, 1)))
    print("all-gather 后每行 = 全局各块和：\n", final.astype(int))
    print("验证：与 x.sum(axis=0) 全等 →", bool(ok))
    print(f"每卡发送量 = 2(P-1)/P·M = {2*(P-1)/P:.3f}·M（主节点/树状：主卡要收广播 2M，是瓶颈）")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    args = ap.parse_args()
    ok = simulate_ring(args.p)

    print("\n===== 带宽数量级表 =====")
    rows = [("NVLink 3.0 (A100)", 600), ("NVLink 4.0 (H100)", 900),
            ("PCIe 4.0 x16", 32), ("InfiniBand 200G", 25)]
    print(f"{'链路':<22} {'~带宽GB/s':>10}")
    for n, b in rows:
        print(f"{n:<22} {b:>10}")
    print("→ TP 每层都要通信 → 必须走 NVLink 同机；跨机的 PP/DP 用 IB/以太即可。")

    import matplotlib.pyplot as plt
    P = np.arange(2, 17)
    vol_ring = [2 * (p - 1) / p for p in P]
    vol_bt = [2.0] * len(P)      # tree/主节点：主卡收 P 份再发 P 份 ≈2M（主卡视角）
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(P, vol_ring, "o-", label="ring allreduce (per node)")
    ax.plot(P, vol_bt, "s--", label="tree / master-broadcast (master)")
    ax.set_xlabel("num GPUs P"); ax.set_ylabel("data volume (x M)")
    ax.set_title("ring saturates at 2M, tree master does ~2M but is a bottleneck")
    ax.legend(); ax.grid(alpha=.3)
    viz.save(fig, "d23_ring_allreduce.png")
    print("\n结论：ring 把通信均摊、量级与节点数基本无关；实际瓶颈=延迟与带宽延迟积。")


if __name__ == "__main__":
    main()
