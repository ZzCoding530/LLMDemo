# -*- coding: utf-8 -*-
"""考点26 ZeRO / FSDP 显存记账：为什么 7B 全参要 8×80G，各 stage 每卡需要多少
类型：演示实现（Python 显存记账 + 曲线；真实跨卡分片不跑，直讲原理）

讲解思路（备课用）：
1. 混合精度 + Adam 下"模型状态"每参数 16 字节：
   fp16 参数 2B + fp16 梯度 2B + Adam(master fp32 4B + m 4B + v 4B) = 16B；
   7B → 模型状态 ≈ 112GB，任何单卡都放不下 → 必须分片或卸载；
2. ZeRO-1 只分片优化器状态；ZeRO-2 再分片梯度；ZeRO-3 连参数也分片（用时 gather）；
   每卡显存随卡数 P 下降公式见代码；FSDP 就是 ZeRO-3 的 PyTorch 实现；
3. Offload：优化器状态扔 CPU 内存 → 显存更低，代价是慢（CPU↔GPU 拷贝）；
4. 结论：单卡放不下不是"不够大"，而是优化器状态爆炸——面试必答口径。
运行：python d21_zero_memory_accounting.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz

# fp16/bf16 参数+梯度各 2B，Adam 状态(master fp32+m+v) 12B → 模型状态 16B/参数
PARAM_B, GRAD_B, OPT_B = 2, 2, 12


def per_gpu_bytes(N, P, stage):
    """返回每卡模型状态字节数。stage: 0=baseline DP, 1,2,3。"""
    if stage == 0:
        return (PARAM_B + GRAD_B + OPT_B) * N
    if stage == 1:
        return PARAM_B * N + GRAD_B * N + OPT_B * N / P
    if stage == 2:
        return PARAM_B * N + (GRAD_B + OPT_B) * N / P
    if stage == 3:
        return (PARAM_B + GRAD_B + OPT_B) * N / P
    raise ValueError


def main():
    import matplotlib.pyplot as plt
    N = 7e9
    print("7B 模型，bf16 训练 + AdamW（不含激活）")
    print(f"{'ZeRO stage':<12} {'P':>3} {'每卡模型状态':>12} {'×P 总量':>12}")
    Pgrid = [1, 2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for stage in [0, 1, 2, 3]:
        gb = [per_gpu_bytes(N, P, stage) / 1e9 for P in Pgrid]
        ax.plot(Pgrid, gb, "o-", label=f"ZeRO-{stage}" if stage else "DP baseline")
        if stage == 0:
            print(f"{'DP baseline':<12} {'1':>3} {gb[0]:>10.1f} GB")
        else:
            print(f"{'ZeRO-'+str(stage):<12} {'1':>3} {gb[0]:>10.1f} GB "
                  f"| P=8: {gb[Pgrid.index(8)]:.1f} GB/卡")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("num GPUs P"); ax.set_ylabel("model-state GB per GPU")
    ax.set_title("7B mixed-precision Adam: model states per GPU")
    ax.legend(); ax.grid(alpha=.3, which="both")
    viz.save(fig, "d21_zero_accounting.png")

    # Offload 示意（优化器状态到 CPU）
    cpu = OPT_B * N / 1e9
    gpu = (PARAM_B + GRAD_B) * N / 1e9
    print(f"\nOffload(ZeRO-3 + offload optimizer)：GPU 只留 参数+梯度 ≈{gpu:.0f}GB，"
          f"优化器状态 ≈{cpu:.0f}GB 放 CPU 内存 → 单卡 80G 也能微调 7B（代价是速度）")
    print("\n结论：全参微调 7B 用 8×80G 是因为单卡装不下 16B/参数×7B≈112GB 的模型状态；"
          "ZeRO 按 stage 依次切 优化器状态/梯度/参数，FSDP=ZeRO-3 工程实现。")


if __name__ == "__main__":
    main()
