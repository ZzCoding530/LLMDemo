# -*- coding: utf-8 -*-
"""考点7 后 Transformer：一维选择性 SSM（Mamba 谱系核心）+ 递推==卷积验证
类型：演示实现（numpy 数值验证，可选极简对比）

讲解思路（备课用）：
1. 一维 SSM 递推：h_t = A·h_{t-1} + B·x_t,  y_t = C·h_t；
   展开可得 y_t = Σ_k C·A^k·B·x_{t-k} —— 与"输入 × 固定脉冲响应"的卷积完全等价（LTI 情形）；
2. 用这个等价性可以做并行训练（卷积/扫除），推理仍是常数状态递推 —— 显存/状态 O(状态维) 与长度无关；
   对比 attention 的 O(T²) 打分矩阵，长序列省显存省计算；
3. 选择性(Mamba 的关键)：让 A/B/C 依赖输入 x_t，才能"该记则记、该忘则忘"；
   代价：不再是 LTI，不能直接卷积，需要硬件级 scan 优化（面试常问取舍点）；
4. 一维 toy 数值上验证：(a)递推==卷积；(b)门控选择性让它学会"复制最近一段/忽略前缀"。
运行：python d06_ssm_recursion.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def ssm_recurrent(A, B, C, x):
    """标量 SSM 逐时间步递推。A,B,C 标量或 (T,)；x:(T,)"""
    T = len(x)
    h = np.zeros(T)
    y = np.zeros(T)
    for t in range(T):
        if t == 0:
            h[t] = B * x[0]
        else:
            h[t] = A * h[t - 1] + B * x[t]
        y[t] = C * h[t]
    return y, h


def ssm_convolution(A, B, C, x):
    """LTI 情形：y = 输入 与 脉冲响应 C·A^k·B 的卷积（前向累积）"""
    T = len(x)
    impulse = np.array([C * (A ** k) * B for k in range(T)])
    y = np.array([np.dot(impulse[:t + 1][::-1], x[:t + 1]) for t in range(T)])
    return y


def copy_task(alpha=0.9, T=30, seed=0):
    """验证选择性：信号在 [0,10) 有内容，之后输入全 0。
    - 恒定门控(非选择性)：记忆会指数衰减 → 远端内容被遗忘；
    - 输入相关门控(选择性)：gate 在无输入时仍"保持"(A≈1) → 能长程记住。
    用一个 toy：B 依赖 |x|，A 依赖 gate 实现 keep/forget。"""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    x[:10] = rng.standard_normal(10)
    # 非选择性固定参数
    y_fixed, h_fixed = ssm_recurrent(alpha, 1.0, 1.0, x)
    # 选择性：A_t = 0.999（无输入时近乎保持），有输入时 A=0.5 快速吸收
    Asel = np.where(np.abs(x) > 1e-6, 0.5, 0.999)
    h = np.zeros(T); y_sel = np.zeros(T)
    for t in range(T):
        if t == 0:
            h[t] = x[0]
        else:
            h[t] = Asel[t] * h[t - 1] + x[t]
        y_sel[t] = h[t]
    # 在第 25 步看"还记得 t=0..9 的内容吗"：用 h 里保有的能量近似
    e_fixed = np.abs(h_fixed[25]); e_sel = np.abs(h[25])
    return y_fixed, y_sel, e_fixed, e_sel


def main():
    # ---- 1) 递推 == 卷积 ----
    rng = np.random.default_rng(1)
    A, B, C = 0.7, 1.2, 0.9
    x = rng.standard_normal(20)
    y_rec, _ = ssm_recurrent(A, B, C, x)
    y_conv = ssm_convolution(A, B, C, x)
    print(f"递推 vs 卷积 最大误差 = {np.abs(y_rec - y_conv).max():.2e}（应≈0 → LTI 时二者等价）")

    # ---- 2) 选择性：记/忘 ----
    y_fixed, y_sel, e_fixed, e_sel = copy_task()
    print(f"\n选择性 toy：15 步后隐藏状态里残留的早期信息 |h| = "
          f"固定门控 {e_fixed:.3f}  vs 选择性门控 {e_sel:.3f}")
    print("固定门控指数衰减->远端遗忘；输入相关门控在无输入时保持记忆->长程可用。")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    t = np.arange(20)
    axes[0].plot(t, y_rec, "o-", label="recurrent")
    axes[0].plot(t, y_conv, "s--", label="convolution")
    axes[0].set_title("SSM: recurrent == convolution (LTI)")
    axes[0].legend(); axes[0].grid(alpha=.3)
    # 记忆衰减图
    A_arr = np.array([A ** k for k in range(30)])
    axes[1].semilogy(A_arr, "o-")
    axes[1].set_title("fixed-gate impulse response decays")
    axes[1].set_xlabel("steps back"); axes[1].grid(alpha=.3)
    # 复制任务输出
    axes[2].plot(y_fixed, "o-", label="fixed gate (forgets)")
    axes[2].plot(y_sel, "s--", label="selective gate (keeps)")
    axes[2].axvline(10, color="gray", ls=":")
    axes[2].set_title("selective SSM keeps far info")
    axes[2].legend(); axes[2].grid(alpha=.3)
    viz.save(fig, "d06_ssm_toy.png")

    # ---- 3) 复杂度对比 ----
    T = 10 ** 5
    print(f"\n复杂度对比（T={T:,}）：attention 打分矩阵显存≈{T*T/1e9:.1f}G 元素 "
          f"(O(T²))；SSM 状态为常数 d²≈O(1)，推理每步只做 d 维递推。")


if __name__ == "__main__":
    main()
