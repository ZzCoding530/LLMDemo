# -*- coding: utf-8 -*-
"""考点14 RLHF 三阶段 + PPO 关键机制（bandit 版策略梯度 + KL 惩罚 + clip 目标）
类型：演示实现（轻量 torch/纯概率 toy，CPU 秒级-分钟级；真实四模型 PPO 见表格说明）

讲解思路（备课用）：
1. RLHF 三阶段：SFT → 训练 RM(给(y|x)打分) → RL(PPO 最大化 reward，受 KL 约束别跑太远)；
2. 手写核心：策略梯度 L = -E[R·logπ]；直接用 R 会让 π 快速偏离 SFT 模型(π_ref)，
   于是加 KL 惩罚：L = -E[R·logπ] + λ·KL(π‖π_ref)。λ=0 时 reward 涨但 KL 猛增(跑偏)；
   λ>0 时在两者间平衡——本 demo 画 reward 与 KL 两条曲线对照；
3. PPO clip surrogate：r_t(θ)=π_θ/π_old，L=min(r·A, clip(r,1-ε,1+ε)·A)，防止单步更新过大；
   附一张数值表：对一批 (r,A) 手算 clipped 目标，直观看到"太大/太小的 r 被截断"。
运行：python d12_rlhf_bandit_ppo.py [--iters 600] [--lamb 0.3]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def reward_of(a, true_q):
    """可被 exploit 的奖励：真质量 true_q 决定长期指标；模型只能看到 reward 函数。"""
    return true_q[a] + 0.3


def make_env(K=200, seed=0):
    rng = np.random.default_rng(seed)
    # 每个 action 有隐藏"真质量"：多数平庸，少数真正好(agent 不可见)
    true_q = rng.beta(2, 5, size=K)
    return true_q


def rollout(theta, n):
    """从策略采样 n 个 action。"""
    p = F.softmax(theta, -1)
    return torch.multinomial(p, n, replacement=True)


def logp(theta, a):
    return F.log_softmax(theta, 0)[a]


def kl_between(theta, theta_ref):
    p = F.softmax(theta, 0)
    pr = F.softmax(theta_ref.detach(), 0)
    return (p * (p.log() - pr.log())).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lamb", type=float, default=0.8)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()

    K = 200
    true_q = torch.from_numpy(make_env(K)).float()
    # π_ref = SFT 快照：略微偏好某些 action（相当于 SFT 学到的先验）
    torch.manual_seed(0)
    theta_ref = torch.randn(K) * 0.3
    theta_ref[10:30] += 1.0

    def run(lamb, iters):
        theta = theta_ref.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=5e-2)
        hist_r, hist_kl = [], []
        for _ in range(iters):
            acts = rollout(theta, 64)
            rw = true_q[acts] + 0.3
            rw = (rw - rw.mean()) / (rw.std() + 1e-6)     # advantage 归一化
            lp = logp(theta, acts)
            loss = -(lp * rw).mean() + lamb * kl_between(theta, theta_ref)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                hist_r.append(float((true_q * F.softmax(theta, -1)).sum()))
                hist_kl.append(float(kl_between(theta, theta_ref).mean()))
        return theta, hist_r, hist_kl

    # λ=0：无 KL，reward 涨但跑偏；λ>0：平衡
    theta0, r0, kl0 = run(0.0, args.iters)
    thetal, rl, kll = run(args.lamb, args.iters)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xs = np.arange(args.iters)
    axes[0].plot(xs, r0, label="KL penalty=0")
    axes[0].plot(xs, rl, label=f"KL penalty={args.lamb}")
    axes[0].set_title("expected true reward (scores rise)")
    axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(xs, kl0, label="KL penalty=0")
    axes[1].plot(xs, kll, label=f"KL penalty={args.lamb}")
    axes[1].set_title("KL(policy||sft-ref) divergence")
    axes[1].legend(); axes[1].grid(alpha=.3)
    print(f"无KL:  reward={r0[-1]:.3f} KL={kl0[-1]:.3f}  |  λ={args.lamb}: reward={rl[-1]:.3f} KL={kll[-1]:.3f}")
    print("对照：λ=0 时 reward 冲高但 KL 爆炸=策略跑偏；λ>0 抑制跑偏。")

    # ---- PPO clip 数值表 ----
    eps = args.eps
    A = torch.tensor([1.0, 1.0, -1.0, -0.5, 0.2, 2.0])
    ratio = torch.tensor([0.5, 1.5, 0.8, 2.5, 1.1, 0.2])
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    obj = torch.min(ratio * A, clipped * A)
    print(f"\nPPO clip 数值（ε={eps}）：")
    print(f"{'ratio r':>8} {'adv A':>6} {'r·A':>6} {'clip(r)':>7} {'clip·A':>7} {'min':>7}")
    for r_, a, c, o in zip(ratio.tolist(), A.tolist(), clipped.tolist(), obj.tolist()):
        print(f"{r_:8.2f} {a:6.1f} {r_*a:6.2f} {c:7.2f} {c*a:7.2f} {o:7.2f}")
    axes[2].bar(np.arange(len(ratio)) - .2, (ratio * A).numpy(), .4, label="r*A")
    axes[2].bar(np.arange(len(ratio)) + .2, obj.numpy(), .4, label="clipped objective")
    axes[2].set_title("PPO clip objective: step size is capped")
    axes[2].legend(); axes[2].grid(alpha=.3)
    viz.save(fig, "d12_rlhf_bandit_ppo.png")
    print("\n结论：RLHF 的核心矛盾=最大化奖励 vs 别离 SFT 太远；KL 惩罚与 PPO clip 就是两道刹车。")


if __name__ == "__main__":
    main()
