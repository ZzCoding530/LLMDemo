# -*- coding: utf-8 -*-
"""考点11 灾难性遗忘：顺序学习 A→B 后 A 能力暴跌 + 经验回放 / EWC 缓解
类型：AB 对比实验（torch MLP + 合成双任务，CPU 快速）

讲解思路（备课用）：
1. 现象：顺序训练任务 A 再到任务 B，共享参数被 B 改写 → A 的准确率断崖下跌；
2. 三种策略对比（核心产出：acc_A 三条曲线）：
   - 基线(无保护)：只训 B → A 崩；
   - 经验回放(replay)：B 的 batch 混入 10-20% A 数据 → 最朴素有效的基线；
   - EWC：对"对 A 重要的参数"加二次惩罚 Σ F_i(θ_i-θ_A*)^2，F 用 A 上梯度平方(Fisher)近似；
3. 面试延伸：持续预训练/增量学习都要防灾难性遗忘；回放性价比最高。
运行：python d10_catastrophic_forgetting.py [--steps-b 260]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


class MLP(nn.Module):
    def __init__(self, din=8, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def gen_data(rng, w, n=2000, seed=0):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, 8)).astype("float32")
    y = ((X @ w + r.normal(0, 0.3, n)) > 0).astype("int64")
    return torch.from_numpy(X), torch.from_numpy(y)


def train_steps(model, X, y, steps, opt, bs=32, Xa=None, ya=None, mix=0.0,
                fisher=None, thetaA=None, ewc_lambda=0.0):
    """在任务B上训 steps；Xa/ya 供回放；fisher/thetaA 供 EWC。返回平均 loss。"""
    n = len(X)
    loss_sum = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        xb, yb = X[idx], y[idx]
        if mix > 0 and Xa is not None:
            ia = torch.randint(0, len(Xa), (int(bs * mix),))
            xb = torch.cat([xb, Xa[ia]]); yb = torch.cat([yb, ya[ia]])
        opt.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        if ewc_lambda > 0 and fisher is not None:
            pen = 0.0
            for p, f, pt in zip(model.parameters(), fisher, thetaA):
                pen = pen + (f * (p - pt) ** 2).sum()
            loss = loss + ewc_lambda * pen
        loss.backward(); opt.step()
        loss_sum += float(loss.item())
    return loss_sum / steps


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(-1)
    model.train()
    return (pred == y).float().mean().item()


def compute_fisher(model, X, y):
    """EWC Fisher：对 A 数据算 logp(y) 梯度平方的期望（diag）。"""
    model.eval()
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    n = len(X)
    for _ in range(120):
        idx = torch.randint(0, n, (32,))
        out = model(X[idx])
        lp = F.log_softmax(out, -1)[torch.arange(len(idx)), y[idx]].sum()
        grads = torch.autograd.grad(lp, model.parameters(), retain_graph=False)
        for g, f in zip(grads, fisher):
            if g is not None:
                f += g.detach() ** 2
    model.train()
    return [f / 120 for f in fisher]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=400)
    ap.add_argument("--steps-b", type=int, default=260)
    ap.add_argument("--ewc-lambda", type=float, default=3.0)
    ap.add_argument("--replay-mix", type=float, default=0.35)
    args = ap.parse_args()

    rng = np.random.default_rng(3)
    # wB 由 wA 旋转约 60°，让两任务共享特征空间 → 复用参数 → 顺序学习会互相改写
    rw = np.random.default_rng(0)
    wA = rw.standard_normal(8); wA /= np.linalg.norm(wA)
    v = rw.standard_normal(8); v -= v @ wA * wA; v /= np.linalg.norm(v)
    wB = np.cos(1.05) * wA + np.sin(1.05) * v
    Xa, ya = gen_data(rng, wA, seed=1)
    Xb, yb = gen_data(rng, wB, seed=2)

    def fresh():
        torch.manual_seed(0)
        return MLP()

    # ---- 先在任务 A 上训好 ----
    m_base = fresh()
    train_steps(m_base, Xa, ya, args.steps_a, torch.optim.Adam(m_base.parameters(), lr=3e-3))
    accA0 = accuracy(m_base, Xa, ya)
    print(f"任务A训练后 accA={accA0:.3f}；开始顺序学任务B并持续观测 accA。")

    def run_with(protect):
        m = fresh()
        m.load_state_dict(m_base.state_dict())
        fisher = theta = None
        if protect == "ewc":
            fisher = compute_fisher(m, Xa, ya)
            theta = [p.detach().clone() for p in m.parameters()]
        accA, accB = [], []
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        Xam = Xa[:800]; yam = ya[:800]
        # 分 10 段记录
        for seg in range(10):
            if protect == "replay":
                train_steps(m, Xb, yb, args.steps_b // 10, opt, Xa=Xam, ya=yam,
                            mix=args.replay_mix)
            elif protect == "ewc":
                train_steps(m, Xb, yb, args.steps_b // 10, opt,
                            fisher=fisher, thetaA=theta, ewc_lambda=args.ewc_lambda)
            else:
                train_steps(m, Xb, yb, args.steps_b // 10, opt)
            accA.append(accuracy(m, Xa, ya))
            accB.append(accuracy(m, Xb, yb))
        return accA, accB

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    labels = {"none": "baseline (no protect)", "replay": "replay 20% A", "ewc": "EWC"}
    summary = {}
    for tag in ["none", "replay", "ewc"]:
        accA, accB = run_with(tag)
        summary[tag] = (accA[-1], accB[-1])
        xs = np.arange(1, 11)
        ax.plot(xs, accA, "o-", label=f"accA {labels[tag]}")
        print(f"[{tag:<6}] 任务B训练中 accA: 起点={accA0:.3f} -> 终点={accA[-1]:.3f}"
              f"（掉了 {(accA0-accA[-1])*100:.1f}pp）；最终 accB={accB[-1]:.3f}")
    ax.axhline(accA0, ls="--", color="gray", alpha=.7)
    ax.text(1, accA0, f" accA after task A = {accA0:.2f}", color="gray")
    ax.set_xlabel("progress on task B (segments)"); ax.set_ylabel("task A accuracy")
    ax.set_title("catastrophic forgetting & mitigation")
    ax.legend(); ax.grid(alpha=.3)
    viz.save(fig, "d10_forgetting_curves.png")
    print("\n结论：顺序训练必遗忘；回放与 EWC 都能显著缓解，回放最朴素有效。")


if __name__ == "__main__":
    main()
