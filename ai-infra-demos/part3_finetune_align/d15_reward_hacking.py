# -*- coding: utf-8 -*-
"""考点19 对齐税 / Reward Hacking：可被钻空子的奖励 → reward 涨、真实指标先涨后跌
类型：演示 + 对照（torch 分类策略 toy，CPU 秒级；对应表格 Demo19）

讲解思路（备课用）：
1. RL 只优化"你给的 reward 函数"；如果 reward 与真实意图有缝，agent 会 exploit 缝
   （钻空子/奖励黑客）：训练曲线 reward 一路上涨，但真实指标先涨后跌；
2. 本 toy：动作=(是否含关键词 kw, 长度 len)。真实质量 = 关键词的 0.5 分 - 过长惩罚；
   但给模型的"可 hack 奖励"= 2×kw + 0.2×len（长度无限加分）→ 模型学会刷长度；
3. 修复：reward shaping（把长度加分限制在合理区间 + 对齐真实质量的权重），
   重训后真实指标回升 → 引出一课：reward 设计/防 hacking 是 RLHF 工程质量核心，
   对齐税 = 对齐过程牺牲通用能力，假对齐 = 表面满足 reward 实则漏洞。
运行：python d15_reward_hacking.py [--iters 500]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def build_actions(K=240):
    """每个动作 i 的属性：是否含关键词 kw，长度 len（覆盖各组合）。"""
    kw = np.zeros(K, dtype=np.float32)
    ln = np.zeros(K, dtype=np.float32)
    for i in range(K):
        kw[i] = float(i % 2)
        ln[i] = float((i // 2) % 20 + 1)
    return torch.from_numpy(kw), torch.from_numpy(ln)


def true_metric(kw, ln):
    """真实质量：关键词+0.5，但长度超过 6 开始罚（agent 看不到）。"""
    return 0.5 * kw - 0.05 * torch.clamp(ln - 6, min=0)


def reward_hackable(kw, ln):
    """漏洞奖励：长度给 0.2/单位无限加分 → 刷长度是最优 hack。"""
    return 2.0 * kw + 0.2 * ln


def reward_shaped(kw, ln):
    """修复后的奖励：长度加分封顶在 6（与真实质量的合理长度一致）。"""
    return 1.5 * kw + 0.35 * torch.clamp(ln, max=6.0)


def run(reward_fn, iters, seed=0):
    torch.manual_seed(seed)
    K = 240
    kw, ln = build_actions(K)
    q = true_metric(kw, ln)
    theta = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=3e-2)
    hist_q, hist_r = [], []
    for _ in range(iters):
        p = F.softmax(theta, 0)
        a = torch.multinomial(p, 128, replacement=True)
        r = reward_fn(kw[a], ln[a])
        adv = r - r.mean()
        lp = F.log_softmax(theta, 0)[a]
        loss = -(lp * adv).mean() + 0.02 * (p * p.log()).sum()   # 小熵正则防塌
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p2 = F.softmax(theta, 0)
            hist_q.append(float((p2 * q).sum()))
            hist_r.append(float((p2 * reward_fn(kw, ln)).sum()))
    return hist_q, hist_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()

    q_hack, r_hack = run(reward_hackable, args.iters, seed=0)
    q_fix, r_fix = run(reward_shaped, args.iters, seed=1)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(args.iters)
    axes[0].plot(xs, r_hack, label="hackable reward (2*kw + 0.2*len)")
    axes[0].plot(xs, r_fix, label="shaped reward (len bonus capped)")
    axes[0].set_title("reward curve keeps rising")
    axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(xs, q_hack, label="true quality (hackable reward)")
    axes[1].plot(xs, q_fix, label="true quality (shaped reward)")
    axes[1].set_title("true quality: up then down when gamed")
    axes[1].legend(); axes[1].grid(alpha=.3)
    viz.save(fig, "d15_reward_hacking.png")
    print(f"真实质量区间(首段均值 → 最终)：hackable reward → {q_hack[len(q_hack)//3]:.3f} → {q_hack[-1]:.3f}"
          f"（reward 涨、质量跌=被打洞）")
    print(f"真实质量区间(首段均值 → 最终)：shaped  reward → {q_fix[len(q_fix)//3]:.3f} → {q_fix[-1]:.3f}"
          f"（持续上升=修复生效）")
    print("\n结论：reward 与真实意图有缝就会被打洞；reward hacking 靠 KL 约束 + 更贴真的 reward/shaping 防。"
          "由此引出对齐税(对齐降通用能力)与假对齐(表面达标)概念。")


if __name__ == "__main__":
    main()
