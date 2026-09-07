# -*- coding: utf-8 -*-
"""考点15 对齐谱系：DPO / ORPO / SimPO 从零实现 + synthetic 偏好对训练
类型：AB 对比实验（torch 手写三种 loss，tiny GPT，CPU 可跑；表格 Demo15）

讲解思路（备课用）：
1. DPO 关键推导：RLHF 的奖励 r(x,y)=β·log(π/π_ref)+β·logZ(x) 可解析消掉；
   于是无需 reward model 和 RL，直接最大化解：
   loss = -log σ( β( logπ(yw|x)-logπ_ref(yw|x)-logπ(yl|x)+logπ_ref(yl|x) ) )；
   隐式奖励 = β·log(π/π_ref)（面试常考）；
2. ORPO = chosen 的 NLL + λ·[-log σ(β·(logit(w)-logit(l)))]（去参考模型，一次训练同时
   "学会生成想要的、压制不想要的"）；
3. SimPO = 无 ref + 长度归一 logπ + margin γ：loss=-log σ(β·(meanlogπ(yw)-meanlogπ(yl)-γ))；
4. 实验：同一 SFT base，分别用三种 loss 训同一批 synthetic 偏好对 → 对比留出集上
   偏好差距 Δ=logπ(yw)-logπ(yl) 与 win-rate。
运行：python d13_dpo_family.py [--warm-steps 150] [--train-steps 220]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_qa, make_plain, encode, decode, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream

N_NAMES = ["amy", "ben", "cara", "dan", "eva", "finn", "gwen", "hugo"]
N_PLACES = ["hill", "river", "forest", "cave", "meadow", "pond", "ridge", "valley"]


def build_pref_pairs(n=120, seed=0):
    """(x, yw, yl)：问题 x，chosen=正确回答，rejected=名字/地点张冠李戴。"""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        who = rng.choice(N_NAMES); where = rng.choice(N_PLACES)
        wrong_who = rng.choice([w for w in N_NAMES if w != who])
        x = f"q {who} is where ? "
        yw = f"a {who} is near the {where} ."
        if rng.random() < 0.5:
            yl = f"a {wrong_who} is near the {where} ."
        else:
            yl = f"a {who} is near the {rng.choice([p for p in N_PLACES if p != where])} ."
        pairs.append((encode(x), encode(yw), encode(yl)))
    return pairs


def answer_mean_logp(model, p, a):
    """answer 每 token 平均 logπ（带梯度）。ids=[p,a]；只对 a 段的预测取 logp。"""
    ids = np.concatenate([p, a])
    x = torch.from_numpy(ids[:-1]).unsqueeze(0)
    tgt = torch.from_numpy(ids[1:]).unsqueeze(0)
    logits, _, _ = model(x)
    lp = F.log_softmax(logits, -1)                    # (1,T,V)
    P = len(p)
    sel = lp[0, P - 1:]                                # 预测下一 token 的位置从 P-1 起
    tg = tgt[0, P - 1:]
    toks = sel.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
    return toks.mean()


@torch.no_grad()
def ref_mean_logp(ref, p, a):
    return answer_mean_logp(ref, p, a).item()


def dpo_loss(model, ref, batch, beta=0.1):
    """loss = -log σ(β·(π_w-π_ref_w - π_l+π_ref_l))。ref 全程冻结。"""
    losses = []
    for p, yw, yl in batch:
        pw, pl = answer_mean_logp(model, p, yw), answer_mean_logp(model, p, yl)
        rw = torch.tensor(ref_mean_logp(ref, p, yw))
        rl = torch.tensor(ref_mean_logp(ref, p, yl))
        ratio = beta * ((pw - rw) - (pl - rl))
        losses.append(-F.logsigmoid(ratio))
    return torch.stack(losses).mean()


def orpo_loss(model, batch, beta=0.1, lam=0.1):
    """ORPO ≈ -mean(logπ(yw)) + λ·[-log σ(β·(odds_w - odds_l))]"""
    ce, odds = [], []
    for p, yw, yl in batch:
        pw, pl = answer_mean_logp(model, p, yw), answer_mean_logp(model, p, yl)
        ce.append(-pw)
        p_w = torch.exp(pw.clamp(max=0)).clamp(min=1e-6, max=1 - 1e-6)
        p_l = torch.exp(pl.clamp(max=0)).clamp(min=1e-6, max=1 - 1e-6)
        odds.append(-F.logsigmoid(beta * (torch.log(p_w / (1 - p_w)) - torch.log(p_l / (1 - p_l)))))
    return torch.stack(ce).mean() + lam * torch.stack(odds).mean()


def simpo_loss(model, batch, beta=0.8, gamma=0.5):
    """SimPO：无 ref，奖励=长度归一 logπ，margin γ。"""
    losses = []
    for p, yw, yl in batch:
        pw, pl = answer_mean_logp(model, p, yw), answer_mean_logp(model, p, yl)
        losses.append(-F.logsigmoid(beta * (pw - pl - gamma)))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, held):
    d = []
    for p, yw, yl in held:
        w = answer_mean_logp(model, p, yw).item()
        l = answer_mean_logp(model, p, yl).item()
        d.append(w - l)
    d = np.array(d)
    return float(d.mean()), float((d > 0).mean())


def clone_with_trainable(model):
    m2 = TinyGPT(model.cfg)
    m2.load_state_dict(model.state_dict())
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-steps", type=int, default=150)
    ap.add_argument("--train-steps", type=int, default=220)
    args = ap.parse_args()

    pairs = build_pref_pairs(120, seed=0)
    held = build_pref_pairs(40, seed=9)
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=64)

    torch.manual_seed(0)
    base = TinyGPT(cfg)
    from common.train import train_lm
    train_lm(base, PretrainStream(make_plain(1200, seed=5)), steps=180, bs=16, log=False)

    print("SFT-warm on chosen answers ...")
    torch.manual_seed(0)
    sft_base = TinyGPT(cfg); sft_base.load_state_dict(base.state_dict())
    opt = torch.optim.AdamW(sft_base.parameters(), lr=5e-4)
    for _ in range(args.warm_steps):
        p, yw, _ = pairs[np.random.randint(len(pairs))]
        loss = -answer_mean_logp(sft_base, p, yw)
        opt.zero_grad(); loss.backward(); opt.step()

    ref = clone_with_trainable(sft_base)
    for p in ref.parameters():
        p.requires_grad_(False)

    base_delta, base_wr = evaluate(sft_base, held)
    print(f"SFT-warm 后(训练前) Δ={base_delta:+.3f}  win-rate={base_wr:.2f}")

    results = {}
    for name, loss_fn, needs_ref in [("DPO", dpo_loss, True),
                                     ("ORPO", orpo_loss, False),
                                     ("SimPO", simpo_loss, False)]:
        torch.manual_seed(0)
        m = clone_with_trainable(sft_base)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        for _ in range(args.train_steps):
            idx = np.random.choice(len(pairs), 6, replace=False)
            batch = [pairs[i] for i in idx]
            loss = loss_fn(m, ref, batch) if needs_ref else loss_fn(m, batch)
            opt.zero_grad(); loss.backward(); opt.step()
        delta, wr = evaluate(m, held)
        results[name] = (delta, wr)
        print(f"{name:<6} Δ={delta:+.3f}  win-rate={wr:.2f}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(results)
    axes[0].bar(names, [results[n][0] for n in names])
    axes[0].axhline(base_delta, ls="--", color="gray")
    axes[0].set_ylabel("mean logp(yw)-logp(yl) on held-out")
    axes[0].set_title("preference gap after alignment")
    axes[0].grid(alpha=.3, axis="y")
    axes[1].bar(names, [results[n][1] for n in names])
    axes[1].axhline(base_wr, ls="--", color="gray")
    axes[1].set_ylabel("win-rate (chosen over rejected)")
    axes[1].set_title("preference win-rate on held-out")
    axes[1].grid(alpha=.3, axis="y")
    viz.save(fig, "d13_dpo_family.png")
    print("\n结论：三种 loss 都能让模型偏向 chosen；DPO 需冻结 ref，ORPO/SimPO 去 ref。"
          "DPO 隐式奖励 = β·log(π/π_ref)。")


if __name__ == "__main__":
    main()
