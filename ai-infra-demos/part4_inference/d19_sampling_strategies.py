# -*- coding: utf-8 -*-
"""考点24 解码策略：greedy / temperature / top-k / top-p / beam + 为什么 LLM 不用 beam search
类型：演示 + 对照（numpy 手写采样器 + tiny LM，CPU）

讲解思路（备课用）：
1. greedy：每次取 argmax → 确定但易重复、单一；
2. temperature：p=softmax(logits/T)，T>1 变平(多样)、T<1 变尖(确定)；
3. top-k：只保留概率最大的 k 个再归一（缺点：k 固定，分布平坦时截断过多）；
4. top-p(nucleus)：按累计概率保留到 p 为止的最小集合（自适应截断）——目前主流；
5. beam search：保留 top-B 条完整路径 → 搜索"全局"高概率序列。
   但对开放式生成(LLM)不合适：高分路径趋同+越长的序列概率越小 → 生成重复/空洞；
   且代价高、无法采样多样化 → 所以 chat/续写用 sampling；
6. 实验：同一 tiny LM 同一 prompt，不同策略各生成 3-5 条，比 distinct-2/重复/平均 logP。
运行：python d19_sampling_strategies.py [--len 80] [--samples 4]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import encode, decode, VOCAB_SIZE, CHARS
from common.toylm import load_or_train


@torch.no_grad()
def step_dist(model, ids):
    inp = ids[-model.cfg.block_size:].unsqueeze(0)
    logits, _, _ = model(inp)
    return logits[0, -1]


def sample_from(logits, T=1.0, top_k=0, top_p=0.0, rng=None):
    rng = rng or np.random.default_rng()
    logits = logits.detach().float().numpy() / T
    if top_k and top_k > 0:
        th = np.sort(logits)[-top_k]
        logits[logits < th] = -np.inf
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    if top_p and 0 < top_p < 1:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        keep = np.where(cum - p[order] <= top_p)[0]
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = p * mask
        p = p / p.sum()
    return int(rng.choice(VOCAB_SIZE, p=p))


def generate(model, prompt, n, strategy, T=0.9, top_k=0, top_p=0.0, seed=0):
    ids = prompt.clone()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        logits = step_dist(model, ids)
        if strategy == "greedy":
            nxt = int(torch.argmax(logits))
        elif strategy == "beam":        # beam 单独处理（此处不会调用）
            nxt = int(torch.argmax(logits))
        else:
            nxt = sample_from(logits, T=T, top_k=top_k, top_p=top_p, rng=rng)
        ids = torch.cat([ids, torch.tensor([nxt])])
    return ids


def beam_search(model, prompt, width=3, n=80):
    beams = [(prompt.clone(), 0.0)]
    for _ in range(n):
        new = []
        for ids, score in beams:
            logits = step_dist(model, ids)
            lp = torch.log_softmax(logits, -1)
            topv, topi = torch.topk(lp, 2 * width)
            for v, i in zip(topv.tolist(), topi.tolist()):
                new.append((torch.cat([ids, torch.tensor([i])]), score + v))
        new.sort(key=lambda x: x[1], reverse=True)
        beams = new[:width]
    return beams[0][0]


def metrics(text_ids):
    ids = text_ids.tolist()
    bigrams = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    distinct = len(set(bigrams)) / max(len(bigrams), 1)
    # 重复率：最常出现的单 token 占比（粗略）
    from collections import Counter
    c = Counter(ids)
    rep = c.most_common(1)[0][1] / max(len(ids), 1)
    return distinct, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=80)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    model, text = load_or_train()
    prompt = torch.tensor(encode(text[2000:2040]))

    print("===== 各策略生成对比（不同 seed 重复 %d 次）=====" % args.samples)
    print(f"{'strategy':<12} {'distinct-2':>10} {'top-token重复率':>12} 样例")
    table = {}

    def run(strategy, T=1.0, k=0, p=0.0):
        ds, rps = [], []
        ex = None
        base_seed = sum(map(ord, strategy)) % 1000
        for s in range(args.samples):
            ids = generate(model, prompt, args.len, strategy, T=T, top_k=k, top_p=p,
                           seed=base_seed + s * 31)
            d, rp = metrics(ids)
            ds.append(d); rps.append(rp)
            if ex is None:
                ex = decode(ids[len(prompt):].tolist())
        print(f"{strategy:<12} {np.mean(ds):>10.3f} {np.mean(rps):>12.3f}  {ex[:46]!r}")
        table[strategy] = (np.mean(ds), np.mean(rps))

    run("greedy")
    run("temperature0.8")
    run("topk10")
    run("topp0.9")
    beam_ids = beam_search(model, prompt, width=3, n=args.len)
    bd, br = metrics(beam_ids)
    print(f"{'beam(w=3)':<12} {bd:>10.3f} {br:>12.3f}  {decode(beam_ids[len(prompt):].tolist())[:46]!r}")
    table["beam"] = (bd, br)

    # 分布截断可视化
    import matplotlib.pyplot as plt
    logits = step_dist(model, prompt)
    p = F.softmax(logits, -1).detach().numpy()
    order = np.argsort(p)[::-1]
    po = p[order]
    # top-p=0.9 需保留前几个
    cum = np.cumsum(po)
    n_p = int(np.argmax(cum >= 0.9)) + 1
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = np.arange(len(po))
    ax.bar(xs, po, color=["#4C72B0"] * n_p + ["#C0C0C0"] * (len(po) - n_p))
    ax.axvline(9.5, ls="--", color="gray")
    ax.text(12, max(po) * 0.9, "top-k=10 cutoff", color="gray")
    ax.set_xlabel("token rank (desc prob)"); ax.set_ylabel("prob")
    ax.set_title(f"real next-token distribution: top-p(0.9) keeps {n_p} tokens")
    viz.save(fig, "d19_sampling_truncation.png")

    names = list(table)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(names, [table[n][0] for n in names], color="#55A868")
    ax.set_ylabel("distinct bigram ratio"); ax.set_title("diversity by strategy")
    ax.grid(alpha=.3, axis="y")
    viz.save(fig, "d19_diversity.png")
    print("\n结论：sampling(top-p) 多样且流畅；beam 在开放式生成长序列易重复/空洞，"
          "加上计算代价高——这是 LLM 解码不用 beam 的原因。")


if __name__ == "__main__":
    main()
