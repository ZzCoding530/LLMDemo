# -*- coding: utf-8 -*-
"""考点22 投机解码(Speculative Decoding)：n-gram draft + 拒绝采样 + 接受率/加速比
类型：演示 + AB 对照（自训 tiny target + n-gram draft，CPU；四代演进在讲解稿）

讲解思路（备课用）：
1. 动机：decode 每步只出 1 token 且受显存带宽限制；用便宜的 draft 一次猜 K 个，
   target 一次前向并行验证 K+1 个位置 → 平均每步产出 >1 token；
2. 拒绝采样：第 j 个 draft token 以 min(1, p_target(a)/p_draft(a)) 接受；
   被拒则从 max(0, p_target-p_draft) 归一化分布重采样并停 → 输出分布与 target 单独解码一致；
3. 每步期望产出 = (1-α^{K+1})/(1-α)，α=接受率；
4. 工程演进：EAGLE(自回归 draft 用 target 特征)/MTP(DeepSeek 多 token 预测头)/DFlash/DSpark。
运行：python d18_speculative_decoding.py [--k 4] [--tokens 200]
"""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.corpus import encode, VOCAB_SIZE
from common.toylm import load_or_train


class NGramDraft:
    """n-gram 近似分布：给定前 n-1 个字符预测下一个。零训练。"""
    def __init__(self, text, n=3, vocab_size=VOCAB_SIZE):
        self.n, self.vocab = n, vocab_size
        self.counts = {}
        toks = encode(text)
        for i in range(len(toks) - n + 1):
            ctx = tuple(toks[i:i + n - 1])
            nxt = toks[i + n - 1]
            d = self.counts.setdefault(ctx, np.zeros(vocab_size))
            d[nxt] += 1

    def probs(self, ctx_tokens):
        ctx = tuple(ctx_tokens[-(self.n - 1):])
        d = self.counts.get(ctx)
        if d is None:
            return np.full(self.vocab, 1.0 / self.vocab)
        d = d + 0.1                          # 平滑，避免 0 概率导致拒绝采样除零
        return d / d.sum()


@torch.no_grad()
def next_dist_last(model, seq_ids):
    """seq_ids 截断到上下文后，取最后一个位置的 next-token 分布（自回归单步用）。"""
    inp = seq_ids[-model.cfg.block_size:]
    logits, _, _ = model(inp.unsqueeze(0))
    return F.softmax(logits[0, -1], -1).numpy()


@torch.no_grad()
def speculative_step_batched(model, draft, prefix, K, rng):
    """一步投机：draft 猜 K 个 → target 单次前向(整段)验证 K+1 位置。返回(新prefix, 接受数)。"""
    # 1) draft 猜 K 个
    draft_toks = []
    cur = prefix
    for _ in range(K):
        q = draft.probs(cur.tolist())
        a = int(rng.choice(VOCAB_SIZE, p=q))
        draft_toks.append(a)
        cur = torch.cat([cur, torch.tensor([a])])
    # 2) target 一次前向：得到每个位置的分布（含 bonus 位）
    L = len(prefix)
    seq = cur
    inp = seq[-model.cfg.block_size:]
    logits, _, _ = model(inp.unsqueeze(0))
    offset = len(seq) - inp.shape[0]          # seq 前被截掉的长度（对齐用）
    dists = []
    for j in range(K + 1):
        row = L + j - offset - 1               # 预测全局位置 L+j 的那一行
        dists.append(F.softmax(logits[0, row], -1).numpy())

    # 3) 逐位拒绝采样
    accepted = []
    cur2 = prefix
    for j in range(K):
        a = draft_toks[j]
        p_j = dists[j][a]
        q_j = draft.probs(cur2.tolist())[a]
        if rng.random() <= min(1.0, p_j / max(q_j, 1e-8)):
            accepted.append(a)
            cur2 = torch.cat([cur2, torch.tensor([a])])
        else:
            residual = np.maximum(dists[j] - draft.probs(cur2.tolist()), 0)
            residual /= residual.sum()
            bonus = int(rng.choice(VOCAB_SIZE, p=residual))
            cur2 = torch.cat([cur2, torch.tensor([bonus])])
            break
    else:
        bonus_p = dists[K]
        bonus = int(rng.choice(VOCAB_SIZE, p=bonus_p))
        cur2 = torch.cat([cur2, torch.tensor([bonus])])
    return cur2, len(accepted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()

    model, text = load_or_train()
    draft = NGramDraft(text, n=3)
    rng = np.random.default_rng(0)
    prefix = torch.tensor(encode(text[300:360]))

    # ---- baseline：自回归（每 token 一次前向）----
    t0 = time.perf_counter()
    cur = prefix
    for _ in range(args.tokens):
        p = next_dist_last(model, cur)
        a = int(rng.choice(VOCAB_SIZE, p=p))
        cur = torch.cat([cur, torch.tensor([a])])
    dt_base = time.perf_counter() - t0

    # ---- 投机解码（每步一次前向）----
    cur2 = prefix
    total_acc = 0
    steps = 0
    t0 = time.perf_counter()
    while len(cur2) - len(prefix) < args.tokens:
        cur2, n_acc = speculative_step_batched(model, draft, cur2, args.k, rng)
        total_acc += n_acc
        steps += 1
    dt_spec = time.perf_counter() - t0

    produced = len(cur2) - len(prefix)
    alpha = total_acc / (steps * args.k) if steps else 0
    theory = (1 - alpha ** (args.k + 1)) / (1 - alpha) if alpha < 1 else args.k + 1
    print(f"baseline : {produced} tokens / {steps if False else args.tokens} 次前向，"
          f"用时 {dt_base*1e3:.0f} ms")
    print(f"speculate: draft K={args.k}，target 前向 {steps} 次产出 {produced} tokens，"
          f"用时 {dt_spec*1e3:.0f} ms")
    print(f"接受率 α={alpha:.3f}，每步实际产出 {produced/steps:.2f}，"
          f"理论 (1-α^(K+1))/(1-α)={theory:.2f}；加速比（省前向次数）="
          f"{args.tokens/steps:.2f}×")
    print("注：toy CPU 上每前向代价极小，Python 循环开销会吃掉部分收益；真实 GPU 大模型"
          "decode 每前向代价高，省前向=省带宽才有显著加速。")

    # 分布一致性：两种方法"首个新 token"的直方图应一致（投机解码不掉点）
    hist_a = np.zeros(VOCAB_SIZE); hist_b = np.zeros(VOCAB_SIZE)
    for i in range(300):
        p = next_dist_last(model, prefix)
        hist_a[int(rng.choice(VOCAB_SIZE, p=p))] += 1
        cur3, _ = speculative_step_batched(model, draft, prefix, args.k,
                                           np.random.default_rng(5000 + i))
        first_new = int(cur3[len(prefix)])
        hist_b[first_new] += 1
    pa = hist_a / hist_a.sum(); pb = hist_b / hist_b.sum()
    m = 0.5 * (pa + pb)
    js = 0.5 * ((pa * np.log((pa + 1e-9) / (m + 1e-9))).sum()
                + (pb * np.log((pb + 1e-9) / (m + 1e-9))).sum())
    print(f"首 token 分布一致性 JS 距离={js:.4f}（≈0 → 投机解码与原分布一致，质量不掉）")


if __name__ == "__main__":
    main()
