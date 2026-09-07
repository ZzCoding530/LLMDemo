# -*- coding: utf-8 -*-
"""考点16·17 GRPO（组内相对优势砍掉 Value Model）+ RLVR(可验证奖励)
类型：演示实现（torch tiny GPT + 规则 verifier，CPU 可跑；7B 真实版在 gpu/g02）

讲解思路（备课用）：
1. PPO 需要 value model 估 baseline → 又一个大模型、又一份显存；
   GRPO 砍掉它：对同一个问题采样 G 个回答，组内归一化 advantage_i=(r_i-mean)/std；
   "组内谁比谁好"就是 baseline——打印一组手算数值让同学看到；
2. RLVR：reward 不靠人类 RM，而是代码/规则 verifier（如答案精确匹配、判题器），
   本 demo verifier = 字符串判 a+b 结果 → 训练目标是可验证的数学能力；
3. 流程：SFT-warm 让模型会一点 → GRPO 若干轮（每轮每组 G 个采样）→ pass@1 曲线上升；
4. loss = -mean(adv_i·logπ_i) + β·KL(π‖π_ref)。
运行：python d14_grpo_rlvr.py [--warm 200] [--rounds 25] [--g 8]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz

# ---- 本地字符词表：数字与 + = #（# 表示回答结束） ----
CH = "0123456789+=#"
CID = {c: i for i, c in enumerate(CH)}


def enc(s):
    return [CID[c] for c in s]


def dec(ids):
    return "".join(CH[i] for i in ids)


def make_problems(n, seed, hi=9):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = int(rng.integers(0, hi + 1)); b = int(rng.integers(0, hi + 1))
        out.append((a, b, f"{a}+{b}="))
    return out


class TinyMathGPT(nn.Module):
    """极小的因果 GPT（数字语料专用）。"""
    def __init__(self, vocab=len(CH), d=48, block=24, n_layer=1, n_head=2):
        super().__init__()
        self.vocab, self.d, self.block = vocab, d, block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, block, d))
        nn.init.normal_(self.pos, 0, 0.02)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln2 = nn.LayerNorm(d)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.causal_mask = None

    def forward(self, idx, full=False):
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T]
        a = self.ln1(x)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        a, _ = self.attn(a, a, a, attn_mask=mask)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        logits = self.head(self.lnf(x))
        if full:
            return logits
        return logits

    def logits(self, ids):
        ids = ids[-self.block:]
        return self.forward(ids.unsqueeze(0), full=True)[0]


def generate_answer(model, prompt_str, max_new=4, greedy=False, seed=0):
    """从 'a+b=' 生成最多 4 个字符（数字+#），遇 # 停。返回 ids、结束原因。"""
    ids = torch.tensor([enc(prompt_str)])
    rng = torch.Generator().manual_seed(seed)
    for _ in range(max_new):
        logits = model.logits(ids[0])[-1]                 # (V,)
        if greedy:
            nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
        else:
            p = F.softmax(logits / 0.9, -1)
            nxt = torch.multinomial(p, 1, generator=rng).unsqueeze(0)
        ids = torch.cat([ids, nxt], 1)
        if dec(ids[0].tolist())[-1] == "#":
            break
    return ids[0]


def parse_answer(ids, plen):
    s = dec(ids[plen:].tolist()).split("#")[0].strip("=")
    return s


def verifier(a, b, s):
    """RLVR：可编程的细粒度打分——完全正确=1；前缀逐位给 0.5 比例分。
    演示要点：verifier 是代码规则，不需人类标注，且可以比 0/1 更"有梯度"。"""
    exp = str(a + b)
    if s == exp:
        return 1.0
    # 前缀匹配奖励：前 k 位正确给 k/len·0.5，帮助稀疏奖励下学习
    k = 0
    for i in range(min(len(s), len(exp))):
        if s[i] == exp[i]:
            k += 1
        else:
            break
    return 0.5 * k / len(exp) if k > 0 else 0.0


def answer_mean_logp(model, ids, plen):
    """回答段(从 plen 起)每 token 平均 logπ（带梯度）。"""
    logits = model.logits(ids)                            # (L,V)
    # logits[t] 预测 ids[t+1]；回答字符 ids[P:] 由位置 P-1..L-2 预测
    P = plen
    lp = F.log_softmax(logits[P - 1:-1], -1)
    tg = ids[P:]
    return lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1).mean()


def sft_warm(model, problems, steps=200, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        a, b, q = problems[np.random.randint(len(problems))]
        ans = str(a + b) + "#"
        seq = enc(q) + enc(ans)
        ids = torch.tensor(seq)
        logits = model.logits(ids)                         # (L,V)
        P = len(q)
        loss = F.cross_entropy(logits[P - 1:-1], ids[P:])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def pass_at_1(model, probs, seed=0):
    ok = 0
    for a, b, q in probs:
        ids = generate_answer(model, q, greedy=True)
        s = parse_answer(ids, len(q))
        ok += verifier(a, b, s)
    return ok / len(probs)


def grpo_round(model, ref, problems, g=8, beta=0.01, lr=3e-4, seed=0):
    """一轮 GRPO：对 batch 个问题各采样 g 个 → 组内归一化 advantage → PG + KL。"""
    B = min(6, len(problems))
    batch = [problems[np.random.randint(len(problems))] for _ in range(B)]
    # 采样
    rollouts = {}                      # (a,b,q) -> list of (ids, r)
    for (a, b, q) in batch:
        rollouts.setdefault((a, b, q), [])
        for k in range(g):
            ids = generate_answer(model, q, greedy=False, seed=seed * 100 + k)
            s = parse_answer(ids, len(q))
            rollouts[(a, b, q)].append((ids, verifier(a, b, s)))

    # 打印第一组的 advantage 手算（教学演示）
    first = list(rollouts)[0]
    rws = torch.tensor([r for _, r in rollouts[first]])
    advs = (rws - rws.mean()) / (rws.std() + 1e-6)
    print(f"  [demo] q='{first[2]}' rewards={rws.tolist()} -> advantages="
          f"{[round(x, 2) for x in advs.tolist()]}  (组内归一化, 无 value model)")

    # loss = -mean(adv·logπ) + β·KL(π‖π_ref)，advantage 在组内按序对应
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()
    total = torch.tensor(0.0)
    n = 0
    for (a, b, q), lst in rollouts.items():
        rw = torch.tensor([r for _, r in lst], dtype=torch.float)
        adv = (rw - rw.mean()) / (rw.std() + 1e-6)
        for i, (ids, r) in enumerate(lst):
            plen = len(q)
            lp = answer_mean_logp(model, ids, plen)
            if ref is not None:
                with torch.no_grad():
                    rl = answer_mean_logp(ref, ids, plen)
                total = total - adv[i] * lp + beta * (lp - rl)
            else:
                total = total - adv[i] * lp
            n += 1
    loss = total / n
    loss.backward()
    opt.step()
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--g", type=int, default=8)
    args = ap.parse_args()

    probs = make_problems(160, seed=1)
    eval_probs = make_problems(60, seed=2)
    torch.manual_seed(0)
    model = TinyMathGPT()
    ref = TinyMathGPT(); ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    print(f"[0] 随机初始化 pass@1={pass_at_1(model, eval_probs):.2f}")
    model = sft_warm(model, probs, steps=args.warm)
    start_p1 = pass_at_1(model, eval_probs)
    print(f"[SFT-warm] pass@1={start_p1:.2f}")

    curve = [(0, start_p1)]
    for rnd in range(args.rounds):
        loss = grpo_round(model, ref, probs, g=args.g, seed=rnd)
        if rnd % 5 == 4 or rnd == args.rounds - 1:
            p1 = pass_at_1(model, eval_probs)
            curve.append((rnd + 1, p1))
            print(f"[GRPO round {rnd+1}/{args.rounds}] loss={loss:.3f} pass@1={p1:.2f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4))
    xs = [c[0] for c in curve]
    ax.plot(xs, [c[1] for c in curve], "o-")
    ax.axhline(start_p1, ls="--", color="gray")
    ax.set_xlabel("GRPO round"); ax.set_ylabel("pass@1 on held-out (verifiable reward)")
    ax.set_title("RLVR: policy improves via rule verifier only")
    ax.grid(alpha=.3)
    viz.save(fig, "d14_grpo_rlvr.png")
    print("\n结论：GRPO 用组内相对优势替代 value model（省一份大模型显存），"
          "配合规则 verifier 实现 RLVR。7B 真实 RLVR 脚本见 gpu/g02_grpo_7b_rlvr.py。")


if __name__ == "__main__":
    main()
