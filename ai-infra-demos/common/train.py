# -*- coding: utf-8 -*-
"""字符级 tiny LM 的训练循环 + 数据集封装（预训练 / SFT-mask 两用）。

讲解思路：
- 预训练：把全文当 token 流，随机切窗口做 next-token 预测（自监督）；
- SFT：文本带 'q ... a ...' 结构，只让 'a ' 之后的 token 参与 loss（问答格式对齐）；
  面试常问"SFT 数据为什么要把 prompt 部分 mask 掉"，用本模块的 loss_mask 演示。
"""
import math
import numpy as np
import torch
from . import corpus


def _tok(s):
    return np.array(corpus.encode(s), dtype=np.int64)


class PretrainStream:
    """把整段纯文本编码为 token 流，每次采样随机连续窗口。"""
    def __init__(self, text, seed=0):
        self.rng = np.random.default_rng(seed)
        self.tokens = _tok(text)
        self.len = len(self.tokens)
        assert self.len > 0

    def batch(self, bs, block):
        n = self.len - block - 1
        starts = self.rng.integers(0, n, size=bs)
        x = np.stack([self.tokens[s:s + block] for s in starts])
        y = np.stack([self.tokens[s + 1:s + block + 1] for s in starts])
        return torch.from_numpy(x), torch.from_numpy(y)


class SFTDataset:
    """qa 文本（'q ... a ...' 每行一对）→ (prompt, answer) 对；
    batch 返回 x/y/mask，mask 只覆盖 answer 区间的 token。"""
    def __init__(self, qa_text, seed=0):
        self.rng = np.random.default_rng(seed)
        self.pairs = []
        for line in qa_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 行格式 'q ... a ...'：answer 从 ' a ' 之后开始
            idx = line.find(" a ")
            if idx < 0:
                continue
            prompt, ans = line[:idx], line[idx:]
            self.pairs.append((_tok(prompt), _tok(ans)))
        assert self.pairs, "SFTDataset: 没有解析到任何 'q ... a ...' 对"

    def __len__(self):
        return len(self.pairs)

    def batch(self, bs, block):
        B, T = bs, block
        x = np.zeros((B, T), dtype=np.int64)
        y = np.zeros((B, T), dtype=np.int64)
        m = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            p, a = self.pairs[int(self.rng.integers(0, len(self.pairs)))]
            seq = np.concatenate([p, a])
            # 随机偏移取窗口（短样本直接从头取，末尾补 0，由 mask 排除）
            if len(seq) > block + 1:
                offset = int(self.rng.integers(0, len(seq) - block))
            else:
                offset = 0
            win = seq[offset:offset + block + 1]
            x[b] = np.pad(win[:T], (0, max(0, T - len(win[:T]))))
            y[b] = np.pad(win[1:T + 1], (0, max(0, T - len(win[1:T + 1]))))
            # 只有 answer 区间的 token 参与 loss：y 第 j 列对应原位置 offset+j+1
            a0 = offset + len(p)                 # answer 起始（原 seq 下标）
            a1 = offset + len(p) + len(a)        # answer 结束
            lo = max(0, a0 - offset - 1)
            hi = min(T, a1 - offset - 1)
            if hi > lo:
                m[b, lo:hi] = 1.0
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m))


def train_lm(model, train_data, steps=300, lr=3e-3, bs=16, block=64,
             val_data=None, eval_every=60, device="cpu", seed=0,
             sft=False, log=True, wd=0.01):
    """通用 LM 训练；val_data 与 train_data 同类型；返回 loss 历史与 val 历史。"""
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0)
    hist, val_hist = [], []
    for step in range(steps):
        if sft:
            x, y, mask = train_data.batch(bs, block)
            _, loss, _ = model(x.to(device), targets=y.to(device), loss_mask=mask.to(device))
        else:
            x, y = train_data.batch(bs, block)
            _, loss, _ = model(x.to(device), targets=y.to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.item()))
        if eval_every and val_data is not None and (step % eval_every == 0 or step == steps - 1):
            v = eval_loss(model, val_data, sft=sft, device=device)
            val_hist.append((step, v))
            if log:
                print(f"step {step:4d} train_loss={loss.item():.4f} val_loss={v:.4f}")
        elif log and (step % max(1, steps // 8) == 0):
            print(f"step {step:4d} train_loss={loss.item():.4f}")
    return hist, val_hist


@torch.no_grad()
def eval_loss(model, data, sft=False, device="cpu", batches=10, bs=16, block=64):
    model.eval()
    tot, cnt = 0.0, 0
    for _ in range(batches):
        if sft:
            x, y, mask = data.batch(bs, block)
            _, loss, _ = model(x.to(device), targets=y.to(device), loss_mask=mask.to(device))
        else:
            x, y = data.batch(bs, block)
            _, loss, _ = model(x.to(device), targets=y.to(device))
        tot += float(loss.item()) * bs
        cnt += bs
    model.train()
    return tot / cnt
