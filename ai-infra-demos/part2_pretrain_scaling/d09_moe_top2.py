# -*- coding: utf-8 -*-
"""考点10 MoE 稀疏模型：top-2 路由 + 负载均衡 aux loss + 路由可视化
类型：AB 对比实验（Dense vs MoE；aux loss on/off；torch tiny，CPU 可跑）

讲解思路（备课用）：
1. MoE = 把 FFN 复制成 E 个 Expert，router 每 token 选 top-k(k=2) 加权组合；
   总参数多（几百 B），每 token 只激活 k 个专家（激活参数少）→ 训练/推理更省算力；
2. 风险：路由"羊群效应"——少数字典被疯狂选中、多数专家饿死 → 负载不均、浪费参数；
   解法：负载均衡 aux loss（如 Switch：E·Σ_c f_c·P_c），鼓励各专家被选概率均匀；
3. 本 demo：同一个小型 LM 任务，比较 ①Dense(激活参数量相当) ②MoE 不加aux ③MoE 加aux：
   看 val loss 与 训练后 expert 使用分布（柱状图：失衡 vs 均匀）；
4. 微调挑战（面试常问）：稀疏激活下全参微调只动被选专家 → 需 LoRA/只训 router 等策略。
运行：python d09_moe_top2.py [--steps 220]
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, VOCAB_SIZE
from common.tinygpt import GPTConfig, CausalSelfAttention
from common.train import PretrainStream, eval_loss


class MoEFFN(nn.Module):
    def __init__(self, d, n_experts=8, hidden=128, top_k=2):
        super().__init__()
        self.E = n_experts
        self.top_k = top_k
        self.hidden = hidden
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
            for _ in range(n_experts)])

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        N = flat.shape[0]
        g = F.softmax(self.gate(flat), dim=-1)          # (N,E) 全部分布
        topv, topi = torch.topk(g, self.top_k, dim=-1)  # (N,k)
        w = topv / (topv.sum(-1, keepdim=True) + 1e-8)  # top-k 归一化权重
        out = torch.zeros_like(flat)
        counts = torch.zeros(self.E, device=flat.device)
        for e in range(self.E):
            sel = (topi == e).any(-1)                   # (N,) 该专家是否被选中
            if sel.any():
                w_e = torch.where(topi == e, w, torch.zeros_like(w)).sum(-1)  # (N,)
                out[sel] += w_e[sel].unsqueeze(-1) * self.experts[e](flat[sel])
                counts[e] = sel.sum().item()
        # 负载均衡 aux loss（Switch 形式）：E·Σ_c f_c·P_c，均匀时最小
        frac = counts / counts.sum().clamp(min=1)
        aux = self.E * (frac * g.mean(0)).sum()
        return out.reshape(B, T, d), aux

    def activation_ratio(self):
        return self.top_k / self.E


class MoEGPT(nn.Module):
    """1 层 MoE transformer：emb + causal attn + MoE-FFN。"""
    def __init__(self, vocab, d=64, block=64, heads=4, n_experts=8, hidden=128, top_k=2):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, block, d))
        nn.init.normal_(self.pos, 0, 0.02)
        cfg = GPTConfig(vocab_size=vocab, block_size=block, n_layer=1,
                        n_head=heads, n_embd=d, pos_kind="learned")
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.moe = MoEFFN(d, n_experts, hidden, top_k)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, targets=None, aux_weight=0.0, loss_mask=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T]
        a, _ = self.attn(self.ln1(x))
        x = x + a
        m, aux = self.moe(self.ln2(x))
        x = x + m
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
            if aux_weight > 0:
                loss = loss + aux_weight * aux
        return logits, loss, aux


class DenseGPT(nn.Module):
    """激活参数与 MoE(top2,hidden=128) 相近的 Dense 对照：FFN 宽 2*hidden。"""
    def __init__(self, vocab, d=64, block=64, heads=4, ffn=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, block, d))
        nn.init.normal_(self.pos, 0, 0.02)
        cfg = GPTConfig(vocab_size=vocab, block_size=block, n_layer=1,
                        n_head=heads, n_embd=d, pos_kind="learned")
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, targets=None, aux_weight=0.0):
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T]
        a, _ = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1)) if targets is not None else None
        return logits, loss, None


def train_quick(model, tr, steps, aux_weight=0.0, bs=16, block=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last = None
    for _ in range(steps):
        x, y = tr.batch(bs, block)
        _, loss, _ = model(x, targets=y, aux_weight=aux_weight)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return last


def expert_usage(model, tr, n=400, block=64):
    """统计 router 把 token 派到各 expert 的次数（训练完成后）。"""
    model.eval()
    cnt = torch.zeros(model.moe.E)
    with torch.no_grad():
        for _ in range(n):
            x, _ = tr.batch(4, block)
            xb = model.tok(x) + model.pos[:, :block]
            a, _ = model.attn(model.ln1(xb))
            gate_in = model.ln2(xb + a)                  # 与 forward 中 moe 的输入一致
            g = F.softmax(model.moe.gate(gate_in), -1)
            topi = torch.topk(g, model.moe.top_k, -1).indices
            for e in range(model.moe.E):
                cnt[e] += (topi == e).any(-1).sum().item()
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=220)
    args = ap.parse_args()

    text = make_plain(1200, seed=41)
    tr = PretrainStream(text)
    va = PretrainStream(make_plain(150, seed=42), seed=99)
    d, E, hidden, topk = 64, 8, 128, 2

    print("===== Dense vs MoE (同一 token 预算) =====")
    dense = DenseGPT(VOCAB_SIZE, d=d, ffn=2 * hidden)
    dn = sum(p.numel() for p in dense.parameters())
    tr_loss = train_quick(dense, tr, args.steps)
    dv = eval_loss(dense, va)
    print(f"Dense(ffn={2*hidden})      params={dn:>9,}  train≈{tr_loss:.3f} val={dv:.4f}")

    moe_off = MoEGPT(VOCAB_SIZE, d=d, n_experts=E, hidden=hidden, top_k=topk)
    mn = sum(p.numel() for p in moe_off.parameters())
    tr_loss = train_quick(moe_off, tr, args.steps, aux_weight=0.0)
    mv_off = eval_loss(moe_off, va)
    print(f"MoE(top-{topk}/{E}, 不加aux) params={mn:>9,}  train≈{tr_loss:.3f} val={mv_off:.4f}")

    moe_on = MoEGPT(VOCAB_SIZE, d=d, n_experts=E, hidden=hidden, top_k=topk)
    tr_loss = train_quick(moe_on, tr, args.steps, aux_weight=0.01)
    mv_on = eval_loss(moe_on, va)
    print(f"MoE(top-{topk}/{E}, 加aux)   params={mn:>9,}  train≈{tr_loss:.3f} val={mv_on:.4f}")

    # 激活/总参数 & 使用分布
    active_mlp = topk * (2 * d * hidden)              # 2* 因为两个 Linear
    total_mlp = E * (2 * d * hidden)
    print(f"\nMoE MLP 部分：总参数≈{total_mlp/1e3:.0f}K，每 token 激活≈{active_mlp/1e3:.0f}K"
          f"（激活占比 {moe_on.moe.activation_ratio()*100:.0f}%）")
    cu, co = expert_usage(moe_off, tr), expert_usage(moe_on, tr)
    print("expert 使用次数  no-aux:", " ".join(f"{int(c):>5d}" for c in cu))
    print("expert 使用次数  aux:   ", " ".join(f"{int(c):>5d}" for c in co))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(E)
    axes[0].bar(x - .2, cu.numpy(), .4, label="no aux loss")
    axes[0].bar(x + .2, co.numpy(), .4, label="with aux loss")
    axes[0].set_title("expert usage after training")
    axes[0].set_xlabel("expert"); axes[0].legend(); axes[0].grid(alpha=.3, axis="y")
    axes[1].bar(["Dense", "MoE(no aux)", "MoE(aux)"],
                [dv, mv_off, mv_on], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("val loss comparison (same token budget)")
    axes[1].grid(alpha=.3, axis="y")
    viz.save(fig, "d09_moe.png")
    print("\n结论：aux loss 让专家使用从失衡变均匀；MoE 用更少激活参数达到可比 loss。"
          "面试延伸：负载不均是 MoE 训练/微调核心坑之一。")


if __name__ == "__main__":
    main()
