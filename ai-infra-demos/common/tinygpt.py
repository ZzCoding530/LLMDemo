# -*- coding: utf-8 -*-
"""通用 tiny GPT（torch, CPU/GPU 均可）——所有"训 tiny 模型"类 demo 的底座。

覆盖的架构旋钮（供不同考点复用）：
- 位置编码 pos_kind: learned | sincos | rope | alibi
- 注意力 head 结构: attn_kind: mha | gqa(n_kv_heads) | mqa
- 因果滑动窗口 window>0

设计说明（讲解思路）：
- 刻意不依赖 transformers/nanoGPT，全部手写，便于面试手撕时讲解每一行；
- 默认参数极小（CPU 秒级-分钟级收敛）；真实跑演示可调大参数与步数。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50
    block_size: int = 64        # 训练上下文长度
    n_layer: int = 2
    n_head: int = 4
    n_kv_heads: int = None      # None -> MHA; 1 -> MQA; g -> GQA(g 组)
    n_embd: int = 64
    pos_kind: str = "learned"   # learned | sincos | rope | alibi
    attn_kind: str = "mha"
    window: int = 0             # 0=全因果; >0=滑动窗口注意力
    dropout: float = 0.0
    rope_base: float = 10000.0
    # SFT 用途：当给出 qa_mask 时只对指定位置算 loss（见 forward）
    _cfg_name: str = "gpt"


def precompute_rope_freqs(dim, max_pos, base):
    # 每两维一个频率 theta_i = base^{-2i/d}
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)  # (max_pos, dim/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin):
    """x: (..., T, n_head, dh); cos/sin: (T, dh/2)。按相邻两维为一组旋转。"""
    *prefix, T, H, dh = x.shape
    x = x.reshape(*prefix, T, H, dh // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]                 # 每组两维拆开
    c = cos[:T].reshape(1, T, 1, dh // 2)
    s = sin[:T].reshape(1, T, 1, dh // 2)
    rx0 = x0 * c - x1 * s
    rx1 = x0 * s + x1 * c
    out = torch.stack([rx0, rx1], dim=-1).reshape(*prefix, T, H, dh)
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        d = cfg.n_embd
        assert d % cfg.n_head == 0
        self.dh = d // cfg.n_head
        if cfg.attn_kind == "mha":
            self.n_kv_heads = cfg.n_head
        elif cfg.attn_kind == "gqa":
            self.n_kv_heads = cfg.n_kv_heads or max(1, cfg.n_head // 2)
        elif cfg.attn_kind == "mqa":
            self.n_kv_heads = 1
        else:
            raise ValueError(cfg.attn_kind)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, self.n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d, self.n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.pos_kind == "rope":
            cos, sin = precompute_rope_freqs(self.dh, cfg.block_size * 8, cfg.rope_base)
            self.register_buffer("_cos", cos)
            self.register_buffer("_sin", sin)

    def forward(self, x, use_cache=False, cache=None):
        """cache: dict {layer_name: (K,B,T,head,dh)}；返回 (out, cache)。"""
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.dh)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.dh)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.dh)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
            cache = {"k": k, "v": v}
        if self.cfg.pos_kind == "rope":
            q = apply_rope(q, self._cos, self._sin)
            k = apply_rope(k, self._cos, self._sin)
        # GQA: K/V head 数 -> n_head 数
        if self.n_kv_heads != self.n_head:
            rep = self.n_head // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)                      # (B,H,T,dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        Tq, Tk = q.size(2), k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # (B,H,Tq,Tk)
        # 因果 / 滑动窗口 mask
        bias = torch.full((Tk, Tq), float("-inf"), device=x.device)
        col = torch.arange(Tq).unsqueeze(1)
        row = torch.arange(Tk).unsqueeze(0)
        causal = row <= col                              # (Tk,Tq) 上三角为 inf
        if self.cfg.window and self.cfg.window > 0:
            causal = causal & (col - row < self.cfg.window)
        att = att.masked_fill(~causal.bool(), float("-inf"))
        # ALiBi: 对 logits 减去 head 相关的线性距离惩罚（无参数）
        if self.cfg.pos_kind == "alibi":
            head_id = torch.arange(self.n_head, device=x.device).view(-1, 1, 1)
            m = 2.0 ** (-8.0 * head_id / self.n_head)   # 经典 8 头斜率分布
            dist = (col - row).clamp(min=0).to(x.dtype)  # (Tk,Tq)
            att = att - m * dist.unsqueeze(0)
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v                                     # (B,H,Tq,dh)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.wo(y), cache


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, use_cache=False, cache=None):
        a, cache = self.attn(self.ln1(x), use_cache, cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, cache


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_kind in ("learned", "sincos"):
            if cfg.pos_kind == "learned":
                self.pos = nn.Parameter(torch.zeros(1, cfg.block_size * 2, cfg.n_embd))
                nn.init.normal_(self.pos, 0.0, 0.02)
            else:  # sin/cos 位置编码：固定矩阵（非学习）
                self.register_buffer("pos", self._sincos(cfg))
        else:
            self.register_buffer("pos", None)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._name = cfg._cfg_name

    @staticmethod
    def _sincos(cfg):
        pe = torch.zeros(1, cfg.block_size * 2, cfg.n_embd)
        pos = torch.arange(cfg.block_size * 2).unsqueeze(1).float()
        i = torch.arange(0, cfg.n_embd, 2).float()
        pe[0, :, 0::2] = torch.sin(pos / 10000 ** (i / cfg.n_embd))
        pe[0, :, 1::2] = torch.cos(pos / 10000 ** (i / cfg.n_embd))
        return pe

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _merge_pos(self, x, T):
        if self.pos is None:
            return x
        return x + self.pos[:, :T]

    def forward(self, idx, targets=None, loss_mask=None, cache=None):
        """idx:(B,T). targets 与 loss_mask 可选（SFT 用 mask 屏蔽 question 部分）。
        cache: list of per-layer dicts（长度=n_layer），用于 KV cache 推理。"""
        cfg = self.cfg
        B, T = idx.shape
        x = self.tok(idx)
        x = self._merge_pos(x, T)
        new_cache = []
        for i, blk in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, layer_cache = blk(x, use_cache=cache is not None, cache=layer_cache)
            new_cache.append(layer_cache)
        x = self.ln_f(x)
        logits = self.head(x)                            # (B,T,V)
        loss = None
        if targets is not None:
            logits = logits.view(B * T, -1)
            tgt = targets.view(-1)
            losses = F.cross_entropy(logits, tgt, reduction="none")  # (B*T,)
            if loss_mask is not None:
                losses = losses * loss_mask.view(-1)
                denom = loss_mask.sum().clamp(min=1)
                loss = losses.sum() / denom
            else:
                loss = losses.mean()
        return logits, loss, new_cache

    @torch.no_grad()
    def generate(self, prompt, max_new=80, temperature=1.0, top_k=0, top_p=0.0,
                 stop_char=None):
        """朴素自回归（每步全量重算，用于采样展示；性能 demo 在 part4 单独实现）。"""
        self.eval()
        idx = prompt.clone() if isinstance(prompt, torch.Tensor) else torch.tensor([prompt])
        for _ in range(max_new):
            inp = idx[:, -self.cfg.block_size:]
            logits, _, _ = self(inp)
            logits = logits[:, -1, :] / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p and top_p > 0.0:
                probs = F.softmax(logits, dim=-1)
                sorted_p, inds = probs.sort(descending=True)
                cum = sorted_p.cumsum(-1)
                keep = cum - sorted_p <= top_p
                probs = probs.scatter(-1, inds, torch.where(keep, sorted_p,
                                      torch.zeros_like(sorted_p)))
                probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_char is not None and int(nxt[0, 0]) == stop_char:
                break
        return idx


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (not trainable_only or p.requires_grad))
