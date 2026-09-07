# -*- coding: utf-8 -*-
"""考点12·13 LoRA 从零实现：零初始化 + 可训练参数量/显存对比 + tiny 微调 AB
类型：AB 对比实验（torch 手写 LoRALinear，CPU 可跑；真实 7B QLoRA 在 gpu/ 分支）

讲解思路（备课用）：
1. LoRA 直觉：ΔW = (α/r)·B·A；冻结原 W，只学低秩 A/B；
   B 初始化 0（A 用高斯）→ 训练一开始输出与冻结模型完全一致（一致性断言），
   之后逐步偏离；这保证微调"从原始模型出发"而非从噪声出发；
2. 为什么省显存/省算力：原 W 不参与梯度/优化器状态 → 参数量占比 <1%；
   （完整显存分解在 part5 的 d22/d23 讲）
3. 更新哪些参数（考点13）：A/B 只加在 W 上；反向时 W 的梯度为 0；
   一般把 LoRA 挂在 Attention 的 q/k/v/o 上（任务相关）而不是 FFN（通用知识）；
   demo 同预算对比：全量 vs LoRA(attn) vs LoRA(ffn)。
运行：python d11_lora_from_scratch.py [--steps 200] [--r 4]
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, make_qa, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream, SFTDataset, eval_loss


class LoRALinear(nn.Module):
    """y = x·W^T + (α/r)·(x·A)·B^T；W 冻结，B 初始 0。"""
    def __init__(self, base: nn.Linear, r=4, alpha=8.0, seed=0):
        super().__init__()
        self.r, self.alpha = r, alpha
        self.base = base
        self.base.weight.requires_grad_(False)
        if base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        g = torch.Generator().manual_seed(seed)
        # A: (d_in, r) 用 kaiming 缩放；B: (d_out, r)=0 → 初始 ΔW=0
        self.A = nn.Parameter(torch.randn(d_in, r, generator=g) * (1.0 / r) ** 0.5)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r

    def forward(self, x):
        base_out = F.linear(x, self.base.weight, self.base.bias)
        return base_out + self.scaling * (x @ self.A) @ self.B.T

    def lora_delta_norm(self):
        with torch.no_grad():
            dW = self.scaling * (self.A @ self.B.T)
            return float(dW.norm())


def freeze_non_lora(model):
    """只保留 LoRALinear 的 A/B 可训练，其余全部冻结。"""
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith(".A") or n.endswith(".B"))


def patch_gpt(model, targets, r=4, alpha=8.0):
    """把 TinyGPT 中 targets 命名的 nn.Linear 替换成 LoRALinear（原位替换）。"""
    names = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(t in n for t in targets):
            names.append(n)
    # 逐层替换（注意 nn.Module 里名字带点的模块用 setattr 到父级）
    for n in names:
        parent = model
        parts = n.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        base = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRALinear(base, r=r, alpha=alpha))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--qlora-note", action="store_true",
                    help="提示 QLoRA=4bit 量化底座+LoRA（本 demo 用 bf16 底座演示机制）")
    args = ap.parse_args()

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=64)

    # ============ 1) B=0 一致性断言 ============
    torch.manual_seed(0)
    base_model = TinyGPT(cfg)
    x = torch.randint(0, VOCAB_SIZE, (2, 32))
    orig_out, _, _ = base_model(x)
    # 手工把一个 Linear 换 LoRA 后输出不变
    lora_net = TinyGPT(cfg)
    lora_net.load_state_dict(base_model.state_dict())
    patch_gpt(lora_net, ["wq"], r=args.r)
    lora_out, _, _ = lora_net(x)
    diff = (orig_out - lora_out).abs().max().item()
    print(f"[断言] B=0 初始化时 LoRA 模型输出与冻结模型逐元素一致？ max|Δ|={diff:.2e}（应≈0）")

    # ============ 2) 参数量/可训练占比 ============
    def trainable_counts(m):
        total = sum(p.numel() for p in m.parameters())
        tra = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, tra

    # ============ 3) AB：全量 vs LoRA(attn qkv/o) vs LoRA(ffn) ============
    qa_text = make_qa(150, seed=6)
    tr_sft = SFTDataset(qa_text, seed=7)
    val_sft = SFTDataset(make_qa(40, seed=8), seed=10)
    # 预训练一个 base（用普通语料）当底座
    torch.manual_seed(0)
    base_pt = TinyGPT(cfg)
    from common.train import train_lm
    train_lm(base_pt, PretrainStream(make_plain(1200, seed=5)), steps=180, bs=16, log=False)

    print(f"\n===== tiny SFT：全量 vs LoRA(r={args.r}) =====")
    results = {}

    def sft_eval(m):
        return eval_loss(m, val_sft, sft=True, batches=6)

    # --- 全量 ---
    m_full = TinyGPT(cfg); m_full.load_state_dict(base_pt.state_dict())
    total, tra = trainable_counts(m_full)
    train_lm(m_full, tr_sft, steps=args.steps, bs=16, sft=True, log=False)
    results["full"] = sft_eval(m_full)
    print(f"[full ] 可训练占比 {tra/total*100:5.1f}%   val={results['full']:.4f}")

    # --- LoRA on attn(q/k/v/o) ---
    m_lora_a = TinyGPT(cfg); m_lora_a.load_state_dict(base_pt.state_dict())
    patch_gpt(m_lora_a, ["wq", "wk", "wv", "wo"])
    freeze_non_lora(m_lora_a)
    train_lm(m_lora_a, tr_sft, steps=args.steps, bs=16, sft=True, log=False)
    total, tra = trainable_counts(m_lora_a)
    results["lora_attn"] = sft_eval(m_lora_a)
    print(f"[lora-attr] 可训练占比 {tra/total*100:5.3f}%   val={results['lora_attn']:.4f}")

    # --- LoRA on ffn(mlp) ---
    m_lora_f = TinyGPT(cfg); m_lora_f.load_state_dict(base_pt.state_dict())
    patch_gpt(m_lora_f, ["mlp"])
    freeze_non_lora(m_lora_f)
    train_lm(m_lora_f, tr_sft, steps=args.steps, bs=16, sft=True, log=False)
    total, tra = trainable_counts(m_lora_f)
    results["lora_ffn"] = sft_eval(m_lora_f)
    print(f"[lora-ffn ] 可训练占比 {tra/total*100:5.3f}%   val={results['lora_ffn']:.4f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = list(results)
    ax.bar(names, [results[n] for n in names],
           color=["#C44E52", "#4C72B0", "#DD8452"])
    ax.set_ylabel("SFT answer-only val loss"); ax.set_title("full vs LoRA(attn) vs LoRA(ffn)")
    ax.grid(alpha=.3, axis="y")
    viz.save(fig, "d11_lora_compare.png")

    # W 的梯度应为 0（验证"LoRA 不动原权重"）
    g = m_lora_a.blocks[0].attn.wq.A.grad
    if g is not None:
        print("\n[验证] LoRA 训练后 base W 梯度为 0：",
              all(p.grad is None or float(p.grad.abs().max()) == 0
                  for p in m_lora_a.parameters() if not p.requires_grad))
    print("\n结论：LoRA 只训几个百分点参数(大模型上 <1%)即接近全量效果(同预算步数下略差，加步数/学习率可追)；"
          "挂在 attention(q/k/v/o) 上通常优于挂 FFN。真实 QLoRA(4bit 底座) 大模型实验见 gpu/g01_qlora_7b_qkv_vs_ffn.py。")


if __name__ == "__main__":
    main()
