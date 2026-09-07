# -*- coding: utf-8 -*-
"""考点30 合成数据与修正扩展律：real 与 synthetic 不同配比训 tiny 模型的效果
类型：AB 对比实验（torch tiny GPT ×3 组 + 可选模型崩溃演示，CPU）

讲解思路（备课用）：
1. 合成数据价值：无成本扩充/控分布；风险：噪声/同质/误差被放大；
2. toy 设计：real=语料A分布；syn=语料B分布(作为"合成"近似)，按 real:syn 配比
   {100:0, 70:30, 0:100} 训三个同预算 tiny GPT，在 real 的留出集上测 val loss；
   结论曲线：syn 占比过高 loss 变差（过量有害）；
3. 模型崩溃：只用"上一个模型生成的文本"迭代训练 2-3 代，loss 逐代劣化（可选演示）；
4. 引申修正扩展定律：真实+合成按比例混合，效果可用"修正扩展律"预测（配比/多样性/崩溃）。
运行：python d25_synthetic_data_mix.py [--steps 180] [--collapse 1]
"""
import argparse
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, decode, encode, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream, train_lm, eval_loss


def train_on(text, steps, block=64, seed=0):
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=block, n_layer=2, n_head=4, n_embd=64)
    torch.manual_seed(seed)
    m = TinyGPT(cfg)
    train_lm(m, PretrainStream(text), steps=steps, bs=16, log=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--collapse", type=int, default=1,
                    help="1=跑 2 代自训练崩溃演示; 0=跳过")
    args = ap.parse_args()

    real = make_plain(1600, seed=101)
    syn = make_plain(1600, seed=202)          # 另一种分布，充当"合成/模型生成"近似
    val_real = make_plain(200, seed=303)
    ratios = [(100, 0), (70, 30), (0, 100)]
    print("同预算训 3 组 tiny GPT（real:syn 配比）→ 在真实留出集上测 val loss")
    vals = {}
    for r, s in ratios:
        text = real if s == 0 else syn if r == 0 else (real[:int(len(real) * r / 100)]
                                                        + "\n" + syn[:int(len(syn) * s / 100)])
        m = train_on(text, args.steps)
        v = eval_loss(m, PretrainStream(val_real, seed=42))
        vals[(r, s)] = v
        print(f"real:{r}% syn:{s}% -> val_loss(real)={v:.4f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = [0, 30, 100]
    ax.plot(xs, [vals[(r, s)] for (r, s) in ratios], "o-")
    ax.set_xticks(xs, [f"{r}:{s}" for (r, s) in ratios])
    ax.set_xlabel("real:syn mixing ratio"); ax.set_ylabel("val loss on real held-out")
    ax.set_title("synthetic ratio effect (too much synthetic hurts)")
    ax.grid(alpha=.3)
    viz.save(fig, "d25_synthetic_mix.png")

    # ---- 可选：模型崩溃 ----
    if args.collapse:
        print("\n[模型崩溃演示] 用上一代模型生成文本训练下一代（2 代）")
        gen_text = syn
        losses = []
        for g in range(3):
            m = train_on(gen_text, args.steps, seed=g)
            v = eval_loss(m, PretrainStream(val_real, seed=42))
            losses.append(v)
            print(f"  第{g}代: 在真实留出集 val_loss={v:.4f}")
            if g < 2:
                # 用该模型在 seed 文本前缀上继续生成，作为"下一代合成语料"
                m.eval()
                seed_ids = torch.tensor([encode(make_plain(3, seed=g + 9)
                                                .replace("\n", " ") + " ")])
                gen = m.generate(seed_ids, max_new=1200, temperature=0.9)
                gen_text = decode(gen[0].tolist())
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(range(3), losses, "o-")
        ax.set_xlabel("generation (training on own outputs)")
        ax.set_ylabel("val loss on real data")
        ax.set_title("model collapse: quality degrades")
        ax.grid(alpha=.3)
        viz.save(fig, "d25_model_collapse.png")
    print("\n结论：合成数据要控制占比并做质量过滤/去重/混合，过量=有害；修正扩展律可预测最优配比。")


if __name__ == "__main__":
    main()
