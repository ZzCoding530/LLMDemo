# -*- coding: utf-8 -*-
"""考点5 注意力变体 GQA / MQA / 滑动窗口 + KV cache 显存对比
类型：AB 对比实验（同预算训 MHA vs GQA vs MQA，CPU 可跑 tiny）

讲解思路（备课用）：
1. 推理时 KV cache 每 token 开销 = 2×n_layer×n_kv_heads×d_head×dtype_bytes；
   MHA 里 K/V 每头各一份 → KV 大；GQA 让 n 个 Q 头共享 g 组 K/V（g<n），MQA 共享 1 组；
   显存随 n_kv 线性下降，质量损失很小（GQA 论文结论：约等于 MHA）；
2. 本 demo：同一语料同 token 预算训 MHA / GQA-2 / MQA，比较 val loss 与 KV 显存；
3. 滑动窗口注意力：限制每个 query 只看前 W 个 key（mask 截断），长文成本降为 O(T·W)。
运行：python d04_gqa_mqa_attention.py [--steps 160]
"""
import argparse
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream, train_lm


def kv_bytes_per_token(n_layer, n_kv_heads, d_head, dtype_bytes=2):
    return 2 * n_layer * n_kv_heads * d_head * dtype_bytes


def kv_table():
    print("\n==== KV cache 每 token / 32K 上下文 显存对比 ====")
    print(f"{'结构':<8} {'n_head':>6} {'n_kv':>5} {'每token':>8} {'@32K':>10}")
    cases = [("MHA", 32, 32), ("GQA-8", 32, 8), ("GQA-4", 32, 4), ("MQA", 32, 1)]
    for name, nh, nkv in cases:
        per = kv_bytes_per_token(n_layer=80, n_kv_heads=nkv, d_head=128)
        print(f"{name:<8} {nh:>6} {nkv:>5} {per:>7}B {per * 32768 / 1e9:>9.2f}GB")


def run_ab(steps, block=48):
    text = make_plain(900, seed=21)
    val_text = make_plain(120, seed=22)
    base = dict(vocab_size=VOCAB_SIZE, block_size=block, n_layer=2, n_head=4,
                n_embd=64, dropout=0.0)
    cfgs = [("MHA", dict(attn_kind="mha"), "MHA"),
            ("GQA-2(4Q头共享2KV头)", dict(attn_kind="gqa", n_kv_heads=2), "GQA-2"),
            ("MQA(全部Q头共享1KV头)", dict(attn_kind="mqa"), "MQA")]
    rows = {}
    tr = PretrainStream(text)
    va = PretrainStream(val_text, seed=99)
    for name, extra, tag in cfgs:
        torch.manual_seed(0)
        model = TinyGPT(GPTConfig(**base, **extra))
        n_par = sum(p.numel() for p in model.parameters())
        _, vh = train_lm(model, tr, steps=steps, bs=12, val_data=va, log=False)
        # 每层 KV 头数（本模型实现里 GQA 的 K/V 投影宽度=n_kv*dh）
        nkv = model.blocks[0].attn.n_kv_heads
        kv_per = kv_bytes_per_token(n_layer=model.cfg.n_layer, n_kv_heads=nkv,
                                    d_head=model.cfg.n_embd // model.cfg.n_head)
        val = vh[-1][1]
        rows[tag] = val
        print(f"{name:<34} params={n_par:>7,}  kv_per_token={kv_per:>4}B  val_loss={val:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=160)
    args = ap.parse_args()
    kv_table()
    rows = run_ab(args.steps)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = list(rows)
    ax.bar(names, [rows[n] for n in names], color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("val loss"); ax.set_title("same data & budget: MHA vs GQA vs MQA")
    ax.grid(alpha=.3, axis="y")
    viz.save(fig, "d04_gqa_mqa_val_loss.png")
    print("\n结论：KV 显存随 n_kv 成比例下降；val loss 损失很小（GQA≈MHA、MQA 略差），"
          "这正是 DeepSeek/Gemma/LLaMA3 用 GQA 的原因。滑动窗口可进一步把注意力成本压到 O(T·W)。")


if __name__ == "__main__":
    main()
