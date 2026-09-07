# -*- coding: utf-8 -*-
"""考点20 KV Cache / PagedAttention / Prompt Caching
类型：AB 对比实验（torch tiny LM 实测耗时曲线 + 显存公式 + block table 模拟；表格 Demo20）

讲解思路（备课用）：
1. 无 cache 解码：每生成 1 个 token 都要把"整个前缀"重新算一遍 attention，
   第 t 步代价 O(t²) → 越生越慢；有 cache 后 prefill 只算一次，后续每步
   只对"最新 1 个 token"算 Q，K/V 直接 append → 每步 O(t·1)=O(t)；
2. 显存代价：KV cache/每 token = 2·L·n_kv·d_head·bytes；上下文翻倍 KV 翻倍 →
   长上下文推理吃显存（这里 7B-4K 算给你看）；
3. PagedAttention：KV 按固定 block 分配(block table 记录 逻辑块→物理块)，
   消除显存碎片/内部碎片并支持前缀共享；用一张小表演示逻辑/物理映射；
4. Prompt Caching：相同前缀的多请求命中缓存 → 只算增量。
运行：python d16_kv_cache.py [--gen-len 120]
"""
import argparse
import time
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import encode, VOCAB_SIZE
from common.toylm import load_or_train


@torch.no_grad()
def decode_no_cache(model, prompt, n_new):
    """每步全量重算（无 KV cache）。"""
    idx = prompt.clone()
    times = []
    for _ in range(n_new):
        inp = idx[:, -model.cfg.block_size:]
        t0 = time.perf_counter()
        logits, _, _ = model(inp)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
        times.append(time.perf_counter() - t0)
    return idx, times


@torch.no_grad()
def decode_with_cache(model, prompt, n_new):
    """prefill 一次 + 增量解码（KV cache）。"""
    idx = prompt.clone()
    cache = [None] * model.cfg.n_layer
    # prefill
    logits, _, cache = model(idx, cache=cache)
    times = []
    for _ in range(n_new):
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        t0 = time.perf_counter()
        logits, _, cache = model(nxt, cache=cache)
        times.append(time.perf_counter() - t0)
        idx = torch.cat([idx, nxt], 1)
    return idx, times


def kv_memory_formula(n_layer, n_kv_heads, d_head, L, dtype_bytes=2):
    return 2 * n_layer * n_kv_heads * d_head * dtype_bytes * L


def paged_demo(block=4, tokens=13, T=8):
    """逻辑 token 序列 -> 按 block 切 -> 物理块表（演示外部碎片问题）。"""
    nblock = int(np.ceil(tokens / block))
    logical = [f"L{i}" for i in range(nblock)]
    physical = list(range(T))
    table = {lk: f"P{p}" for lk, p in zip(logical, physical[:nblock])}
    print("\n[PagedAttention] block_size=4, 请求 tokens=%d → 逻辑块 %s" % (tokens, logical))
    print("  block table:", table)
    print("  13 tokens 只需 %d 个物理块，无需 13 个连续槽位 → 碎片基本消除" % nblock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-len", type=int, default=120)
    args = ap.parse_args()

    model, text = load_or_train()
    prompt = torch.tensor([encode(text[500:560])])
    L = args.gen_len

    print("同 prompt 生成 %d tokens：no-cache 每步全量重算 vs cache 增量解码..." % L)
    _, t_no = decode_no_cache(model, prompt, L)
    _, t_ca = decode_with_cache(model, prompt, L)
    # 平滑曲线：累计时间随生成步数增长
    cum_no = np.cumsum(t_no)
    cum_ca = np.cumsum(t_ca)
    print(f"总耗时  no-cache={cum_no[-1]*1e3:.0f}ms   cache={cum_ca[-1]*1e3:.0f}ms"
          f"（最后 10 步均值：{(t_no[-10:]) and np.mean(t_no[-10:])*1e3:.2f} vs "
          f"{np.mean(t_ca[-10:])*1e3:.2f} ms/步）")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(1, L + 1)
    axes[0].plot(xs, cum_no * 1e3, label="no KV cache (recompute all)")
    axes[0].plot(xs, cum_ca * 1e3, label="with KV cache (incremental)")
    axes[0].set_xlabel("generated token #"); axes[0].set_ylabel("cumulative ms")
    axes[0].set_title("decode latency: O(T^2) vs O(T)")
    axes[0].legend(); axes[0].grid(alpha=.3)

    # KV 显存随上下文长度增长（7B 参数示意 + tiny 实测配置）
    lens = np.linspace(1000, 100000, 50).astype(int)
    kb7 = [kv_memory_formula(28, 8, 128, l) / 1e9 for l in lens]
    axes[1].plot(lens / 1000, kb7)
    axes[1].set_xlabel("context length (K)"); axes[1].set_ylabel("KV cache GB (bf16)")
    axes[1].set_title("KV cache grows linearly with context (7B, GQA-8)")
    axes[1].grid(alpha=.3)
    viz.save(fig, "d16_kv_cache.png")

    print("\n[显存公式] KV/token = 2·L·n_kv_heads·d_head·bytes")
    for L_, kv in [(4096, kv_memory_formula(28, 8, 128, 4096)),
                   (32768, kv_memory_formula(28, 8, 128, 32768))]:
        print(f"  7B/GQA-8 上下文 {L_}: KV≈{kv/1e9:.2f} GB")
    paged_demo()
    print("\n结论：KV cache 让增量解码从 O(T²) 变 O(T)；显存换时间。"
          "PagedAttention 解决 KV 碎片；Prompt Caching 让同前缀请求只算增量。")


if __name__ == "__main__":
    main()
