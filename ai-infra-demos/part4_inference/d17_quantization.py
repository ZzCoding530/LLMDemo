# -*- coding: utf-8 -*-
"""考点21 量化 INT8/INT4 + 对称/非对称 + per-tensor/per-channel + 位宽扫描
类型：演示 + 对照（numpy/torch 自实现张量级量化，CPU；真实 4bit 加载/AWQ 在 gpu/g03）

讲解思路（备课用）：
1. 对称量化：s = max|w| / 2^(b-1)-1；wq = round(w/s)；反量化 ŵ = wq·s（面试手撕四行）；
   非对称：额外存 min，用 zero-point 平移（对非对称分布更省误差，多花存储）；
2. per-tensor vs per-channel：权重矩阵按输出通道(channel)各给一个 scale 精度高很多，
   GPTQ/AWQ 用的就是 per-group(如 128 一组)量化；
3. 位宽-精度-显存 trade-off：8bit→4bit 显存减半，误差(相对 MSE/余弦)如何上升；
   演示在真实 tiny 模型权重矩阵上量化后，模型 logits 与 fp16 的偏离程度；
4. 引出：activation 比 weight 更难量化(动态范围大、有 outlier) → AWQ 保护显著通道
   （gpu 分支做真实 7B）；KV cache 可做 fp8 省一半。
运行：python d17_quantization.py
"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.toylm import load_or_train


def quantize_symmetric(w, bits):
    """对称 per-tensor。返回 (wq, s)。"""
    b = 2 ** (bits - 1) - 1
    s = w.abs().max() / b
    if s == 0:
        return w, s
    return torch.clamp(torch.round(w / s), min=-b - 1, max=b).to(torch.int32), s


def quantize_asymmetric(w, bits):
    """非对称：zero-point 平移。"""
    b = 2 ** bits - 1
    lo, hi = w.min(), w.max()
    s = (hi - lo) / b
    if s == 0:
        return w, s, 0
    zp = torch.clamp(torch.round(-lo / s), min=0, max=b)
    wq = torch.clamp(torch.round(w / s) + zp, min=0, max=b).to(torch.int32)
    return wq, s, zp


def dequant(wq, s, zp=0):
    return (wq.float() - zp) * s


def per_channel_quant(w, bits):
    """per-output-channel：每行(channel)一个 scale。w:(out,in)"""
    b = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True) / b
    wq = torch.clamp(torch.round(w / scale.clamp(min=1e-12)), min=-b - 1, max=b)
    return wq.to(torch.int32), scale


def err_metrics(w, w_hat):
    rel = ((w - w_hat) ** 2).mean() / (w ** 2).mean()
    cos = 1 - (w * w_hat).sum() / (w.norm() * w_hat.norm() + 1e-8)
    return float(rel), float(cos)


def main():
    model, _ = load_or_train()
    W = model.blocks[0].mlp[0].weight.detach()          # 真实线性层 (out,in)
    print(f"量化对象：TinyGPT 第1层 FFN 权重 {tuple(W.shape)}，fp16 原始~{(W.numel()*2)/1e6:.2f}MB")

    rows = []
    for bits in [8, 6, 4, 3]:
        wq, s = quantize_symmetric(W, bits)
        w_hat = dequant(wq, s)
        wq2, s2, zp = quantize_asymmetric(W, bits)
        w_hat2 = dequant(wq2, s2, zp)
        wq3, s3 = per_channel_quant(W, bits)
        w_hat3 = dequant(wq3, s3)
        rows.append(dict(
            bits=bits,
            sym=err_metrics(W, w_hat), asym=err_metrics(W, w_hat2),
            ch=err_metrics(W, w_hat3)))
        print(f"int{bits:<3} 对称per-tensor MSE={rows[-1]['sym'][0]:.2e} cos={rows[-1]['sym'][1]:.2e}"
              f" | 非对称 MSE={rows[-1]['asym'][0]:.2e}"
              f" | per-channel MSE={rows[-1]['ch'][0]:.2e}")

    # logits 偏离：量化全部线性层权重后看模型对同一 prompt 的 logits 变化
    dev = model
    dev_q, dev_free = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2:
            wq, s = quantize_symmetric(p.data, 4)
            dev_q.append(dequant(wq, s).to(p.dtype))
        else:
            dev_q.append(p.data.clone())
    # 替换参数做一次前向
    import torch.nn as nn
    from common.corpus import encode
    x = torch.tensor([encode("the quick fox runs near")[:50]])
    ref_logits, _, _ = model(x)
    tmp = dict(model.named_parameters())
    state = {n: p.data.clone() for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        if p.ndim == 2:
            wq, s = quantize_symmetric(p.data, 4)
            p.data.copy_(dequant(wq, s))
    q_logits, _, _ = model(x)
    logit_drift = (ref_logits - q_logits).abs().mean().item()
    for n, p in model.named_parameters():
        p.data.copy_(state[n])
    print(f"\n全部线性层 int4 后，logits 平均漂移 = {logit_drift:.4f}（说明权重量化对输出的扰动幅度）")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bits_list = [r["bits"] for r in rows]
    axes[0].semilogy(bits_list, [r["sym"][0] for r in rows], "o-", label="symmetric per-tensor")
    axes[0].semilogy(bits_list, [r["asym"][0] for r in rows], "s--", label="asymmetric")
    axes[0].semilogy(bits_list, [r["ch"][0] for r in rows], "d-.", label="per-channel")
    axes[0].set_xlabel("bits"); axes[0].set_ylabel("relative MSE")
    axes[0].set_title("quantization error vs bit-width")
    axes[0].legend(); axes[0].grid(alpha=.3, which="both")
    # 显存对比
    mems = {16: W.numel() * 2, 8: W.numel() * 1, 4: W.numel() * 0.5}
    axes[1].bar(list(mems), [mems[k] / 1e3 for k in mems])
    axes[1].set_title("weight memory vs precision (this layer, KB)")
    axes[1].set_ylabel("KB"); axes[1].grid(alpha=.3, axis="y")
    viz.save(fig, "d17_quantization.png")
    print("\n结论：位宽越低越省显存但误差越大；per-channel/group 显著降低误差；"
          "真实大模型推理量化(4bit/AWQ)与激活量化见 gpu/g03。")


if __name__ == "__main__":
    main()
