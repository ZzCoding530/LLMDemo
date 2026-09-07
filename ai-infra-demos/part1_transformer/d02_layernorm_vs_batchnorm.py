# -*- coding: utf-8 -*-
"""考点3 LayerNorm 对哪个维度归一化 + BatchNorm 对照
类型：演示 + 对照（numpy 手写 + torch 微型分类器对比）

讲解思路（备课用）：
1. LayerNorm：对每个样本的最后一维(feature/hidden)统计 mean/var，γ/β 可学习 → 输出与同 batch 其它样本无关；
2. BatchNorm：对 batch 维统计（训练期用 batch 统计，推理期用滑动平均）；
3. 为什么 Transformer 用 LN 不用 BN：NLP batch 小、序列变长/短不一，BN 的 batch 统计抖动大，
   且 batch 内样本差异大时不稳定；LN 逐样本归一化天然稳定；
4. 数值实验：构造 batch=2 且第二个样本是"离群"输入，看 BN 统计被污染 / LN 不受影响；
5. 可学习参数：LN 每特征 γ/β → 参数量 = 2×hidden（回答"有没有可学习参数"追问）。
"""
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz


def layernorm_np(x, gamma, beta, eps=1e-5):
    """x:(B,L,H) 对最后一维归一化。"""
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def batchnorm_np(x, gamma, beta, eps=1e-5, mode="train", running_mean=None, running_var=None, momentum=0.1):
    """x:(B,H) 对 batch 维归一化（示意 2D，演示原理足够）。"""
    if mode == "train":
        mean = x.mean(0, keepdims=True)
        var = x.var(0, keepdims=True)
        if running_mean is None:
            running_mean, running_var = mean[0], var[0]
        else:
            running_mean = (1 - momentum) * running_mean + momentum * mean[0]
            running_var = (1 - momentum) * running_var + momentum * var[0]
    else:
        mean, var = running_mean.reshape(1, -1), running_var.reshape(1, -1)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta, running_mean, running_var


def torch_classifier_compare(batch_size=2, steps=120, seed=0):
    """tiny MLP 上 BN vs LN 的 loss 曲线（batch 极小时 BN 抖动更明显）。"""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = np.random.randn(240, 8).astype("float32")
    w = np.random.randn(8, 1).astype("float32")
    y = (X @ w > 0).astype("int64").ravel()
    Xt = torch.from_numpy(X[:80]); yt = torch.from_numpy(y[:80])
    # 线性可分但带噪声：训练 set 更大
    Xtr = torch.from_numpy(X[80:]); ytr = torch.from_numpy(y[80:])

    def make(use_bn):
        m = nn.Sequential()
        m.append(nn.Linear(8, 16))
        m.append(nn.BatchNorm1d(16) if use_bn else nn.LayerNorm(16))
        m.append(nn.ReLU())
        m.append(nn.Linear(16, 2))
        return m

    losses = {}
    for tag, use_bn in [("BN-batch2", True), ("LN-batch2", False)]:
        model = make(use_bn)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        ce = nn.CrossEntropyLoss()
        hist = []
        n = len(Xtr)
        for step in range(steps):
            perm = torch.randperm(n)[:batch_size]
            xb, yb = Xtr[perm], ytr[perm]
            model.train()
            opt.zero_grad()
            out = model(xb)
            loss = ce(out, yb)
            loss.backward()
            opt.step()
            if step % 20 == 0:
                model.eval()
                with torch.no_grad():
                    vl = ce(model(Xt), yt)
                hist.append((step, float(vl)))
        losses[tag] = hist
    # 打印收敛情况并画图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, hist in losses.items():
        xs = [h[0] for h in hist]; ys = [h[1] for h in hist]
        ax.plot(xs, ys, "o-", label=f"val loss {tag}")
        print(f"{tag} 最后 val loss = {ys[-1]:.3f} (起点 {ys[0]:.3f})")
    ax.set_xlabel("step"); ax.set_ylabel("val loss")
    ax.set_title(f"BN vs LN on tiny MLP, batch_size={batch_size}")
    ax.legend(); ax.grid(alpha=.3)
    viz.save(fig, "d02_bn_vs_ln_loss.png")
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    H = 6
    gamma = np.ones(H); beta = np.zeros(H)

    # ---- 手撕数值验证：LN 只对最后一维 ----
    x = rng.standard_normal((2, 4, H)).astype("float64")
    y = layernorm_np(x, gamma, beta)
    # LN 输出每个样本自己的 feature 维 mean≈0/std≈1
    means = y.mean(-1); stds = y.std(-1)
    print("LN: 每(样本,位置)的 feature 维 mean 应≈0、std≈1")
    print(f"  max|mean|={np.abs(means).max():.2e}  std∈[{stds.min():.3f},{stds.max():.3f}]")
    # 关键性质：把 batch 里某个样本替换成离群值，LN 输出不变（逐样本归一化）
    x2 = x.copy(); x2[1] = x2[1] * 1000 + 50
    y2 = layernorm_np(x2, gamma, beta)
    print("替换 batch 中另一样本为离群值后，样本0的 LN 输出不变：",
          np.allclose(y[0], y2[0]))
    # LN 可学习参数 = 2×H（回答追问）
    print(f"LN 可学习参数 = 2×hidden = {2 * H}（γ、β 各 H 个）")

    # ---- BN 在 batch 小时被离群样本污染 ----
    X = rng.standard_normal((8, H))
    rm = np.zeros(H); rv = np.ones(H)
    _, rm, rv = batchnorm_np(X, gamma, beta, mode="train", running_mean=rm, running_var=rv)
    X_big = X.copy(); X_big[0] += 40            # 同 batch 里混入一个离群样本
    out_normal, _, _ = batchnorm_np(X, gamma, beta, mode="train")
    out_big, _, _ = batchnorm_np(X_big, gamma, beta, mode="train")
    print("\nBN: 同 batch 混入离群样本后，其余样本归一化结果被污染：",
          f"norm2 diff = {np.abs(out_normal[1:] - out_big[1:]).max():.2f}")

    # ---- torch 微型对比 ----
    torch_classifier_compare(batch_size=args.batch)
    print("\n结论：LN 沿最后一维(hidden)归一化、与 batch 无关；BN 沿 batch 维统计、小 batch/变长场景不稳定，"
          "故 Transformer 用 LN。")


if __name__ == "__main__":
    main()
