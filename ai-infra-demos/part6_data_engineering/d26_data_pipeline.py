# -*- coding: utf-8 -*-
"""考点31 数据处理 pipeline：清洗 / 去重 / 配比采样 漏斗（数据工程让每个 token 更值钱）
类型：演示实现（Python 处理管线 + 漏斗图，纯 CPU；对应表格 Demo31）

讲解思路（备课用）：
1. 数据工程 = 质量过滤 → 去重(含语义级近重复) → 配比采样 三步漏斗；
2. 清洗：去 HTML/控制符/空行/过短行；质量过滤：重复率、语言启发式；
3. 去重：精确去重最简单；近重复用 MinHash-LSH（本 demo 用 n-gram Jaccard 阈值做轻量实现）；
4. 配比：按"领域"分组采样，输出训练 jsonl；给"数据枯竭"应对（回源/合成/配比/质量优先）。
运行：python d26_data_pipeline.py
"""
import re
import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, make_qa


def jaccard(a, b, n=5):
    def grams(s):
        return set(zip(*[s[i:] for i in range(n)])) if len(s) >= n else {s}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 1.0 if a == b else 0.0
    return len(ga & gb) / len(ga | gb)


def pipeline(raw_docs):
    n0 = len(raw_docs)
    # 1) 清洗
    cleaned = []
    for d in raw_docs:
        d = re.sub(r"<[^>]+>", "", d)          # html
        d = re.sub(r"[\x00-\x1f]+", " ", d)     # 控制符
        d = " ".join(d.split())
        if len(d) > 10:
            cleaned.append(d)
    n1 = len(cleaned)
    # 2) 质量过滤（启发式：字符种类丰富度 & 非字母比例小）
    kept = []
    for d in cleaned:
        letters = sum(c.isalpha() for c in d)
        if letters / max(len(d), 1) > 0.5:
            kept.append(d)
    n2 = len(kept)
    # 3) 精确 + 近重复去重（n-gram Jaccard >=0.8 视为重复）
    seen = []
    dedup = []
    for d in kept:
        if any(jaccard(d, s) >= 0.8 for s in seen):
            continue
        seen.append(d)
        dedup.append(d)
    n3 = len(dedup)
    # 4) 领域配比：按关键词分组采样（demo 用两组伪领域）
    dom_a = [d for d in dedup if "near the" in d]
    dom_b = [d for d in dedup if "near the" not in d]
    print(f"原始 {n0} 条 → 清洗后 {n1} → 质量过滤 {n2} → 去重后 {n3} "
          f"（累计保留 {n3/n0*100:.0f}%）")
    return n0, n1, n2, n3, dom_a, dom_b


def main():
    rng = np.random.default_rng(0)
    raw = make_plain(800, seed=5).splitlines()
    qa = make_qa(50, seed=6).splitlines()
    raw_docs = []
    for d in raw[:700]:
        raw_docs.append(d)
    # 人为加噪：HTML、短行、精确重复、近重复（改一个词）、QA 行
    for d in raw[700:750]:
        raw_docs.append(f"<p>{d}</p>")
    raw_docs += ["x", "", "   ", raw[0], raw[0]]
    dup2 = raw[1].replace("the", "a", 1)        # 近重复
    raw_docs += [raw[1], dup2]
    raw_docs += qa[:30]

    n0, n1, n2, n3, da, db = pipeline(raw_docs)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = ["raw", "clean", "quality\nfilter", "dedup"]
    counts = [n0, n1, n2, n3]
    axes[0].bar(steps, counts, color=["#C44E52", "#DD8452", "#F4B942", "#55A868"])
    for i, c in enumerate(counts):
        axes[0].text(i, c, str(c), ha="center", va="bottom")
    axes[0].set_title("data funnel: clean -> filter -> dedup")
    axes[0].grid(alpha=.3, axis="y")
    axes[1].bar(["domain A\n(near the)", "domain B"], [len(da), len(db)])
    axes[1].set_title("domain distribution after sampling")
    axes[1].grid(alpha=.3, axis="y")
    viz.save(fig, "d26_data_funnel.png")

    # 输出配比 jsonl 样例
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_output.jsonl")
    with open(out_path, "w") as f:
        for d in da[:5]:
            f.write(json.dumps({"text": d, "domain": "A"}) + "\n")
        for d in db[:5]:
            f.write(json.dumps({"text": d, "domain": "B"}) + "\n")
    print("配比样例 →", out_path)
    print("提示：真实场景去重用 MinHash-LSH(亿级)；配比还要控 单轮/多轮、正负样本 比例。"
          "数据枯竭应对：回源(多语言/多模态)、合成数据、配比去重、质量优先。")


if __name__ == "__main__":
    main()
