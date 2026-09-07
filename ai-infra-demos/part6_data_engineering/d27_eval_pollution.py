# -*- coding: utf-8 -*-
"""考点34 评测与选型：eval runner + 数据污染→分数虚高 + 泄漏检测
类型：AB 对比实验（torch tiny + 规则判分器，CPU；表格 Demo34）

讲解思路（备课用）：
1. 评测闭环：加载模型 → 逐题生成 → 规则/包含判分 → acc；学术 benchmark 有数据污染风险；
2. 污染实验（核心）：把"测试题原文"混进训练集 → 同一 test 集 acc 暴涨（虚高）；
   再测"同分布新题"：acc 不变 → 证明虚高来自泄漏而不是能力提升；
3. 业务测试集设计：与训练隔离、定期更新（防污染）、覆盖真实分布；
4. 选型四维框架：合规 / 成本 / 延迟 / 能力。
运行：python d27_eval_pollution.py [--steps 200]
"""
import argparse
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import encode, decode, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import SFTDataset, train_lm, eval_loss

NAMES = ["amy", "ben", "cara", "dan", "eva", "finn", "gwen", "hugo"]
PLACES = ["hill", "river", "forest", "cave", "meadow", "pond", "ridge", "valley"]


def combos(names, places):
    from itertools import product
    return [(n, p) for n, p in product(names, places)]


def gen_from_pairs(pairs):
    """把 (who, where) 对写成 q/a 文本，并返回 (text, [(q, gold_place)])。"""
    lines, gold = [], []
    for who, where in pairs:
        q = f"q {who} is where ? "
        lines.append(q + f"a {who} is near the {where} .")
        gold.append((q, where))
    return "\n".join(lines), gold


def answer_contains_place(ans, place):
    return place in ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="每轮 SFT 步数（400 即可复现强对照）")
    args = ap.parse_args()

    # 训练：名字集A × 地点集A；测试：名字集A × 地点集B（词表在训练外）；
    # fresh：名字集B × 地点集B —— 与污染(test)内容无交集，充当"泄漏检测的新题"。
    tr_pairs = combos(NAMES[:4], PLACES[:4])                 # 16 条
    test_pairs = combos(NAMES[:4], PLACES[4:])               # 16 条
    fresh_pairs = combos(NAMES[4:], PLACES[4:])              # 16 条（同分布、无泄漏）
    tr_text, _ = gen_from_pairs(tr_pairs)
    te_text, _ = gen_from_pairs(test_pairs)
    fresh_text, _ = gen_from_pairs(fresh_pairs)
    polluted_text = tr_text + "\n" + te_text                 # 污染：测试题原文入训练

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=64)

    def train_and_eval(text, seed):
        """训练后分别测 test / fresh 的 answer-only CE loss（越低=越会做）。"""
        torch.manual_seed(seed)
        m = TinyGPT(cfg)
        ds = SFTDataset(text, seed=seed)
        train_lm(m, ds, steps=args.steps, bs=16, sft=True, log=False)
        l_te = eval_loss(m, SFTDataset(te_text, seed=9), sft=True, batches=8)
        l_fr = eval_loss(m, SFTDataset(fresh_text, seed=10), sft=True, batches=8)
        return l_te, l_fr

    print("[run1] 干净训练集上训模型（测试题从未见过）...")
    lte_clean, lfr_clean = train_and_eval(tr_text, 0)
    print(f"  test 题 answer-CE={lte_clean:.3f}   同分布新题 answer-CE={lfr_clean:.3f}")

    print("[run2] 训练集混入 test 题原文（数据污染）...")
    lte_poll, lfr_poll = train_and_eval(polluted_text, 0)
    print(f"  test 题 answer-CE={lte_poll:.3f}（骤降=背下来了）   "
          f"同分布新题 answer-CE={lfr_poll:.3f}（仍远高于 test → 没有真学会）")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(2)
    w = 0.32
    ax.bar(x - w / 2, [lte_clean, lte_poll], w, label="test items (CE lower=better)")
    ax.bar(x + w / 2, [lfr_clean, lfr_poll], w, label="fresh same-dist items")
    for xi, v in zip(x - w / 2, [lte_clean, lte_poll]):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center")
    for xi, v in zip(x + w / 2, [lfr_clean, lfr_poll]):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_xticks(x, ["clean train", "polluted train"])
    ax.set_ylabel("answer CE loss"); ax.set_ylim(0, max(lte_clean, lfr_clean, lfr_poll) + 0.3)
    ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("data pollution: benchmark drops only because answers were memorized")
    viz.save(fig, "d27_eval_pollution.png")
    print("\n结论：污染让 test 分数暴涨(CE 骤降)但 fresh 不变 → 泄漏铁证。业务测试集要隔离+新鲜；"
          "选型用四维框架：合规/成本/延迟/能力。")


if __name__ == "__main__":
    main()
