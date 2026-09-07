# -*- coding: utf-8 -*-
"""确定性合成语料生成器（CPU demo 专用）。

设计目的：
1. 零网络依赖、可复现：给定 seed 生成固定文本；
2. 自带"句子模板"结构，让 tiny GPT 可学到真实分布、loss 能收敛；
3. 同时产出 纯文本(预训练用) 和 Q/A 对(SFT 用)，让"三阶段"demo 有行为差异可观察；
4. 也提供 tiny_shakespeare.txt 风格的本地语料加载入口（若用户传入真实 txt）。

词表为字符级（nanoGPT 风格）：小写字母 + 常见标点 + 数字 + 换行。
讲解思路：面试讲"数据工程"时可用此模块演示语料如何影响训练；真实训练请换真实语料。
"""
import os
import random

CHARS = "abcdefghijklmnopqrstuvwxyz .,?!" + "\n"
VOCAB = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

NOUNS = ["dog", "cat", "fox", "bear", "bird", "wolf", "deer", "lion", "fish", "owl"]
ADJS = ["big", "small", "red", "dark", "quick", "calm", "wild", "brave", "gray", "lazy"]
VERBS = ["runs", "sleeps", "hides", "jumps", "waits", "watches", "grows", "walks", "rests", "flees"]
PLACES = ["hill", "river", "forest", "cave", "meadow", "pond", "ridge", "valley", "plain", "brook"]
NAMES = ["amy", "ben", "cara", "dan", "eva", "finn", "gwen", "hugo", "ira", "june"]


def _word(rng):
    return rng.choice(NOUNS)


def make_plain(n_lines, seed=0):
    """生成 n_lines 行"事实句"：the <adj> <noun> <verb> near the <noun> . 等模板。

    模板库足够多样，字符级 n-gram/模型能学到"像自然语言的"分布。
    """
    rng = random.Random(seed)
    lines = []
    templates = [
        lambda: f"the {rng.choice(ADJS)} {rng.choice(NOUNS)} {rng.choice(VERBS)} near the {rng.choice(NOUNS)} .",
        lambda: f"a {rng.choice(ADJS)} {rng.choice(NOUNS)} {rng.choice(VERBS)} in the {rng.choice(PLACES)} .",
        lambda: f"{rng.choice(NAMES)} sees the {rng.choice(ADJS)} {rng.choice(NOUNS)} .",
        lambda: f"the {rng.choice(NOUNS)} waits for the {rng.choice(ADJS)} {rng.choice(NOUNS)} .",
        lambda: f"on the {rng.choice(PLACES)} a {rng.choice(NOUNS)} {rng.choice(VERBS)} .",
    ]
    for _ in range(n_lines):
        lines.append(templates[rng.randrange(len(templates))]())
    return "\n".join(lines)


def make_qa(n_pairs, seed=1):
    """生成 n_pairs 个问答对，格式 'q <问题> a <答案>'。

    问题来自固定模板、答案由(名称,地点)组成，模型必须在 'q' 段之后
    学到'读问题→回答'的映射，而不是机械续写——SFT 阶段即可看到行为差异。
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        who = rng.choice(NAMES)
        where = rng.choice(PLACES)
        q = f"q {who} is where ?"
        a = f"a {who} is near the {where} ."
        pairs.append(f"{q} {a}")
    return "\n".join(pairs)


def load_text_file(path):
    """本地真实语料入口：读入并归一化为小写字幕表内字符。
    （在 AutoDL 上可放 tiny_shakespeare.txt / OpenWebText 切片等）"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read().lower()
    allowed = set(CHARS)
    return "".join(ch for ch in s if ch in allowed)


# ---------------- 字符级编码/解码（供所有 char-level LM demo 复用） ----------------

def encode(s):
    return [VOCAB[c] for c in s]


def decode(ids):
    return "".join(CHARS[i] for i in ids)


def to_id_tensor(s, dtype=None):
    import torch
    return torch.tensor(encode(s), dtype=torch.long)


if __name__ == "__main__":
    plain = make_plain(20, seed=0)
    qa = make_qa(5, seed=1)
    print("== plain sample ==")
    print(plain[:300])
    print("== qa sample ==")
    print(qa[:300])
    print(f"vocab_size={VOCAB_SIZE} charset={CHARS!r}")
