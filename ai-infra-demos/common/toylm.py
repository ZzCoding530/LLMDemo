# -*- coding: utf-8 -*-
"""Part4 共用的 tiny 语言模型：训练一次后缓存到 outputs/，供 KV/量化/投机/采样 demo 复用。"""
import os
import torch
from . import corpus
from .tinygpt import TinyGPT, GPTConfig
from .train import PretrainStream, train_lm

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "cache_tinygpt.pt")


def load_or_train(seed=0, steps=400, force=False, n_lines=2000):
    """返回 (model, 语料文本)。缓存命中直接加载。"""
    if os.path.exists(CACHE) and not force:
        cfg = GPTConfig(vocab_size=corpus.VOCAB_SIZE, block_size=64,
                        n_layer=2, n_head=4, n_embd=64, pos_kind="learned")
        m = TinyGPT(cfg)
        m.load_state_dict(torch.load(CACHE, map_location="cpu", weights_only=True))
        m.eval()
        text = corpus.make_plain(n_lines, seed=seed)
        return m, text
    text = corpus.make_plain(n_lines, seed=seed)
    cfg = GPTConfig(vocab_size=corpus.VOCAB_SIZE, block_size=64,
                    n_layer=2, n_head=4, n_embd=64, pos_kind="learned")
    torch.manual_seed(0)
    m = TinyGPT(cfg)
    tr = PretrainStream(text)
    va = PretrainStream(corpus.make_plain(200, seed=9), seed=77)
    train_lm(m, tr, steps=steps, bs=16, val_data=va, log=False)
    m.eval()
    torch.save(m.state_dict(), CACHE)
    return m, text
