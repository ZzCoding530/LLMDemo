# -*- coding: utf-8 -*-
"""考点8 预训练 / SFT / 对齐 三阶段流水线（同一个小 GPT）
类型：演示实现（torch tiny GPT，CPU 可跑；对应表格 Demo8）

讲解思路（备课用）：
1. 顺序：预训练(海量语料 next-token) → SFT(指令问答、answer 部分 mask loss) → RL(奖励最大化)；
   缺预训练：模型连语言都不会，直接 SFT 也答不好（附对照可选）；
   缺 SFT：只会续写不会"按问答格式回答"；缺 RL：服从性/格式有但未必对奖励目标优化；
2. 同一 prompt 让 base / sft / rl 三阶段模型生成，并排对比行为差异——这是核心演示产出；
3. RL 用最简单 bandit 式策略梯度（无 value/RM 完整管线），只演示"reward 驱动行为偏移"。
运行：python d07_three_stage_tiny_gpt.py [--pt-steps 350] [--sft-steps 250] [--rl-rounds 4]
"""
import argparse
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import viz
from common.corpus import make_plain, make_qa, encode, decode, VOCAB_SIZE
from common.tinygpt import TinyGPT, GPTConfig
from common.train import PretrainStream, SFTDataset, train_lm

A_MARKER = encode(" a ")
PROMPTS = ["q amy is where ? ", "q ben is where ? ", "q finn is where ? "]


def show_stage(name, model, ps, max_new=36):
    model.eval()
    with torch.no_grad():
        pre = torch.tensor([encode(ps)])
        ids = model.generate(pre, max_new=max_new, temperature=0.9, top_k=8)
    gen = decode(ids[0, len(pre[0]):].tolist())
    print(f"--- [{name}] {ps!r} -> {gen!r}")
    return gen


@torch.no_grad()
def compliance(model, dataset, tries=24, max_new=30):
    """规则指标：答案是否以 '.' 结尾且含 'near'（演示 RL 阶段用）。"""
    ok = 0
    for _ in range(tries):
        p = dataset.pairs[np.random.randint(len(dataset.pairs))][0]
        pre = torch.tensor([np.concatenate([p, A_MARKER]).tolist()])
        ids = model.generate(pre, max_new=max_new, temperature=0.9, top_k=6)
        gen = ids[0, len(pre[0]):].tolist()
        s = decode(gen)
        if s.rstrip().endswith(".") and "near" in s:
            ok += 1
    return ok / tries


def rl_align(model, dataset, rounds=4, batch=8, max_new=30, lr=1e-3):
    """bandit 式策略梯度：采样 -> 规则奖励 -> 最大化 (R-b)·logπ。无 RM/value。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prompts = [np.concatenate([p, A_MARKER]).tolist() for p, _ in dataset.pairs[:40]]
    print("\n[stage3 RL] bandit 策略梯度, reward=以.结尾且含near")
    for r in range(rounds):
        model.eval()
        chosen, rewards, logps = [], [], []
        for _ in range(batch):
            p = prompts[np.random.randint(len(prompts))]
            pre = torch.tensor([p])
            with torch.no_grad():
                ids = model.generate(pre, max_new=max_new, temperature=0.9, top_k=8)
            full = ids[0]
            gen_len = len(full) - len(pre[0])
            s = decode(full[len(pre[0]):].tolist())
            rew = 1.0 if (s.rstrip().endswith(".") and "near" in s) else 0.0
            chosen.append(full); rewards.append(rew)
        rw = torch.tensor(rewards)
        adv = rw - rw.mean()                      # 减均值作为 baseline，鼓励更好样本
        # 重新前向算生成段的 log 概率并做加权更新
        model.train(); opt.zero_grad()
        tot_loss = torch.tensor(0.0)
        for full, a in zip(chosen, adv):
            full = full[:model.cfg.block_size]
            logits, _, _ = model(full[:-1].unsqueeze(0))
            lg = torch.log_softmax(logits[0], -1)
            logp = lg[torch.arange(len(full) - 1), full[1:]].mean()   # 整段近似
            tot_loss = tot_loss - a * logp
        tot_loss = tot_loss / batch
        tot_loss.backward()
        opt.step()
        rate = compliance(model, dataset)
        print(f"  round {r+1}: mean_reward={rw.mean().item():.2f} "
              f"compliance(held)={rate:.2f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-steps", type=int, default=350)
    ap.add_argument("--sft-steps", type=int, default=250)
    ap.add_argument("--rl-rounds", type=int, default=4)
    args = ap.parse_args()

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=64)
    plain = make_plain(1200, seed=5)
    qa_text = make_qa(150, seed=6)
    qa_val = SFTDataset(make_qa(40, seed=7))

    # ---- 阶段1：预训练 ----
    torch.manual_seed(0)
    base = TinyGPT(cfg)
    print("[stage1 pretrain]")
    train_lm(base, PretrainStream(plain), steps=args.pt_steps, bs=16,
             val_data=PretrainStream(plain[:len(plain) // 3], seed=77), log=False)

    # ---- 阶段2：SFT ----
    torch.manual_seed(0)
    sft = TinyGPT(cfg); sft.load_state_dict(base.state_dict())
    print("\n[stage2 SFT on q/a (answer-only loss)]")
    sft_tr = SFTDataset(qa_text, seed=8)
    _, sft_vh = train_lm(sft, sft_tr, steps=args.sft_steps, bs=16,
                         sft=True, val_data=qa_val, log=False)
    print(f"  SFT answer-only val_loss 最后={sft_vh[-1][1]:.4f}")

    # ---- 阶段3：RL(可选) ----
    torch.manual_seed(0)
    rl = TinyGPT(cfg); rl.load_state_dict(sft.state_dict())
    rl_align(rl, sft_tr, rounds=args.rl_rounds)

    # ---- 三阶段同 prompt 行为对比 ----
    print("\n===== 三阶段生成对比（同一 prompt）=====")
    for ps in PROMPTS:
        print(f"\nprompt: {ps!r}")
        show_stage("base 预训练后", base, ps)
        show_stage("sft  对齐后  ", sft, ps)
        show_stage("rl   RL 后   ", rl, ps)
    print(f"\nholdout 规则指标 compliance：SFT={compliance(sft, sft_tr):.2f}  "
          f"RL={compliance(rl, sft_tr):.2f}")


if __name__ == "__main__":
    main()
