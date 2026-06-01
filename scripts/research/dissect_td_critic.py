"""Validate the TD critic's advantage c*=Q(s,a)-V(s) on the SAME live states where
the old deviation-c* chose catastrophic moves (break a triplet/run).  If the TD
critic ranks hand-preserving / safe discards high and the catastrophes low, the
value-bootstrapped signal fixed the OOD pathology.

Loads base (pi0) + the LoRA TD critic (Q/V heads). For each dumped live decision,
computes pi0, c*_TD = Q(cand)-V(state), and shows the top picks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM
from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", default="outputs/ccc_critic_td")
    ap.add_argument("--dump", default="outputs/live_dump.jsonl")
    args = ap.parse_args()
    dev = "cuda"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer"); V = tok.get_vocab()
    base = AutoModelForCausalLM.from_pretrained("mitsutani/mahjonglm-10m", dtype=torch.float32).to(dev).eval()
    critic_base = AutoModelForCausalLM.from_pretrained("mitsutani/mahjonglm-10m", dtype=torch.float32,
                                                       output_hidden_states=True)
    critic = PeftModel.from_pretrained(critic_base, str(ROOT / args.critic)).to(dev).eval()
    ck = torch.load(ROOT / args.critic / "ccc_critic_heads.pt", map_location=dev)
    d = critic.config.hidden_size
    vhead = nn.Linear(d, 1).to(dev); vhead.load_state_dict(ck["vhead"])
    qhead = nn.Linear(d, 1).to(dev); qhead.load_state_dict(ck["qhead"])
    torch.set_grad_enabled(False)

    lines = [json.loads(l) for l in open(ROOT / args.dump, encoding="utf-8")]
    for di, dd in enumerate(lines):
        if not (any(a.startswith("discard_") for a in dd["allowed"]) and len(dd["allowed"]) >= 6):
            continue
        ids = [V[x] for x in dd["tokens"]]; cand = [V[a] for a in dd["allowed"]]
        # pi0 from base
        lp = torch.log_softmax(base(torch.tensor([ids], device=dev)).logits[0, -1][torch.tensor(cand, device=dev)].float(), -1)
        pi0 = lp.exp()
        # critic V(state), Q(cand)
        out = critic(torch.tensor([ids], device=dev), output_hidden_states=True)
        state_h = out.hidden_states[-1][0, -1]
        Vs = float(vhead(state_h.unsqueeze(0)).squeeze())
        batch = torch.tensor([ids + [c] for c in cand], device=dev)
        phi = critic(batch, output_hidden_states=True).hidden_states[-1][:, -1, :]
        Q = qhead(phi).squeeze(-1)
        cstar = (Q - Vs)
        names = [a.replace("discard_", "").replace("_tedashi", "").replace("_tsumogiri", "^") for a in dd["allowed"]]
        order = torch.argsort(cstar, descending=True)
        p0top = int(pi0.argmax())
        print(f"decision {di}: pi0-argmax={names[p0top]}(p={float(pi0[p0top]):.2f})  "
              f"TD c*-argmax={names[int(cstar.argmax())]}")
        print("   by TD c* (desc):", [(names[i], round(float(pi0[i]), 3), round(float(cstar[i]), 3)) for i in order.tolist()[:6]])
        if di >= 5:
            break


if __name__ == "__main__":
    main()
