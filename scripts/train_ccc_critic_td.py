"""CCC critic — the clever offline extraction: VALUE-BOOTSTRAPPED (TD) advantage.

Why step-1 found ~0 placement advantage: it regressed Q(s,a) to the TERMINAL
placement Y, whose variance (all future rounds + luck) swamps the one-step action
effect. The fix is NOT self-play (placement is just as distal there) but a better
offline TARGET: bootstrap on the strong value V.

  V(s)    : MC target = uma(final placement) Y           (placement equity; Corr~0.59)
  Q(s,a)  : TD target = V(s_next)  [stop-grad]            (value of the NEXT viewer state)
            (the action determines s_next: deal-in crashes equity, good shape keeps it;
             the action-processed hidden phi(s,a)=hs[pos+1] encodes that next state)
  A(s,a)  = Q(s,a) - V(s) = one-step placement-equity change  -> the improvement direction.

This is a proper action-VALUE (not a deviation), so argmax/condition picks genuinely
high-equity moves and does NOT select OOD catastrophes (break a triplet -> phi encodes
a bad resulting hand -> low V(s_next) -> low Q). Captures offense (shape/value) AND
defense (deal-in) via V(s_next).  Saved (LoRA + V/Q heads) for the chi^2 server.

Held-out: Corr(V,Y) (value), and the one-step advantage's consistency
Corr(Q-V, V(s_next)-V).  The real test is the match (deploy via chi^2).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from gpt2.viewer_decisions import iter_viewer_decisions
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
UMA4 = [90.0, 45.0, -45.0, -90.0]


def viewer_placement(toks, viewer):
    for t in toks:
        m = FINAL_RANK_RE.match(t)
        if m and int(m.group(1)) == viewer:
            return int(m.group(2))
    return None


def corr(x, y):
    x = (x - x.mean()).double(); y = (y - y.mean()).double()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shards", nargs="+", default=[f"2024/data-{i:05d}-of-00016.arrow" for i in range(8)])
    ap.add_argument("--num-games", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default="outputs/ccc_critic_td")
    args = ap.parse_args()
    device = "cuda"; torch.manual_seed(0)
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    BOS = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<bos>")

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.use_cache = False; base.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg).to(device)
    d = model.config.hidden_size
    vhead = nn.Linear(d, 1).to(device, torch.float32)
    qhead = nn.Linear(d, 1).to(device, torch.float32)
    for h in (vhead, qhead):
        nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    parts = []
    for sh in args.shards:
        dsx = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", sh, repo_type="dataset"))
        parts.append(dsx.filter(lambda v, s: v == "imperfect" and int(s) == 4,
                                input_columns=["view_type", "seat_count"]))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    rows = [r for r in (row for row in ds) if r["length"] <= args.max_len]
    rows = rows[:args.num_games]
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); cut = int(0.85 * len(rows))
    tr_idx, te_idx = idx[:cut].tolist(), idx[cut:].tolist()

    def prep(row):
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        p = viewer_placement(toks, int(row["viewer_seat"]))
        if p is None:
            return None
        dec = [pos for (pos, dt, seat) in iter_viewer_decisions(toks, int(row["viewer_seat"]))
               if pos >= 1 and pos + 1 <= len(ids)]
        if len(dec) < 2:
            return None
        return ids, sorted(dec), UMA4[p - 1]

    ytr = [prep(rows[i])[2] for i in tr_idx if prep(rows[i])]
    muY, sdY = float(np.mean(ytr)), float(np.std(ytr) + 1e-6)
    print(f"4p train={len(tr_idx)} test={len(te_idx)}  Y~N({muY:.1f},{sdY:.1f})")

    params = [p for p in model.parameters() if p.requires_grad] + list(vhead.parameters()) + list(qhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def run(order, train):
        model.train(train); step = 0; opt.zero_grad()
        Qs, Vs, Vn, Ys = [], [], [], []
        for ri in order:
            pk = prep(rows[ri])
            if pk is None:
                continue
            ids, dec, y = pk; yz = (y - muY) / sdY
            x = torch.tensor([[BOS] + ids], device=device)
            with torch.set_grad_enabled(train):
                hs = model(x, output_hidden_states=True).hidden_states[-1][0]
                pos = torch.tensor(dec, device=device)
                V = vhead(hs[pos].float()).squeeze(-1)            # V(s_i)
                Q = qhead(hs[pos + 1].float()).squeeze(-1)        # Q(s_i, a_i)  (action-processed)
                # TD target for Q: V(s_{i+1}) for i<last (stop-grad), yz for last
                Vnext = torch.empty_like(V)
                Vnext[:-1] = V.detach()[1:]
                Vnext[-1] = yz
                loss = F.mse_loss(V, torch.full_like(V, yz)) + F.mse_loss(Q, Vnext)
                if train:
                    (loss / args.accum).backward(); step += 1
                    if step % args.accum == 0:
                        opt.step(); opt.zero_grad()
                else:
                    Qs.append(Q.detach().float().cpu().numpy()); Vs.append(V.detach().float().cpu().numpy())
                    Vn.append(Vnext.detach().float().cpu().numpy()); Ys.append(np.full(len(dec), yz))
        if train:
            return None
        return (np.concatenate(Qs), np.concatenate(Vs), np.concatenate(Vn), np.concatenate(Ys))

    for ep in range(args.epochs):
        run(rng.permutation(tr_idx).tolist(), True)
        Q, V, Vn, Y = run(te_idx, False)
        Qt, Vt, Vnt, Yt = (torch.tensor(z) for z in (Q, V, Vn, Y))
        adv = Qt - Vt
        print(f"[ep {ep+1}] Corr(V,Y)={corr(Vt, Yt):+.4f}  "
              f"Corr(TD-adv=Q-V, V(s')-V)={corr(adv, Vnt - Vt):+.4f}  "
              f"Corr(Q-V, Y-V)={corr(adv, Yt - Vt):+.4f}  adv_std={float(adv.std()):.3f}")

    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    torch.save({"vhead": vhead.state_dict(), "qhead": qhead.state_dict(), "muY": muY, "sdY": sdY},
               out / "ccc_critic_heads.pt")
    print(f"[done] saved TD critic to {out}")


if __name__ == "__main__":
    main()
