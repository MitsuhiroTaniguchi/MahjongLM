"""CCC critic — step 1: 4-player, action-PROCESSED, LoRA fine-tuned value critic
for the placement (dan-point) objective.  Fixes the two implementation faults of
the linear-readout c*:
  A1  mixed 3p/4p fit  -> 4-PLAYER ONLY (seat_count==4, imperfect view).
  A2  linear readout of the FROZEN imitation hidden -> LoRA fine-tune so the
      representation is shaped for VALUE, with an action-PROCESSED head.

For each viewer decision (bos-prefixed sequence; the model's in-distribution
input):
  state hidden  = hs[pos]      (token before the action)         -> V(s)=vhead(.)
  action hidden = hs[pos+1]    (after the model reads the action) -> Q(s,a)=qhead(.)
Both regress the game's placement dan-points Y (standardized).  The conditioning
score is the advantage  c*(s,a) = Q(s,a) - V(s).  Saved (LoRA + heads + Y stats)
for the chi^2 server and as the foundation for omniscient distillation / multi-axis.

Held-out report: Corr(Q-V, Y-V) (advantage alignment; compare to the frozen
linear-readout's 0.12) and Corr(V, Y) (state-value quality), plus the per-action
advantage R^2 gain R2(Q)-R2(V).
"""
from __future__ import annotations

import argparse
import json
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
UMA4 = [90.0, 45.0, -45.0, -90.0]   # 4p placement dan-points (1st..4th)


def viewer_placement(toks, viewer):
    for t in toks:
        m = FINAL_RANK_RE.match(t)
        if m and int(m.group(1)) == viewer:
            return int(m.group(2))
    return None


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shards", nargs="+",
                    default=[f"2024/data-{i:05d}-of-00016.arrow" for i in range(8)])
    ap.add_argument("--num-games", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default="outputs/ccc_critic")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    BOS = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<bos>")

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg).to(device)
    d = model.config.hidden_size
    qhead = nn.Linear(d, 1).to(device, torch.float32)
    vhead = nn.Linear(d, 1).to(device, torch.float32)
    for h in (qhead, vhead):
        nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    # ---- load 4-PLAYER imperfect games ----
    parts = []
    for sh in args.shards:
        dsx = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", sh, repo_type="dataset"))
        parts.append(dsx.filter(lambda v, s: v == "imperfect" and int(s) == 4,
                                input_columns=["view_type", "seat_count"]))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    rows = []
    for row in ds:
        if row["length"] <= args.max_len:
            rows.append(row)
        if len(rows) >= args.num_games:
            break
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); cut = int(0.85 * len(rows))
    tr_idx, te_idx = idx[:cut].tolist(), idx[cut:].tolist()
    # placement target stats (train)
    def game_Y(row):
        toks = tok.convert_ids_to_tokens([int(t) for t in row["input_ids"]])
        p = viewer_placement(toks, int(row["viewer_seat"]))
        return UMA4[p - 1] if p is not None else None
    ytr = [game_Y(rows[i]) for i in tr_idx]; ytr = [y for y in ytr if y is not None]
    muY, sdY = float(np.mean(ytr)), float(np.std(ytr) + 1e-6)
    print(f"4p games train={len(tr_idx)} test={len(te_idx)}  Y mean={muY:.1f} std={sdY:.1f}")

    params = [p for p in model.parameters() if p.requires_grad] + list(qhead.parameters()) + list(vhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def prep(row):
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        p = viewer_placement(toks, int(row["viewer_seat"]))
        if p is None:
            return None
        dec = iter_viewer_decisions(toks, int(row["viewer_seat"]))
        dec = [(pos, dt) for (pos, dt, seat) in dec if pos >= 1 and pos + 1 <= len(ids)]
        if not dec:
            return None
        yz = (UMA4[p - 1] - muY) / sdY
        return ids, dec, yz

    def run(order, train):
        model.train(train)
        se = 0.0; nb = 0; step = 0; opt.zero_grad()
        Qs, Vs, Ys = [], [], []
        for ri in order:
            pk = prep(rows[ri])
            if pk is None:
                continue
            ids, dec, yz = pk
            x = torch.tensor([[BOS] + ids], device=device)
            with torch.set_grad_enabled(train):
                hs = model(x, output_hidden_states=True).hidden_states[-1][0]  # [L+1, d]
                pos = torch.tensor([p for (p, _) in dec], device=device)
                Q = qhead(hs[pos + 1].float()).squeeze(-1)   # action-processed
                V = vhead(hs[pos].float()).squeeze(-1)        # pre-action state
                y = torch.full((len(dec),), yz, device=device)
                loss = F.mse_loss(Q, y) + F.mse_loss(V, y)
                if train:
                    (loss / args.accum).backward(); step += 1
                    if step % args.accum == 0:
                        opt.step(); opt.zero_grad()
                else:
                    Qs.append(Q.detach().float().cpu().numpy()); Vs.append(V.detach().float().cpu().numpy())
                    Ys.append(y.cpu().numpy())
            se += float(loss); nb += 1
        if train:
            return se / max(nb, 1)
        Q = np.concatenate(Qs); V = np.concatenate(Vs); Y = np.concatenate(Ys)
        return Q, V, Y

    for ep in range(args.epochs):
        tr = run(rng.permutation(tr_idx).tolist(), True)
        Q, V, Y = run(te_idx, False)
        Qt = torch.tensor(Q); Vt = torch.tensor(V); Yt = torch.tensor(Y)
        adv = Qt - Vt; yres = Yt - Vt
        r2V = 1 - float(((Yt - Vt) ** 2).mean() / (Yt.var() + 1e-9))
        r2Q = 1 - float(((Yt - Qt) ** 2).mean() / (Yt.var() + 1e-9))
        print(f"[ep {ep+1}] train_mse={tr:.4f}  Corr(adv=Q-V, Y-V)={corr(adv, yres):+.4f}  "
              f"Corr(V,Y)={corr(Vt, Yt):+.4f}  R2(Q)-R2(V)={r2Q-r2V:+.4f}  (linear-readout ref 0.12)")

    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    torch.save({"qhead": qhead.state_dict(), "vhead": vhead.state_dict(),
                "muY": muY, "sdY": sdY}, out / "ccc_critic_heads.pt")
    print(f"[done] saved LoRA critic + heads to {out}")


if __name__ == "__main__":
    main()
