"""CCC critic — step I3: MULTI-AXIS action-processed critic (4p, LoRA).

Step-1 showed: fine-tuned V(s) for terminal placement is strong (Corr 0.59) but
the per-action placement ADVANTAGE is ~0 (action barely moves terminal placement
given state, in imperfect expert data). So the extractable offline action signal
must be the IMMEDIATE, action-attributable one. This critic predicts, from the
action-PROCESSED hidden, multiple axes and reports which carry real action-advantage:

  placement : Q_p, V_p -> dan-points Y          (objective; advantage ~0 offline)
  round     : Q_r, V_r -> viewer round score-delta (denser, more attributable)
  deal-in   : D        -> was this discard RON'd  (binary; the discard directly controls)

Conditioning score (extractable offline action signal) = the immediate-attributable
advantage, e.g. round-delta advantage A_r=Q_r-V_r and/or -deal-in risk. Reports
held-out: Corr(A_p,Y_p-V_p), Corr(A_r,Y_r-V_r), deal-in AUC, so we SEE which axis
is non-zero (i.e. extractable). Saves LoRA + heads for the chi^2 server.
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
from gpt2.round_outcome import viewer_round_deltas
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
DISCARD_RE = re.compile(r"^discard_(\d+)_")
RON_RE = re.compile(r"^take_react_(\d+)_ron$")
UMA4 = [90.0, 45.0, -45.0, -90.0]


def viewer_placement(toks, viewer):
    for t in toks:
        m = FINAL_RANK_RE.match(t)
        if m and int(m.group(1)) == viewer:
            return int(m.group(2))
    return None


def dealin_at(toks, i, seat):
    for j in range(i + 1, min(i + 40, len(toks))):
        tj = toks[j]
        if tj.startswith("draw_") or DISCARD_RE.match(tj) or tj == "round_start":
            break
        m = RON_RE.match(tj)
        if m and int(m.group(1)) != seat:
            return 1.0
    return 0.0


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def auc(score, lab):
    score = np.asarray(score); lab = np.asarray(lab)
    npos = int(lab.sum()); nneg = int((lab == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(1, len(score) + 1)
    return float((rk[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--shards", nargs="+", default=[f"2024/data-{i:05d}-of-00016.arrow" for i in range(8)])
    ap.add_argument("--num-games", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default="outputs/ccc_critic_multiaxis")
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
    heads = {k: nn.Linear(d, 1).to(device, torch.float32) for k in ("qp", "vp", "qr", "vr", "di")}
    for h in heads.values():
        nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    parts = []
    for sh in args.shards:
        dsx = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", sh, repo_type="dataset"))
        parts.append(dsx.filter(lambda v, s: v == "imperfect" and int(s) == 4,
                                input_columns=["view_type", "seat_count"]))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    rows = [r for r in (row for row in ds) if r["length"] <= args.max_len]
    rows = rows[:args.num_games] if len(rows) >= args.num_games else rows
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); cut = int(0.85 * len(rows))
    tr_idx, te_idx = idx[:cut].tolist(), idx[cut:].tolist()

    def prep(row):
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        p = viewer_placement(toks, int(row["viewer_seat"]))
        if p is None:
            return None
        rinfo = viewer_round_deltas(toks, viewer_seat=int(row["viewer_seat"]), seat_count=4)
        pos2delta = {}
        for ri in rinfo:
            for j in range(ri["start"], ri["end"]):
                pos2delta[j] = ri["delta"]
        recs = []
        for (pos, dt, seat) in iter_viewer_decisions(toks, int(row["viewer_seat"])):
            if pos < 1 or pos + 1 > len(ids):
                continue
            di = dealin_at(toks, pos, seat) if dt == 0 else -1.0   # deal-in only meaningful for discards
            recs.append((pos, dt, float(pos2delta.get(pos, 0)), di))
        if not recs:
            return None
        return ids, recs, UMA4[p - 1]

    # target stats (train)
    Yp, Yr = [], []
    for i in tr_idx:
        pk = prep(rows[i])
        if pk:
            Yp.append(pk[2]); Yr += [r[2] for r in pk[1]]
    muP, sdP = float(np.mean(Yp)), float(np.std(Yp) + 1e-6)
    muR, sdR = float(np.mean(Yr)), float(np.std(Yr) + 1e-6)
    print(f"4p train={len(tr_idx)} test={len(te_idx)}  Yp~N({muP:.1f},{sdP:.1f})  Yr~N({muR:.0f},{sdR:.0f})")

    params = [p for p in model.parameters() if p.requires_grad] + [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    pos_w = torch.tensor([60.0], device=device)

    def run(order, train):
        model.train(train); step = 0; opt.zero_grad()
        acc = {k: [] for k in ("Qp", "Vp", "Qr", "Vr", "D", "yp", "yr", "di")}
        for ri in order:
            pk = prep(rows[ri])
            if pk is None:
                continue
            ids, recs, yp = pk
            x = torch.tensor([[BOS] + ids], device=device)
            with torch.set_grad_enabled(train):
                hs = model(x, output_hidden_states=True).hidden_states[-1][0]
                pos = torch.tensor([r[0] for r in recs], device=device)
                act = hs[pos + 1].float(); st = hs[pos].float()
                Qp = heads["qp"](act).squeeze(-1); Vp = heads["vp"](st).squeeze(-1)
                Qr = heads["qr"](act).squeeze(-1); Vr = heads["vr"](st).squeeze(-1)
                D = heads["di"](act).squeeze(-1)
                ypz = torch.full((len(recs),), (yp - muP) / sdP, device=device)
                yrz = torch.tensor([(r[2] - muR) / sdR for r in recs], device=device)
                dimask = torch.tensor([r[3] >= 0 for r in recs], device=device)
                dilab = torch.tensor([max(r[3], 0.0) for r in recs], device=device)
                loss = F.mse_loss(Qp, ypz) + F.mse_loss(Vp, ypz) + F.mse_loss(Qr, yrz) + F.mse_loss(Vr, yrz)
                if dimask.any():
                    loss = loss + F.binary_cross_entropy_with_logits(D[dimask], dilab[dimask], pos_weight=pos_w)
                if train:
                    (loss / args.accum).backward(); step += 1
                    if step % args.accum == 0:
                        opt.step(); opt.zero_grad()
                else:
                    for k, v in [("Qp", Qp), ("Vp", Vp), ("Qr", Qr), ("Vr", Vr), ("D", D)]:
                        acc[k].append(v.detach().float().cpu().numpy())
                    acc["yp"].append(ypz.cpu().numpy()); acc["yr"].append(yrz.cpu().numpy())
                    acc["di"].append(np.array([r[3] for r in recs]))
        if not train:
            return {k: np.concatenate(v) for k, v in acc.items()}
        return None

    for ep in range(args.epochs):
        run(rng.permutation(tr_idx).tolist(), True)
        a = run(te_idx, False)
        Qp, Vp, Qr, Vr, D = (torch.tensor(a[k]) for k in ("Qp", "Vp", "Qr", "Vr", "D"))
        yp = torch.tensor(a["yp"]); yr = torch.tensor(a["yr"]); di = a["di"]
        msk = di >= 0
        adv_p = corr(Qp - Vp, yp - Vp); adv_r = corr(Qr - Vr, yr - Vr)
        di_auc = auc(D.numpy()[msk], di[msk])
        print(f"[ep {ep+1}] placement adv Corr(Qp-Vp,Yp-Vp)={adv_p:+.4f} | "
              f"round-delta adv Corr(Qr-Vr,Yr-Vr)={adv_r:+.4f} (Corr(Vr,Yr)={corr(Vr,yr):+.3f}) | "
              f"deal-in AUC={di_auc:.4f}")

    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    torch.save({**{k: h.state_dict() for k, h in heads.items()},
                "muP": muP, "sdP": sdP, "muR": muR, "sdR": sdR}, out / "ccc_critic_heads.pt")
    print(f"[done] saved multi-axis critic to {out}")


if __name__ == "__main__":
    main()
