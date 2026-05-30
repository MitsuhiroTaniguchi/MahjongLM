"""Strengthen the critic to the imperfect-information ceiling by PROCESSING the
candidate action through the model (option 1).

Q(s,a) = value_head( model( prefix_s + action_a )[last] )  — the model attends
to the candidate action token, so its hidden encodes the action's consequences
(vs a per-action head read from the pre-action state, which plateaued ~0.82).
Target: the immediate, action-attributable reward (signed score change at the
round-resolution within a short window; 0 otherwise). LoRA fine-tune + scalar
value head. Rollout-free.

Reports held-out bad-event AUC (deal-in-like) — expect to approach the ~0.93
imperfect-info ceiling — and a KV-cached counterfactual rank evaluation: for
each in-support candidate discard, Q(s,a) via one cached step; rank; condition
on best; measure TV vs pi0, in-support mass, and predicted-reward improvement.
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

from gpt2.round_outcome import _decode_tenbo_run
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_")
RON_RE = re.compile(r"^take_react_(\d+)_ron$")
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
WINDOW = 30


def discard_token_sets(tok):
    by = {}
    for t, tid in tok.get_vocab().items():
        m = DISCARD_RE.match(t)
        if m:
            by.setdefault(int(m.group(1)), []).append(tid)
    return {s: sorted(v) for s, v in by.items()}


def viewer_discards(toks):
    out, seat = [], None
    for i, t in enumerate(toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        elif seat is not None and DISCARD_RE.match(t) and int(t.split("_")[1]) == seat:
            di = 0.0  # immediate attributable BAD event: this discard is ron'd
            for j in range(i + 1, min(i + WINDOW, len(toks))):
                tj = toks[j]
                if tj.startswith("draw_") or (DISCARD_RE.match(tj) and j > i + 1) or tj == "round_start":
                    break
                m = RON_RE.match(tj)
                if m and int(m.group(1)) != seat:
                    di = 1.0; break
            out.append((i, seat, di))
    return out


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
    ap.add_argument("--shards", nargs="+", default=[f"2024/data-{i:05d}-of-00016.arrow" for i in range(6)])
    ap.add_argument("--num-games", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank-games", type=int, default=150)
    ap.add_argument("--support-thresh", type=float, default=0.02)
    args = ap.parse_args()
    device = "cuda"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    dsets = discard_token_sets(tok)

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg).to(device)
    d = model.config.hidden_size
    vhead = nn.Linear(d, 1).to(device, torch.float32); nn.init.zeros_(vhead.weight); nn.init.zeros_(vhead.bias)

    parts = []
    for sh in args.shards:
        dsx = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", sh, repo_type="dataset"))
        parts.append(dsx.filter(lambda v: v == "imperfect", input_columns=["view_type"]))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    rows = []
    for row in ds:
        if row["length"] <= args.max_len:
            rows.append(row)
        if len(rows) >= args.num_games:
            break
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); cut = int(0.85 * len(rows))
    tr_idx, te_idx = idx[:cut], idx[cut:]
    print(f"games train={len(tr_idx)} test={len(te_idx)}")

    params = [p for p in model.parameters() if p.requires_grad] + list(vhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    pos_w = torch.tensor([60.0], device=device)  # deal-in ~1.5% -> class balance

    def prep(row):
        toks = tok.convert_ids_to_tokens([int(t) for t in row["input_ids"]])
        if not any(FINAL_RANK_RE.match(t) for t in toks):
            return None, None
        return [int(t) for t in row["input_ids"]], viewer_discards(toks)

    def run(train):
        model.train(train); order = tr_idx if train else te_idx
        se = n = 0.0; vacc, racc = [], []; step = 0; opt.zero_grad()
        for ri in order:
            ids, dec = prep(rows[ri])
            if not dec:
                continue
            x = torch.tensor([ids], device=device)
            with torch.set_grad_enabled(train):
                hs = model(x, output_hidden_states=True).hidden_states[-1][0]
                pos = torch.tensor([p for (p, _, _) in dec], device=device)   # AT the discard token
                r = torch.tensor([rr for (_, _, rr) in dec], device=device, dtype=torch.float32)
                v = vhead(hs[pos].float()).squeeze(-1)            # deal-in logit
                loss = F.binary_cross_entropy_with_logits(v, r, pos_weight=pos_w)
                if train:
                    (loss / args.accum).backward(); step += 1
                    if step % args.accum == 0:
                        opt.step(); opt.zero_grad()
                else:
                    vacc.append(v.detach().cpu().numpy()); racc.append(r.cpu().numpy())
        if not train:
            y = np.concatenate(racc).astype(int); vv = np.concatenate(vacc)
            return auc(vv, y), float(y.mean())
        return None

    for ep in range(args.epochs):
        run(True)
        au, frac = run(False)
        print(f"[epoch {ep+1}] heldout dealin_AUC={au:.4f}  dealin_frac={frac:.4f}  "
              f"(per-action-head ~0.82, frozen hidden@pos ~0.93)")

    # ---- KV-cached counterfactual ranking eval ----
    model.eval()
    tv_list, ins_list, imp_list = [], [], []
    nstates = 0
    with torch.no_grad():
        for ri in te_idx[:args.rank_games]:
            ids, dec = prep(rows[ri])
            if not dec:
                continue
            base_logits_all = model(torch.tensor([ids], device=device)).logits[0]
            for (pos, seat, _) in dec[::3]:   # subsample decisions
                if seat not in dsets:
                    continue
                cand = torch.tensor(dsets[seat], device=device)
                lp = torch.log_softmax(base_logits_all[pos - 1][cand].float(), -1)
                pi0 = lp.exp()
                sup = pi0 >= args.support_thresh
                if sup.sum() < 2:
                    continue
                # prefix forward up to pos (exclusive) with cache
                pref = torch.tensor([ids[:pos]], device=device)
                past = model(pref, use_cache=True).past_key_values
                risk = torch.full((len(cand),), 1e9, device=device)
                for ci in torch.nonzero(sup).squeeze(-1).tolist():
                    step = model(cand[ci].view(1, 1), past_key_values=past, use_cache=True,
                                 output_hidden_states=True)
                    risk[ci] = torch.sigmoid(vhead(step.hidden_states[-1][0, -1].float()).squeeze(-1))
                # policies over candidates; best = SAFEST (lowest deal-in risk)
                p0 = torch.softmax(lp.masked_fill(~sup, -1e9), -1)
                best = risk.argmin()
                p_best = torch.zeros_like(p0); p_best[best] = 1.0
                tv_list.append(float(0.5 * (p_best - p0).abs().sum()))
                ins_list.append(float(sup[best].item()))
                risk_pi0 = float((p0 * risk.masked_fill(~sup, 0)).sum())
                imp_list.append(risk_pi0 - float(risk[best].item()))   # deal-in risk reduction
                nstates += 1
    import json
    print(json.dumps({
        "counterfactual_rank_eval": {
            "n_states": nstates,
            "TV_safest_vs_pi0": round(float(np.mean(tv_list)), 4) if tv_list else None,
            "best_in_support_frac": round(float(np.mean(ins_list)), 4) if ins_list else None,
            "mean_dealin_risk_reduction_vs_pi0": round(float(np.mean(imp_list)), 5) if imp_list else None,
        }}, indent=2))
    out = ROOT / "outputs" / "critic_processed"; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out)); torch.save(vhead.state_dict(), out / "vhead.pt")
    print(f"[done] saved to {out}")


if __name__ == "__main__":
    main()

