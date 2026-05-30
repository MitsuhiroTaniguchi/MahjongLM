"""Strengthen the advantage critic to the limit — DOMAIN-GENERAL (no mahjong-
specific outcome labels).

Fine-tune the base LM (LoRA) with two heads at each agent decision position
(state s = hidden at pos-1, before the action):
  * Q(s, .)  per-action value head over the candidate actions -> regress the
             general RETURN (the environment's score change for the round).
  * V(s)     state-only value head -> regress the same return.
Only the taken action a+ is supervised. The decisive, general question:
    does knowing the ACTION improve held-out return prediction beyond the state?
    i.e. R2_Q (state+action) - R2_V (state only)  >  0 ?
That gain IS the detectable action-advantage. If a maximally strengthened
critic still gives gain ~ 0, the (general) action-advantage is genuinely
absent in this data (not a capacity problem). The ONLY signal is the general
return (score); no deal-in / ron / hand features anywhere.
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

DISCARD_RE = re.compile(r"^discard_(\d+)_(.+)$")          # locate the agent's action positions
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")
IMMEDIATE_WINDOW = 30                                     # tokens; "immediate" attributable window
REWARD_SCALE = 1000.0


def canonical(tok):
    sufs = sorted({DISCARD_RE.match(t).group(2) for t in tok.get_vocab() if DISCARD_RE.match(t)})
    canon = {s: i for i, s in enumerate(sufs)}
    id2c = {tid: canon[DISCARD_RE.match(t).group(2)] for t, tid in tok.get_vocab().items() if DISCARD_RE.match(t)}
    return id2c, len(canon)


def game_decisions(toks, viewer, sc):
    """(pos, action_token, immediate_reward) for the agent's action positions.

    PRINCIPLE (domain-general): the critic target is the IMMEDIATE, ACTION-
    ATTRIBUTABLE reward = the agent's signed score change at the round-
    resolution that occurs WITHIN a short window after the action (else 0).
    Domain instance: if the discard is ron'd / the round resolves right after,
    the viewer's score_delta (deal-in magnitude, tsumo-payment, ryuukyoku) is
    the immediate reward; otherwise 0. Multi-dimensional, attributable, no
    hand-crafted mahjong features beyond reading the score change."""
    out, seat = [], None
    for i, t in enumerate(toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        elif seat is not None and DISCARD_RE.match(t) and int(t.split("_")[1]) == seat:
            r = 0.0
            marker = f"score_delta_{seat}"
            for j in range(i + 1, min(i + IMMEDIATE_WINDOW, len(toks))):
                if toks[j].startswith("draw_") and not toks[j].endswith("_hidden"):
                    break  # viewer's next turn -> beyond the immediate window
                if toks[j] == marker:
                    val, _ = _decode_tenbo_run(toks, j + 1)
                    r += val / REWARD_SCALE
            out.append((i, t, r))
    return out


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
    args = ap.parse_args()
    device = "cuda"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    id2c, n_act = canonical(tok)

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg).to(device)
    d = model.config.hidden_size
    qhead = nn.Linear(d, n_act).to(device, torch.float32)   # per-action value Q(s,.)
    vhead = nn.Linear(d, 1).to(device, torch.float32)       # state value V(s)
    for h in (qhead, vhead):
        nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

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
    print(f"games train={len(tr_idx)} test={len(te_idx)} n_act={n_act} d={d}")

    params = [p for p in model.parameters() if p.requires_grad] + list(qhead.parameters()) + list(vhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def prep(row):
        toks = tok.convert_ids_to_tokens([int(t) for t in row["input_ids"]])
        if not any(FINAL_RANK_RE.match(t) for t in toks):
            return None, None
        dec = [(pos, id2c[tok.convert_tokens_to_ids(t)], rv)
               for (pos, t, rv) in game_decisions(toks, int(row["viewer_seat"]), int(row["seat_count"]))
               if tok.convert_tokens_to_ids(t) in id2c]
        return [int(t) for t in row["input_ids"]], dec

    def run(train):
        model.train(train)
        order = tr_idx if train else te_idx
        seQ = seV = n = 0.0; yacc = []; qacc = []
        step = 0; opt.zero_grad()
        for ri in order:
            ids, dec = prep(rows[ri])
            if not dec:
                continue
            x = torch.tensor([ids], device=device)
            with torch.set_grad_enabled(train):
                hs = model(x, output_hidden_states=True).hidden_states[-1][0]
                pos = torch.tensor([p - 1 for (p, _, _) in dec], device=device)
                a = torch.tensor([c for (_, c, _) in dec], device=device)
                rv = torch.tensor([v for (_, _, v) in dec], device=device, dtype=torch.float32)
                h = hs[pos].float()
                qr = qhead(h).gather(1, a.view(-1, 1)).squeeze(1)
                vr = vhead(h).squeeze(-1)
                loss = F.mse_loss(qr, rv) + F.mse_loss(vr, rv)
                if train:
                    (loss / args.accum).backward(); step += 1
                    if step % args.accum == 0:
                        opt.step(); opt.zero_grad()
                else:
                    seQ += float(((qr - rv) ** 2).sum()); seV += float(((vr - rv) ** 2).sum())
                    n += len(rv); yacc.append(rv.cpu().numpy()); qacc.append(qr.detach().cpu().numpy())
        if not train:
            y = np.concatenate(yacc); q = np.concatenate(qacc)
            var = float(((y - y.mean()) ** 2).mean())
            # bad-event (immediate negative reward, e.g. deal-in) detection AUC: lower Q = worse
            lab = (y < -1e-6).astype(int); npos = int(lab.sum()); nneg = int((lab == 0).sum())
            o = np.argsort(-q); rk = np.empty(len(q)); rk[o] = np.arange(1, len(q) + 1)
            au = (rk[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
            return 1 - (seQ / n) / var, 1 - (seV / n) / var, float(au), float(lab.mean())
        return None

    for ep in range(args.epochs):
        run(True)
        r2q, r2v, au, frac = run(False)
        print(f"[epoch {ep+1}] R2_Q(state+action)={r2q:.4f}  R2_V(state)={r2v:.4f}  "
              f"action_advantage_gain={r2q - r2v:.4f}  bad_event_AUC={au:.4f}  nonzero_reward_frac={frac:.4f}")
    out = ROOT / "outputs" / "critic_lora_general"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    torch.save({"qhead": qhead.state_dict(), "vhead": vhead.state_dict(), "n_act": n_act}, out / "heads.pt")
    print(f"[done] saved general critic to {out}")


if __name__ == "__main__":
    main()

