"""Riichi take/pass outcome sensitivity for the trained v2 checkpoint.

Feeds the v2 model with reinjected conditioning under counterfactual viewer
placement (1st vs last) and measures P(take riichi). Riichi is the highest
action-influence decision; if it is also flat, terminal-rank conditioning is
genuinely uninformative per-decision.
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.outcome_conditioning import build_outcome_conditioned_v2
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")


def set_placement(ids, toks, tok, viewer, target, sc):
    others = [p for p in range(1, sc + 1) if p != target]
    place = {viewer: target}
    oi = 0
    for s in range(sc):
        if s == viewer:
            continue
        place[s] = others[oi]; oi += 1
    new = list(ids)
    for j, t in enumerate(toks):
        m = FINAL_RANK_RE.match(t)
        if m:
            new[j] = tok.convert_tokens_to_ids(f"final_rank_{int(m.group(1))}_{place[int(m.group(1))]}")
    return new


def viewer_seat_track(toks):
    seat = None; out = []
    for t in toks:
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        out.append(seat)
    return out


@torch.no_grad()
def lp(model, ids, pos, device):
    x = torch.tensor([ids], device=device)
    return torch.log_softmax(model(x).logits[0, pos - 1].float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reinject", default="turn")
    ap.add_argument("--shard", default="2021/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=2600)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    shard = hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset")
    ds = Dataset.from_file(shard)

    rows = []
    games = 0
    for row in ds:
        if row["view_type"] != "imperfect" or row["length"] > args.max_len:
            continue
        sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        top_ids = set_placement(ids, toks, tok, viewer, 1, sc)
        bot_ids = set_placement(ids, toks, tok, viewer, sc, sc)
        top_c, _ = build_outcome_conditioned_v2(top_ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer, reinject=args.reinject)
        bot_c, _ = build_outcome_conditioned_v2(bot_ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer, reinject=args.reinject)
        ctoks = tok.convert_ids_to_tokens(top_c)
        seat_at = viewer_seat_track(ctoks)
        for i, t in enumerate(ctoks):
            s = seat_at[i]
            if s is None:
                continue
            if t in (f"take_self_{s}_riichi", f"pass_self_{s}_riichi"):
                take = tok.convert_tokens_to_ids(f"take_self_{s}_riichi")
                pas = tok.convert_tokens_to_ids(f"pass_self_{s}_riichi")
                lt = lp(model, top_c, i, device); lb = lp(model, bot_c, i, device)
                pt = float(torch.softmax(torch.stack([lt[take], lt[pas]]), 0)[0])
                pb = float(torch.softmax(torch.stack([lb[take], lb[pas]]), 0)[0])
                rows.append((pt, pb))
        games += 1
        if games >= args.num_games:
            break

    import statistics as st
    pts = [r[0] for r in rows]; pbs = [r[1] for r in rows]
    diffs = [abs(a - b) for a, b in rows]
    print(json.dumps({
        "n_riichi_decisions": len(rows), "n_games": games,
        "p_take_top_mean": round(st.mean(pts), 4) if pts else None,
        "p_take_bot_mean": round(st.mean(pbs), 4) if pbs else None,
        "mean_abs_diff": round(st.mean(diffs), 4) if diffs else None,
        "max_abs_diff": round(max(diffs), 4) if diffs else None,
        "frac_diff_gt_0.05": round(sum(d > 0.05 for d in diffs) / len(diffs), 3) if diffs else None,
    }, indent=2))


if __name__ == "__main__":
    main()

