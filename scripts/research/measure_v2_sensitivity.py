"""Outcome-sensitivity probe for the v2 (reinjected) conditioning scheme.

Feeds the model exactly as it was trained (viewer rank re-injected per
``--reinject``), under counterfactual viewer placements (1st vs last), and
measures how much the viewer's discard policy moves. A working OC model should
show top-vs-bot TV >> the v1 adapter's ~0.0008.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from gpt2.outcome_conditioning import build_outcome_conditioned_v2
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_")
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")


def discard_sets(tok):
    by = defaultdict(list)
    for t, i in tok.get_vocab().items():
        m = DISCARD_RE.match(t)
        if m:
            by[int(m.group(1))].append(i)
    return {s: sorted(v) for s, v in by.items()}


def set_placement(ids, toks, tok, viewer, target, seat_count):
    """Return a copy of ids with all final_rank_* replaced by a ranking where
    viewer occupies `target` place (others fill remaining places in seat order)."""
    others = [p for p in range(1, seat_count + 1) if p != target]
    place = {viewer: target}
    oi = 0
    for s in range(seat_count):
        if s == viewer:
            continue
        place[s] = others[oi]
        oi += 1
    new = list(ids)
    for j, t in enumerate(toks):
        m = FINAL_RANK_RE.match(t)
        if m:
            s = int(m.group(1))
            new[j] = tok.convert_tokens_to_ids(f"final_rank_{s}_{place[s]}")
    return new


def viewer_discards_in(cond_toks):
    out = []
    seat = None
    for i, t in enumerate(cond_toks):
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        elif seat is not None and DISCARD_RE.match(t) and int(t.split("_")[1]) == seat:
            out.append((i, seat))
    return out


@torch.no_grad()
def logits_at(model, ids, positions, device):
    x = torch.tensor([ids], device=device)
    lg = model(x).logits[0].float()
    return {p: torch.log_softmax(lg[p - 1], -1) for p in positions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reinject", default="turn", choices=("none", "round", "turn"))
    ap.add_argument("--shard", default="2024/data-00010-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=40)
    ap.add_argument("--max-decisions", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    dsets = discard_sets(tok)
    shard = hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset")
    ds = Dataset.from_file(shard)

    chi2s, tvs, n = [], [], 0
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
        top_toks = tok.convert_ids_to_tokens(top_c)
        dec = viewer_discards_in(top_toks)
        if not dec:
            continue
        if len(dec) > args.max_decisions:
            step = len(dec) / args.max_decisions
            dec = [dec[int(k * step)] for k in range(args.max_decisions)]
        # top_c and bot_c differ only in final_rank token ids, so positions match
        positions = [p for p, _ in dec]
        lt = logits_at(model, top_c, positions, device)
        lb = logits_at(model, bot_c, positions, device)
        for pos, seat in dec:
            aset = dsets[seat]
            pt = torch.softmax(lt[pos][aset], -1)
            pb = torch.softmax(lb[pos][aset], -1)
            chi2s.append(float(((pt - pb) ** 2 / pb.clamp_min(1e-12)).sum()))
            tvs.append(float(0.5 * (pt - pb).abs().sum()))
            n += 1
        games += 1
        if games >= args.num_games:
            break

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    print(json.dumps({
        "model": args.model, "reinject": args.reinject,
        "n_games": games, "n_decisions": n,
        "chi2_top_bot_mean": mean(chi2s),
        "tv_top_bot_mean": mean(tvs),
        "tv_top_bot_max": max(tvs) if tvs else float("nan"),
    }, indent=2))


if __name__ == "__main__":
    main()

