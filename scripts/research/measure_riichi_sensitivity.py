"""Outcome sensitivity at the viewer's riichi take/pass decision.

Riichi is a high-stakes, high-action-influence decision: a good test of
whether the OC adapter's outcome-insensitivity at discards is genuine low
action-influence or undertraining. We compare P(take riichi) under base,
OC(top), and OC(bottom).
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import permutations

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")


def viewer_seat_track(toks):
    seat = None
    out = []
    for t in toks:
        if t == "round_start":
            seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        out.append(seat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=1200)
    ap.add_argument("--shard", default="2024/data-00000-of-00016.arrow")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    base = AutoModelForCausalLM.from_pretrained("mitsutani/mahjonglm-10m", dtype=torch.float32).to(device).eval()
    oc = PeftModel.from_pretrained(base, "mitsutani/mahjonglm-10m-oc").to(device).eval()
    bos = tok.bos_token_id

    def rid(seat, place):
        return tok.convert_tokens_to_ids(f"final_rank_{seat}_{place}")

    def prefixes(seat_count, viewer, place):
        others = [p for p in range(1, seat_count + 1) if p != place]
        res = []
        for perm in permutations(others):
            places = {viewer: place}
            oi = 0
            for s in range(seat_count):
                if s == viewer:
                    continue
                places[s] = perm[oi]; oi += 1
            res.append([rid(s, places[s]) for s in range(seat_count)])
        return res

    @torch.no_grad()
    def logp_at(model, ids, pos):
        x = torch.tensor([ids], device=device)
        return torch.log_softmax(model(x).logits[0, pos - 1].float(), -1)

    shard = hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset")
    ds = Dataset.from_file(shard)

    rows = []
    games = 0
    for row in ds:
        if row["view_type"] != "imperfect" or row["length"] > args.max_len:
            continue
        seat_count = int(row["seat_count"]); viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        seat_at = viewer_seat_track(toks)
        places = {int(m.group(1)): int(m.group(2)) for m in (FINAL_RANK_RE.match(t) for t in toks) if m}
        viewer_place = places.get(viewer)
        # find viewer riichi take/pass decisions
        decisions = []
        for i, t in enumerate(toks):
            s = seat_at[i]
            if s is None:
                continue
            if t in (f"take_self_{s}_riichi", f"pass_self_{s}_riichi"):
                take_id = tok.convert_tokens_to_ids(f"take_self_{s}_riichi")
                pass_id = tok.convert_tokens_to_ids(f"pass_self_{s}_riichi")
                decisions.append((i, take_id, pass_id, t.startswith("take_")))
        if not decisions:
            continue

        def p_take(prefix_ids, pos, take_id, pass_id):
            if prefix_ids is None:
                lp = logp_at(base, ids, pos)
            else:
                full = [bos] + prefix_ids + ids
                lp = logp_at(oc, full, pos + 1 + len(prefix_ids))
            two = torch.softmax(torch.stack([lp[take_id], lp[pass_id]]), 0)
            return float(two[0])

        top_pref = prefixes(seat_count, viewer, 1)
        bot_pref = prefixes(seat_count, viewer, seat_count)
        for pos, take_id, pass_id, took in decisions:
            pb = p_take(None, pos, take_id, pass_id)
            pt = sum(p_take(rp, pos, take_id, pass_id) for rp in top_pref) / len(top_pref)
            pl = sum(p_take(rp, pos, take_id, pass_id) for rp in bot_pref) / len(bot_pref)
            rows.append({"viewer_place": viewer_place, "took_riichi": took,
                         "p_take_base": pb, "p_take_top": pt, "p_take_bot": pl})
        games += 1
        if games >= args.num_games:
            break

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    n = len(rows)
    summary = {
        "n_decisions": n,
        "n_games": games,
        "p_take_base_mean": mean([r["p_take_base"] for r in rows]),
        "p_take_top_mean": mean([r["p_take_top"] for r in rows]),
        "p_take_bot_mean": mean([r["p_take_bot"] for r in rows]),
        "mean_abs_top_minus_bot": mean([abs(r["p_take_top"] - r["p_take_bot"]) for r in rows]),
        "mean_top_minus_bot": mean([r["p_take_top"] - r["p_take_bot"] for r in rows]),
        "mean_abs_top_minus_base": mean([abs(r["p_take_top"] - r["p_take_base"]) for r in rows]),
        "max_abs_top_minus_bot": max([abs(r["p_take_top"] - r["p_take_bot"]) for r in rows], default=float("nan")),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

