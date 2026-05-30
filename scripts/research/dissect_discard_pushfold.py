"""Discard-focused dissection: is outcome-conditioning sensitivity concentrated
at the discards where skill actually shows — push/fold under an opponent's
riichi — rather than the trivial average discard?

For a v3 checkpoint, measures the viewer's DISCARD policy movement between
round_oc_bigwin and round_oc_bigloss conditioning, split by context:
  * under_riichi : an opponent declared riichi earlier in this round (push/fold)
  * free         : no opponent riichi yet

Also reports, on each subset, the true-vs-shuffled-bucket action NLL gap
restricted to discards. If under_riichi >> free, the flat *average* hid a real
effect at the decisions that matter.
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

from gpt2.outcome_conditioning import build_outcome_conditioned_v3, extend_tokenizer_for_v3
from gpt2.round_outcome import viewer_round_deltas
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_")
RIICHI_RE = re.compile(r"^take_self_(\d+)_riichi$")


def discard_sets(tok):
    by = defaultdict(list)
    for t, i in tok.get_vocab().items():
        m = DISCARD_RE.match(t)
        if m:
            by[int(m.group(1))].append(i)
    return {s: sorted(v) for s, v in by.items()}


def viewer_discards_with_context(cond_toks):
    """Yield (pos, seat, under_riichi) for each viewer discard."""
    out = []
    seat = None
    opp_riichi = set()
    for i, t in enumerate(cond_toks):
        if t == "round_start":
            seat = None
            opp_riichi = set()
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            seat = int(t.split("_")[1])
        m = RIICHI_RE.match(t)
        if m:
            opp_riichi.add(int(m.group(1)))
        if seat is not None:
            md = DISCARD_RE.match(t)
            if md and int(md.group(1)) == seat:
                under = any(s != seat for s in opp_riichi)
                out.append((i, seat, under))
    return out


@torch.no_grad()
def logp(model, ids, device):
    return torch.log_softmax(model(torch.tensor([ids], device=device)).logits[0].float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reinject", default="turn")
    ap.add_argument("--shard", default="2021/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=2600)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer"); extend_tokenizer_for_v3(tok)
    dsets = discard_sets(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset"))

    stats = {"under_riichi": {"tv": [], "nll_true": 0.0, "nll_shuf": 0.0, "n": 0},
             "free": {"tv": [], "nll_true": 0.0, "nll_shuf": 0.0, "n": 0}}
    games = 0
    for row in ds:
        if row["view_type"] != "imperfect" or row["length"] > args.max_len:
            continue
        sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        rinfo = viewer_round_deltas(toks, viewer_seat=viewer, seat_count=sc)
        if len(rinfo) < 2:
            continue
        true_b = [d["bucket"] for d in rinfo]
        shuf_b = true_b[1:] + true_b[:1]

        w_ids, _ = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                reinject=args.reinject, override_bucket="round_oc_bigwin")
        l_ids, _ = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                reinject=args.reinject, override_bucket="round_oc_bigloss")
        t_ids, t_mask = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                     reinject=args.reinject, round_buckets=true_b)
        s_ids, s_mask = build_outcome_conditioned_v3(ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer,
                                                     reinject=args.reinject, round_buckets=shuf_b)
        wt = tok.convert_ids_to_tokens(w_ids)
        dec = viewer_discards_with_context(wt)
        if not dec:
            continue
        lw = logp(model, w_ids, device); ll = logp(model, l_ids, device)
        lt = logp(model, t_ids, device); ls = logp(model, s_ids, device)
        # positions in t_ids/s_ids equal those in w_ids/l_ids (same structure; only bucket ids differ)
        for pos, seat, under in dec:
            key = "under_riichi" if under else "free"
            aset = dsets[seat]
            pw = torch.softmax(lw[pos - 1][aset], -1)
            pl = torch.softmax(ll[pos - 1][aset], -1)
            stats[key]["tv"].append(float(0.5 * (pw - pl).abs().sum()))
            # discard-only NLL true vs shuffled (predict the actual discard token).
            # pos is in conditioned coords; t_ids[pos]==w_ids[pos]==the discard token.
            aid = t_ids[pos]
            stats[key]["nll_true"] += -float(lt[pos - 1, aid])
            stats[key]["nll_shuf"] += -float(ls[pos - 1, aid])
            stats[key]["n"] += 1
        games += 1
        if games >= args.num_games:
            break

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    out = {"model": args.model, "n_games": games}
    for key, s in stats.items():
        n = max(1, s["n"])
        out[key] = {
            "n_discards": s["n"],
            "tv_bigwin_vs_bigloss_mean": round(mean(s["tv"]), 5),
            "tv_max": round(max(s["tv"]), 5) if s["tv"] else None,
            "frac_tv_gt_0.05": round(sum(t > 0.05 for t in s["tv"]) / n, 4),
            "discard_nll_true": round(s["nll_true"] / n, 5),
            "discard_nll_shuffled": round(s["nll_shuf"] / n, 5),
            "discard_nll_reduction_from_true": round((s["nll_shuf"] - s["nll_true"]) / n, 5),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

