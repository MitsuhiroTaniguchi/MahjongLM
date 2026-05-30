"""Dissect a trained outcome-conditioned v2 checkpoint.

Held-out analysis (default year 2021, NOT in the 2022-2024 training set) of how
the viewer's discard policy responds to the conditioned final outcome.

Reports:
  * outcome sensitivity  : chi2/TV( pi(top) || pi(bot) ) on discards
  * vs base              : chi2/TV( pi(top) || pi_base )
  * chi2 geometry        : mass pi(top) puts on base-rare actions (Lemma 4.6)
  * directional strength : on the human move a*, p_top(a*)-p_bot(a*),
                           split by the viewer's true final placement
  * phase structure      : sensitivity by game progress (early/mid/late)
  * qualitative examples : highest-sensitivity decisions, top-5 tiles top vs bot
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


def discards_in(cond_toks):
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
def logp_at(model, ids, positions, device):
    x = torch.tensor([ids], device=device)
    lg = model(x).logits[0].float()
    return {p: torch.log_softmax(lg[p - 1], -1) for p in positions}


def chi2(p, q):
    return float(((p - q) ** 2 / q.clamp_min(1e-12)).sum())


def tv(p, q):
    return float(0.5 * (p - q).abs().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--reinject", default="turn")
    ap.add_argument("--shard", default="2021/data-00000-of-00013.arrow")
    ap.add_argument("--num-games", type=int, default=80)
    ap.add_argument("--max-decisions", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--out", default="outputs/research/dissect_oc_v2.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")
    dsets = discard_sets(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(device).eval()
    shard = hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset")
    ds = Dataset.from_file(shard)
    id2tok = {i: t for t, i in tok.get_vocab().items()}

    recs = []
    examples = []
    games = 0
    for row in ds:
        if row["view_type"] != "imperfect" or row["length"] > args.max_len:
            continue
        sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        places = {int(m.group(1)): int(m.group(2)) for m in (FINAL_RANK_RE.match(t) for t in toks) if m}
        if viewer not in places:
            continue
        vplace = places[viewer]

        top_ids = set_placement(ids, toks, tok, viewer, 1, sc)
        bot_ids = set_placement(ids, toks, tok, viewer, sc, sc)
        top_c, _ = build_outcome_conditioned_v2(top_ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer, reinject=args.reinject)
        bot_c, _ = build_outcome_conditioned_v2(bot_ids, tokenizer=tok, seat_count=sc, viewer_seat=viewer, reinject=args.reinject)
        top_toks = tok.convert_ids_to_tokens(top_c)

        cond_dec = discards_in(top_toks)        # (pos, seat) in conditioned seq
        plain_dec = discards_in(toks)           # (pos, seat) in plain seq
        if not cond_dec or len(cond_dec) != len(plain_dec):
            continue
        ndec = len(cond_dec)
        idxs = list(range(ndec))
        if ndec > args.max_decisions:
            step = ndec / args.max_decisions
            idxs = [int(k * step) for k in range(args.max_decisions)]

        cpos = [cond_dec[k][0] for k in idxs]
        ppos = [plain_dec[k][0] for k in idxs]
        lt = logp_at(model, top_c, cpos, device)
        lb = logp_at(model, bot_c, cpos, device)
        lpb = logp_at(base, ids, ppos, device)

        for k in idxs:
            cp, seat = cond_dec[k]
            pp = plain_dec[k][0]
            aset = dsets[seat]
            pt = torch.softmax(lt[cp][aset], -1)
            pb = torch.softmax(lb[cp][aset], -1)
            pbase = torch.softmax(lpb[pp][aset], -1)
            actual_id = ids[pp]
            try:
                ai = aset.index(actual_id)
            except ValueError:
                continue
            rec = {
                "vplace": vplace, "sc": sc, "frac": k / max(1, ndec - 1),
                "tv_top_bot": tv(pt, pb), "chi2_top_bot": chi2(pt, pb),
                "tv_top_base": tv(pt, pbase),
                "p_top_actual": float(pt[ai]), "p_bot_actual": float(pb[ai]),
                "p_base_actual": float(pbase[ai]),
                "max_top_on_base_rare": float(pt[(pbase < 0.01)].max()) if (pbase < 0.01).any() else 0.0,
            }
            recs.append(rec)
            if rec["tv_top_bot"] > 0.15 and len(examples) < 12:
                def top5(p):
                    v, ix = torch.topk(p, min(5, len(p)))
                    return [(id2tok[aset[j]].replace("discard_", "d").replace("_tedashi", "/t").replace("_tsumogiri", "/g"), round(float(x), 3)) for x, j in zip(v, ix)]
                examples.append({
                    "vplace": vplace, "tv": round(rec["tv_top_bot"], 3),
                    "actual": id2tok[actual_id],
                    "top": top5(pt), "bot": top5(pb),
                })
        games += 1
        if games >= args.num_games:
            break

    summary = summarize(recs)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "examples": examples, "n_records": len(recs), "n_games": games},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({"n_games": games, "n_decisions": len(recs), "summary": summary}, indent=2, ensure_ascii=False))
    print("\n=== qualitative examples (TV(top,bot)>0.15) ===")
    for e in examples[:8]:
        print(f"\nviewer_final_place={e['vplace']}  TV={e['tv']}  actual={e['actual']}")
        print("  TOP :", e["top"])
        print("  BOT :", e["bot"])


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(recs):
    if not recs:
        return {}
    last = max(r["sc"] for r in recs)
    won = [r for r in recs if r["vplace"] == 1]
    lost = [r for r in recs if r["vplace"] == r["sc"]]
    def block(rs):
        return {
            "n": len(rs),
            "tv_top_bot_mean": round(mean([r["tv_top_bot"] for r in rs]), 4),
            "chi2_top_bot_mean": round(mean([r["chi2_top_bot"] for r in rs]), 4),
            "tv_top_base_mean": round(mean([r["tv_top_base"] for r in rs]), 4),
            "max_top_on_base_rare_mean": round(mean([r["max_top_on_base_rare"] for r in rs]), 4),
            "p_top_actual_mean": round(mean([r["p_top_actual"] for r in rs]), 4),
            "p_bot_actual_mean": round(mean([r["p_bot_actual"] for r in rs]), 4),
            "p_base_actual_mean": round(mean([r["p_base_actual"] for r in rs]), 4),
        }
    phases = {}
    for name, lo, hi in [("early", 0.0, 0.33), ("mid", 0.33, 0.66), ("late", 0.66, 1.01)]:
        rs = [r for r in recs if lo <= r["frac"] < hi]
        phases[name] = round(mean([r["tv_top_bot"] for r in rs]), 4)
    return {
        "all": block(recs),
        "viewer_won_1st": block(won),
        "viewer_finished_last": block(lost),
        "tv_top_bot_by_phase": phases,
        # directional: does TOP raise the winner's actual move and lower the loser's?
        "directional_winner_top_minus_bot_on_actual": round(
            mean([r["p_top_actual"] - r["p_bot_actual"] for r in won]), 4),
        "directional_loser_top_minus_bot_on_actual": round(
            mean([r["p_top_actual"] - r["p_bot_actual"] for r in lost]), 4),
    }


if __name__ == "__main__":
    main()

