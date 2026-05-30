"""Measure outcome-conditioning policy movement on held-out games.

Compares, at viewer *discard* decision points in `view_imperfect_*` games:
  * pi_base  = base model next-token distribution (no conditioning prefix)
  * pi_top   = OC model conditioned on "viewer finished 1st"
  * pi_bot   = OC model conditioned on "viewer finished last"
  * pi_true  = OC model conditioned on the game's actual final ranking

It reports the diagnostics predicted by Russo (2026), arXiv:2601.18175:
  - chi^2 / KL policy movement chi^2(pi_top || pi_base)         [Prop 4.4]
  - outcome sensitivity chi^2(pi_top || pi_bot)                 [over-conservatism check, P3]
  - chi^2 geometry: mass pi_top puts on base-rare actions       [Lemma 4.6, P2]
  - directional strength: pi_top(a*) vs pi_base(a*) on the human move,
    split by whether the viewer actually finished 1st           [P4]

The only input difference between base and OC is the prepended
`<bos> final_rank_*...` block, so the comparison is clean.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from itertools import permutations

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

from tenhou_tokenizer.huggingface import MahjongTokenizerFast

DISCARD_RE = re.compile(r"^discard_(\d+)_")
FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")


def build_discard_action_sets(tok: MahjongTokenizerFast) -> dict[int, list[int]]:
    """seat -> list of token ids that are `discard_{seat}_*`."""
    by_seat: dict[int, list[int]] = defaultdict(list)
    vocab = tok.get_vocab()
    for token, tid in vocab.items():
        m = DISCARD_RE.match(token)
        if m:
            by_seat[int(m.group(1))].append(tid)
    return {s: sorted(ids) for s, ids in by_seat.items()}


def final_rank_token_id(tok: MahjongTokenizerFast, seat: int, place: int) -> int:
    return tok.convert_tokens_to_ids(f"final_rank_{seat}_{place}")


def conditioning_prefixes(
    tok: MahjongTokenizerFast, seat_count: int, viewer: int, target_place: int
) -> list[list[int]]:
    """All final-rank prefixes consistent with viewer occupying target_place."""
    others = [p for p in range(1, seat_count + 1) if p != target_place]
    seats = list(range(seat_count))
    out: list[list[int]] = []
    for perm in permutations(others):
        places = {viewer: target_place}
        oi = 0
        for s in seats:
            if s == viewer:
                continue
            places[s] = perm[oi]
            oi += 1
        out.append([final_rank_token_id(tok, s, places[s]) for s in seats])
    return out


def true_rank_prefix(toks: list[str], tok: MahjongTokenizerFast, seat_count: int) -> tuple[list[int], dict[int, int]]:
    places: dict[int, int] = {}
    for t in toks:
        m = FINAL_RANK_RE.match(t)
        if m:
            places[int(m.group(1))] = int(m.group(2))
    ids = [final_rank_token_id(tok, s, places[s]) for s in range(seat_count)]
    return ids, places


def viewer_discard_decisions(toks: list[str]) -> list[tuple[int, int]]:
    """Return (position, seat) for each viewer discard.

    The viewer seat rotates per round; the seat whose draw is *non-hidden*
    is the viewer for that round.
    """
    decisions: list[tuple[int, int]] = []
    viewer_seat: int | None = None
    for i, t in enumerate(toks):
        if t == "round_start":
            viewer_seat = None
        elif t.startswith("draw_") and not t.endswith("_hidden"):
            viewer_seat = int(t.split("_")[1])
        elif viewer_seat is not None:
            m = DISCARD_RE.match(t)
            if m and int(m.group(1)) == viewer_seat:
                decisions.append((i, viewer_seat))
    return decisions


@torch.no_grad()
def next_token_logprobs(model, input_ids: list[int], positions: list[int], device) -> dict[int, torch.Tensor]:
    """Return log-softmax logits that predict token at each `pos` (i.e. logits at pos-1)."""
    x = torch.tensor([input_ids], device=device)
    out = model(x).logits[0]  # (seq, vocab)
    logp = torch.log_softmax(out.float(), dim=-1)
    return {pos: logp[pos - 1] for pos in positions}


def restricted_dist(logp: torch.Tensor, action_ids: list[int]) -> torch.Tensor:
    sub = logp[action_ids]
    return torch.softmax(sub, dim=-1)


def chi2(p: torch.Tensor, q: torch.Tensor) -> float:
    # chi^2(p || q) = sum_a (p/q - 1)^2 q = sum_a (p-q)^2/q  [Def 4.2]
    return float(((p - q) ** 2 / q.clamp_min(1e-12)).sum())


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())


def tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mitsutani/mahjonglm-10m")
    ap.add_argument("--adapter", default="mitsutani/mahjonglm-10m-oc")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--shard", default="2024/data-00000-of-00016.arrow")
    ap.add_argument("--num-games", type=int, default=40)
    ap.add_argument("--max-decisions-per-game", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1200)
    ap.add_argument("--out", default="outputs/research/policy_movement.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MahjongTokenizerFast.from_pretrained(args.tokenizer)
    discard_sets = build_discard_action_sets(tok)

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(device).eval()
    oc = PeftModel.from_pretrained(base, args.adapter)
    oc = oc.to(device).eval()
    # NOTE: oc wraps base; oc.disable_adapter() context == base behaviour.

    shard = hf_hub_download("mitsutani/mahjonglm-dataset", args.shard, repo_type="dataset")
    ds = Dataset.from_file(shard)

    bos = tok.bos_token_id
    records = []
    games = 0
    for row in ds:
        if row["view_type"] != "imperfect":
            continue
        if row["length"] > args.max_len:
            continue
        seat_count = int(row["seat_count"])
        viewer = int(row["viewer_seat"])
        ids = [int(t) for t in row["input_ids"]]
        toks = tok.convert_ids_to_tokens(ids)
        decisions = viewer_discard_decisions(toks)
        if not decisions:
            continue
        # subsample decisions evenly
        if len(decisions) > args.max_decisions_per_game:
            step = len(decisions) / args.max_decisions_per_game
            decisions = [decisions[int(k * step)] for k in range(args.max_decisions_per_game)]
        positions = [pos for pos, _ in decisions]

        true_ids, places = true_rank_prefix(toks, tok, seat_count)
        viewer_true_place = places[viewer]

        # base: feed raw ids, read logits at decision positions
        base_logp = {}
        with oc.disable_adapter():
            base_logp = next_token_logprobs(oc, ids, positions, device)

        # OC conditioned variants. prefix = <bos> + final_rank ids ; offset shifts positions
        def oc_dist_for_prefix(rank_ids: list[int]):
            prefix = [bos] + rank_ids
            cond_ids = prefix + ids
            shift = len(prefix)
            shifted = [pos + shift for pos in positions]
            lp = next_token_logprobs(oc, cond_ids, shifted, device)
            return {pos: lp[pos + shift] for pos in positions}

        top_prefixes = conditioning_prefixes(tok, seat_count, viewer, 1)
        bot_prefixes = conditioning_prefixes(tok, seat_count, viewer, seat_count)

        # average distributions over consistent permutations (marginalize other seats)
        def avg_dist(prefixes, seat, pos):
            acc = None
            for rp in prefixes:
                d = oc_dist_for_prefix(rp)[pos]
                pd = restricted_dist(d, discard_sets[seat])
                acc = pd if acc is None else acc + pd
            return acc / len(prefixes)

        # precompute OC logp for top/bot/true once per prefix to avoid recompute per pos
        top_logp_list = [oc_dist_for_prefix(rp) for rp in top_prefixes]
        bot_logp_list = [oc_dist_for_prefix(rp) for rp in bot_prefixes]
        true_logp = oc_dist_for_prefix(true_ids)

        for pos, seat in decisions:
            aset = discard_sets[seat]
            p_base = restricted_dist(base_logp[pos], aset)
            p_top = torch.stack([restricted_dist(l[pos], aset) for l in top_logp_list]).mean(0)
            p_bot = torch.stack([restricted_dist(l[pos], aset) for l in bot_logp_list]).mean(0)
            p_true = restricted_dist(true_logp[pos], aset)
            actual_id = ids[pos]
            try:
                a_idx = aset.index(actual_id)
            except ValueError:
                continue  # actual discard not in viewer set (shouldn't happen)
            # base mass that lands on the discard set (sanity): use full logp
            base_mass = float(torch.softmax(base_logp[pos], -1)[aset].sum())
            records.append({
                "viewer_true_place": viewer_true_place,
                "seat_count": seat_count,
                "n_actions": len(aset),
                "base_mass_on_discards": base_mass,
                "chi2_top_base": chi2(p_top, p_base),
                "kl_top_base": kl(p_top, p_base),
                "tv_top_base": tv(p_top, p_base),
                "chi2_top_bot": chi2(p_top, p_bot),
                "tv_top_bot": tv(p_top, p_bot),
                "chi2_true_base": chi2(p_true, p_base),
                "p_base_actual": float(p_base[a_idx]),
                "p_top_actual": float(p_top[a_idx]),
                "p_bot_actual": float(p_bot[a_idx]),
                "p_true_actual": float(p_true[a_idx]),
                # geometry: max prob OC-top assigns to an action base deemed rare (<1%)
                "max_top_on_base_rare": float(
                    p_top[(p_base < 0.01)].max() if (p_base < 0.01).any() else torch.tensor(0.0)
                ),
            })
        games += 1
        if games >= args.num_games:
            break

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = summarize(records)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "n_records": len(records), "n_games": games}, f, indent=2)
    print(json.dumps({"n_games": games, "n_decisions": len(records), "summary": summary}, indent=2))


def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    return xs[len(xs) // 2] if xs else float("nan")


def summarize(records):
    if not records:
        return {}
    won = [r for r in records if r["viewer_true_place"] == 1]
    lost = [r for r in records if r["viewer_true_place"] == max(r["seat_count"] for r in records)]
    def block(rs):
        return {
            "n": len(rs),
            "chi2_top_base_mean": mean([r["chi2_top_base"] for r in rs]),
            "chi2_top_base_median": median([r["chi2_top_base"] for r in rs]),
            "kl_top_base_mean": mean([r["kl_top_base"] for r in rs]),
            "tv_top_base_mean": mean([r["tv_top_base"] for r in rs]),
            "chi2_top_bot_mean": mean([r["chi2_top_bot"] for r in rs]),
            "tv_top_bot_mean": mean([r["tv_top_bot"] for r in rs]),
            "p_base_actual_mean": mean([r["p_base_actual"] for r in rs]),
            "p_top_actual_mean": mean([r["p_top_actual"] for r in rs]),
            "p_true_actual_mean": mean([r["p_true_actual"] for r in rs]),
            "base_mass_on_discards_mean": mean([r["base_mass_on_discards"] for r in rs]),
            "max_top_on_base_rare_mean": mean([r["max_top_on_base_rare"] for r in rs]),
        }
    return {
        "all": block(records),
        "viewer_won_1st": block(won),
        "viewer_finished_last": block(lost),
    }


if __name__ == "__main__":
    main()
