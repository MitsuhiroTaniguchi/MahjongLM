"""Build the CORRECT objective for CCC: final-placement dan-points (and the
proximal placement-equity-change), replacing the round score-delta bucket.

Only final placement matters (dan points), so Y = uma(final placement).  The
correct PROXIMAL credit-assignment is the change in placement EQUITY across a
round:  dE = E[uma | scores_after, kyoku] - E[uma | scores_before, kyoku],
where the placement-equity function is estimated EMPIRICALLY from the data
(mean final dan-points over games binned by the viewer's rank-among-seats and a
coarse score-gap state).  dE reduces exactly to uma(placement) in expectation
but is action-proximal (oorasu 2000 that flips 3rd->2nd scores high; east-1
8000 with a huge lead scores ~0).

Iterates the dataset in the SAME order/filters as extract_ccc_features_all.py
(view=imperfect, length<=max, has final_rank, iter_viewer_decisions non-empty)
so the per-game arrays align with ccc_features_all.npz's `game` index.

Saves per-GAME: place (viewer final placement 1..n), n_players, y_term
(=uma[place]); and per-DECISION (aligned to ccc_features_all rows): y_term and
y_equity (the round's placement-equity change for that decision's round).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from datasets import Dataset
from huggingface_hub import hf_hub_download

from gpt2.round_outcome import (_round_boundaries, _viewer_round_seat, _seat_score_at,
                                _final_score, _decode_tenbo_run)
from gpt2.viewer_decisions import iter_viewer_decisions
from tenhou_tokenizer.huggingface import MahjongTokenizerFast

FINAL_RANK_RE = re.compile(r"^final_rank_(\d+)_(\d+)$")

# dan-point uma tables (configurable). Monotone, roughly Tenhou-houou shaped.
UMA = {4: [90.0, 45.0, -45.0, -90.0], 3: [75.0, 0.0, -75.0]}


def viewer_placement(toks, viewer_seat):
    for t in toks:
        m = FINAL_RANK_RE.match(t)
        if m and int(m.group(1)) == viewer_seat:
            return int(m.group(2))
    return None


def all_seat_scores_at(toks, lo, hi, n):
    out = []
    for s in range(n):
        v = _seat_score_at(toks, lo, hi, s)
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=["2024/data-00000-of-00016.arrow",
                                                    "2024/data-00001-of-00016.arrow"])
    ap.add_argument("--num-games", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--features", default="outputs/research/ccc_features_all.npz")
    ap.add_argument("--out", default="outputs/research/ccc_placement_target.npz")
    args = ap.parse_args()
    tok = MahjongTokenizerFast.from_pretrained("tokenizer")

    # decision-level game ids from the features file (for alignment / per-decision expansion)
    feat = np.load(args.features)
    feat_game = feat["game"]

    # --- pass 1: per game, gather final placement + per-round (rel_seat, score_gap) -> dan pts ---
    # equity table key: (n, viewer_rank_position 0..n-1, gap_bucket) -> list of final dan pts
    games = []        # list of dicts: place, n, rounds=[(start,end, rank_pos, gap_bucket)]
    equity_samples = {}

    def gap_bucket(scores, seat):
        # viewer rank position (0=leading) and a coarse gap-to-leader/own bucket
        s = scores[seat]
        ranked = sorted(range(len(scores)), key=lambda k: -scores[k])
        pos = ranked.index(seat)
        lead = scores[ranked[0]]
        gap = s - lead if pos > 0 else s - scores[ranked[1]]  # +lead margin if leading
        gb = int(np.clip(gap // 8000, -4, 4))
        return pos, gb

    seen = 0
    for shard in args.shards:
        ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", shard, repo_type="dataset"))
        for row in ds:
            if row["view_type"] != "imperfect" or row["length"] > args.max_len:
                continue
            sc = int(row["seat_count"]); viewer = int(row["viewer_seat"])
            ids = [int(t) for t in row["input_ids"]]
            toks = tok.convert_ids_to_tokens(ids)
            if not any(FINAL_RANK_RE.match(t) for t in toks):
                continue
            dec = iter_viewer_decisions(toks, viewer)
            if not dec:
                continue
            place = viewer_placement(toks, viewer)
            uma = UMA.get(sc)
            if place is None or uma is None:
                # still must advance game index to stay aligned
                games.append(None); seen += 1
                if seen >= args.num_games:
                    break
                continue
            yterm = uma[place - 1]
            bounds = _round_boundaries(toks); bounds_ext = bounds + [len(toks)]
            rinfo = []
            for r in range(len(bounds)):
                lo, hi = bounds_ext[r], bounds_ext[r + 1]
                rseat = _viewer_round_seat(toks, lo, hi, viewer, sc)
                scores = all_seat_scores_at(toks, lo, hi, sc) if rseat is not None else None
                key = None
                if scores is not None and all(v is not None for v in scores):
                    pos, gb = gap_bucket(scores, rseat)
                    key = (sc, pos, gb)
                    equity_samples.setdefault(key, []).append(yterm)
                rinfo.append({"start": lo, "end": hi, "key": key})
            games.append({"place": place, "n": sc, "yterm": yterm, "rounds": rinfo, "viewer": viewer})
            seen += 1
            if seen >= args.num_games:
                break
        if seen >= args.num_games:
            break

    # empirical placement-equity E[uma | key]
    equity = {k: float(np.mean(v)) for k, v in equity_samples.items()}
    global_mean = float(np.mean([g["yterm"] for g in games if g])) if any(games) else 0.0

    def eq_of(key):
        return equity.get(key, global_mean) if key is not None else global_mean

    # per-game y_term array + per-game round equity series
    n_games = len(games)
    place_arr = np.full(n_games, -1, dtype=np.int64)
    nplayers_arr = np.zeros(n_games, dtype=np.int64)
    yterm_game = np.zeros(n_games, dtype=np.float32)
    # round equity-change: for each game, map round index -> dE; and per token pos -> round
    round_dE = {}     # game -> {round_idx: dE}
    round_span = {}   # game -> list of (start,end) per round
    for gi, g in enumerate(games):
        if not g:
            continue
        place_arr[gi] = g["place"]; nplayers_arr[gi] = g["n"]; yterm_game[gi] = g["yterm"]
        eqs = [eq_of(r["key"]) for r in g["rounds"]]
        # equity AFTER round r ~ equity at start of round r+1 (terminal -> yterm)
        dE = {}
        for r in range(len(g["rounds"])):
            before = eqs[r]
            after = eqs[r + 1] if r + 1 < len(eqs) else g["yterm"]
            dE[r] = after - before
        round_dE[gi] = dE
        round_span[gi] = [(r["start"], r["end"]) for r in g["rounds"]]

    # --- expand to per-DECISION arrays aligned with ccc_features_all rows ---
    # We must reproduce per-decision (game, pos) order. The features were built by
    # iter_viewer_decisions over the same games; recompute pos per game and map.
    y_term_dec = np.zeros(len(feat_game), dtype=np.float32)
    y_equity_dec = np.zeros(len(feat_game), dtype=np.float32)
    # build per-game decision pos list once more (same iteration)
    # (re-walk dataset quickly for positions; cheap, tokens only)
    seen = 0; row_ptr = 0
    for shard in args.shards:
        ds = Dataset.from_file(hf_hub_download("mitsutani/mahjonglm-dataset", shard, repo_type="dataset"))
        for row in ds:
            if row["view_type"] != "imperfect" or row["length"] > args.max_len:
                continue
            viewer = int(row["viewer_seat"])
            ids = [int(t) for t in row["input_ids"]]
            toks = tok.convert_ids_to_tokens(ids)
            if not any(FINAL_RANK_RE.match(t) for t in toks):
                continue
            dec = iter_viewer_decisions(toks, viewer)
            if not dec:
                continue
            gi = seen; seen += 1
            g = games[gi]
            spans = round_span.get(gi); dE = round_dE.get(gi)
            for (pos, dt, seat) in dec:
                if pos == 0:
                    continue
                if g is None:
                    yt = global_mean; ye = 0.0
                else:
                    yt = g["yterm"]
                    ye = 0.0
                    if spans is not None:
                        for ridx, (lo, hi) in enumerate(spans):
                            if lo <= pos < hi:
                                ye = dE.get(ridx, 0.0); break
                y_term_dec[row_ptr] = yt; y_equity_dec[row_ptr] = ye; row_ptr += 1
            if seen >= args.num_games:
                break
        if seen >= args.num_games:
            break

    assert row_ptr == len(feat_game), f"decision count mismatch {row_ptr} vs {len(feat_game)}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        y_term=y_term_dec, y_equity=y_equity_dec,
                        place_game=place_arr, nplayers_game=nplayers_arr, yterm_game=yterm_game)
    print(f"games={seen} decisions={row_ptr}; equity_keys={len(equity)}; "
          f"yterm mean={y_term_dec.mean():.2f} std={y_term_dec.std():.2f}; "
          f"y_equity std={y_equity_dec.std():.3f}")


if __name__ == "__main__":
    main()
