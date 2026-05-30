"""Per-round (kyoku) outcome extraction for v3 *proximal* outcome conditioning.

The v2 dissection showed terminal final-rank conditioning carries ~0.003 nats
about the viewer's per-move actions: the rank is too distal / luck-dominated.
v3 instead conditions each round's decisions on **that round's realized score
delta for the viewer** — a proximal outcome tightly coupled to the round's
play, so its action-influence should be far higher.

This module derives, from a tokenized `view_imperfect_{viewer}` sequence, the
viewer's signed score delta for every round and maps it to one of five bucket
tokens. No raw game JSON is needed — everything is read back from the tokens.
"""
from __future__ import annotations

import re
from typing import Sequence

_KYOKU_RE = re.compile(r"^kyoku_(\d+)$")

TENBO_UNIT_VALUES = {
    "TENBO_10000": 10000, "TENBO_9000": 9000, "TENBO_8000": 8000, "TENBO_7000": 7000,
    "TENBO_6000": 6000, "TENBO_5000": 5000, "TENBO_4000": 4000, "TENBO_3000": 3000,
    "TENBO_2000": 2000, "TENBO_1000": 1000, "TENBO_900": 900, "TENBO_800": 800,
    "TENBO_700": 700, "TENBO_600": 600, "TENBO_500": 500, "TENBO_400": 400,
    "TENBO_300": 300, "TENBO_200": 200, "TENBO_100": 100,
}
_TENBO_ANY = set(TENBO_UNIT_VALUES) | {"TENBO_PLUS", "TENBO_MINUS", "TENBO_ZERO"}

# Round-outcome bucket tokens (added to the tokenizer for v3).
ROUND_OC_BIGWIN = "round_oc_bigwin"
ROUND_OC_WIN = "round_oc_win"
ROUND_OC_ZERO = "round_oc_zero"
ROUND_OC_LOSS = "round_oc_loss"
ROUND_OC_BIGLOSS = "round_oc_bigloss"
ROUND_OC_TOKENS = [ROUND_OC_BIGWIN, ROUND_OC_WIN, ROUND_OC_ZERO, ROUND_OC_LOSS, ROUND_OC_BIGLOSS]


def bucket_for_delta(delta: int) -> str:
    if delta >= 8000:
        return ROUND_OC_BIGWIN
    if delta >= 1000:
        return ROUND_OC_WIN
    if delta > -1000:
        return ROUND_OC_ZERO
    if delta > -8000:
        return ROUND_OC_LOSS
    return ROUND_OC_BIGLOSS


def _decode_tenbo_run(tokens: Sequence[str], start: int) -> tuple[int, int]:
    """Decode the TENBO run beginning at index `start`. Returns (value, next_idx)."""
    i = start
    if i >= len(tokens) or tokens[i] not in _TENBO_ANY:
        return 0, start
    if tokens[i] == "TENBO_ZERO":
        return 0, i + 1
    sign = 1 if tokens[i] == "TENBO_PLUS" else -1
    i += 1
    total = 0
    while i < len(tokens) and tokens[i] in TENBO_UNIT_VALUES:
        total += TENBO_UNIT_VALUES[tokens[i]]
        i += 1
    return sign * total, i


def _round_boundaries(tokens: Sequence[str]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t == "round_start"]


def _viewer_round_seat(tokens: Sequence[str], lo: int, hi: int, viewer_seat: int, seat_count: int) -> int | None:
    """Viewer's round-relative seat = (viewer_seat - kyoku) % seat_count.

    Derived from the round's ``kyoku_{k}`` token; validated against the
    non-hidden draw seat across 2811 rounds (0 mismatches). Using the kyoku
    token (rather than the draw) means we also get the seat for rounds where
    the viewer never draws (e.g. an early deal-in), which is essential for an
    unbroken per-round score chain.
    """
    for i in range(lo, hi):
        m = _KYOKU_RE.match(tokens[i])
        if m:
            return (viewer_seat - (int(m.group(1)) % seat_count)) % seat_count
    # Fallback: the non-hidden draw seat.
    for i in range(lo, hi):
        if tokens[i].startswith("draw_") and not tokens[i].endswith("_hidden"):
            return int(tokens[i].split("_")[1])
    return None


def _seat_score_at(tokens: Sequence[str], lo: int, hi: int, seat: int) -> int | None:
    marker = f"score_{seat}"
    for i in range(lo, hi):
        if tokens[i] == marker:
            val, _ = _decode_tenbo_run(tokens, i + 1)
            return val
    return None


def _final_score(tokens: Sequence[str], viewer_seat: int) -> int | None:
    marker = f"final_score_{viewer_seat}"
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == marker:
            val, _ = _decode_tenbo_run(tokens, i + 1)
            return val
    return None


def viewer_round_deltas(tokens: Sequence[str], *, viewer_seat: int, seat_count: int) -> list[dict]:
    """Return one dict per round: {start, end, seat, delta, bucket}.

    `start`/`end` are token indices bounding the round (end exclusive). `seat`
    is the viewer's round-relative seat that round. `delta` is the viewer's
    signed score change during the round; `bucket` the corresponding token.
    """
    bounds = _round_boundaries(tokens)
    if not bounds:
        return []
    bounds_ext = bounds + [len(tokens)]
    # viewer score at the start of each round
    rounds = []
    for r in range(len(bounds)):
        lo, hi = bounds_ext[r], bounds_ext[r + 1]
        seat = _viewer_round_seat(tokens, lo, hi, viewer_seat, seat_count)
        start_score = _seat_score_at(tokens, lo, hi, seat) if seat is not None else None
        rounds.append({"start": lo, "end": hi, "seat": seat, "start_score": start_score})
    final_score = _final_score(tokens, viewer_seat)
    out = []
    for r in range(len(rounds)):
        cur = rounds[r]
        if r + 1 < len(rounds):
            nxt_score = rounds[r + 1]["start_score"]
        else:
            nxt_score = final_score
        if cur["start_score"] is None or nxt_score is None:
            delta = 0
        else:
            delta = nxt_score - cur["start_score"]
        out.append({
            "start": cur["start"], "end": cur["end"], "seat": cur["seat"],
            "delta": delta, "bucket": bucket_for_delta(delta),
        })
    return out

