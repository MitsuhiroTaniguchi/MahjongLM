"""Enumerate ALL viewer decision points and their candidate action sets.

This mirrors :func:`gpt2.outcome_conditioning.is_viewer_action_token` EXACTLY
(same regexes, same stateful transitions) but, instead of a per-token bool,
yields one record per viewer DECISION with its type and seat, and provides the
per-type candidate token-id sets needed for counterfactual ranking.

Decision types (DTYPE):
  discard   : discard_{seat}_{tile}                 -> candidates = seat's discards
  self      : take_self_{seat}_{opt} | pass_self_{seat}_*
                                                     -> candidates = seat's self take/pass
  react     : take_react_{seat}_{opt} | pass_react_{seat}_*_voluntary
                                                     -> candidates = seat's react take / voluntary-pass
  chi_pos   : chi_pos_{high,mid,low}                 -> candidates = the 3 chi positions
  red       : red_{used,not_used}                    -> candidates = the 2 red options
  kan_tile  : tile token right after a viewer ankan/kakan/penuki
                                                     -> candidates = all 34 tiles

The candidate set for a decision is exactly the in-support subset (pi0 >= thresh)
of its type's token group at that position; that is the legal/offered action set
as the trained policy sees it (no game-rule re-derivation needed).
"""
from __future__ import annotations

import re

DTYPE_NAMES = ["discard", "self", "react", "chi_pos", "red", "kan_tile"]
DTYPE = {n: i for i, n in enumerate(DTYPE_NAMES)}

_TILE_RE = re.compile(r"^(?:[mps][0-9]|z[1-7])$")
_DISCARD_RE = re.compile(r"^discard_(\d+)_")
_TAKE_SELF_RE = re.compile(r"^take_self_(\d+)_(\w+)$")
_PASS_SELF_RE = re.compile(r"^pass_self_(\d+)_")
_TAKE_REACT_RE = re.compile(r"^take_react_(\d+)_(\w+)$")
_PASS_REACT_RE = re.compile(r"^pass_react_(\d+)_\w+_(voluntary|forced_priority)$")
_DETAIL_TOKENS = {"chi_pos_high", "chi_pos_mid", "chi_pos_low", "red_used", "red_not_used"}
_SELF_TILE_OPTS = {"ankan", "kakan", "penuki"}


def iter_viewer_decisions(tokens, viewer_seat):
    """Yield (pos, dtype_code, seat) for every viewer decision in ``tokens``.

    Parity guarantee: the set of yielded ``pos`` equals the set of positions
    where ``is_viewer_action_token`` returns True for the same ``tokens``.
    """
    state = {}
    out = []
    for pos, token in enumerate(tokens):
        if token == "round_start":
            state["viewer_seat_round"] = None
            state["last_take_viewer"] = False
            state["expect_self_tile"] = False
            continue
        if isinstance(token, str) and token.startswith("draw_") and not token.endswith("_hidden"):
            state["viewer_seat_round"] = int(token.split("_")[1])
            state["expect_self_tile"] = False
            continue

        vs = state.get("viewer_seat_round")

        if state.get("expect_self_tile"):
            state["expect_self_tile"] = False
            if _TILE_RE.match(token):
                out.append((pos, DTYPE["kan_tile"], vs if vs is not None else -1))
                continue

        if token in _DETAIL_TOKENS:
            if state.get("last_take_viewer"):
                dt = DTYPE["chi_pos"] if token.startswith("chi_pos") else DTYPE["red"]
                out.append((pos, dt, vs if vs is not None else -1))
            continue

        if vs is None:
            continue

        m = _DISCARD_RE.match(token)
        if m:
            if int(m.group(1)) == vs:
                out.append((pos, DTYPE["discard"], vs))
            continue

        m = _TAKE_SELF_RE.match(token)
        if m:
            is_viewer = int(m.group(1)) == vs
            state["last_take_viewer"] = is_viewer
            state["expect_self_tile"] = is_viewer and m.group(2) in _SELF_TILE_OPTS
            if is_viewer:
                out.append((pos, DTYPE["self"], vs))
            continue

        m = _PASS_SELF_RE.match(token)
        if m:
            if int(m.group(1)) == vs:
                out.append((pos, DTYPE["self"], vs))
            continue

        m = _TAKE_REACT_RE.match(token)
        if m:
            is_viewer = int(m.group(1)) == vs
            state["last_take_viewer"] = is_viewer
            if is_viewer:
                out.append((pos, DTYPE["react"], vs))
            continue

        m = _PASS_REACT_RE.match(token)
        if m:
            if int(m.group(1)) == vs and m.group(2) == "voluntary":
                out.append((pos, DTYPE["react"], vs))
            continue
    return out


def build_candidate_groups(tokenizer):
    """Return per-type candidate token-id lists.

    Returns dict:
      'discard'  : {seat: [token_id, ...]}
      'self'     : {seat: [token_id, ...]}   (take_self_* + pass_self_*)
      'react'    : {seat: [token_id, ...]}   (take_react_* + pass_react_*_voluntary)
      'chi_pos'  : [token_id, ...]           (the 3)
      'red'      : [token_id, ...]           (the 2)
      'kan_tile' : [token_id, ...]           (all 34 tiles)
    """
    vocab = tokenizer.get_vocab()
    discard = {}
    selfg = {}
    react = {}
    chi_pos = []
    red = []
    kan_tile = []
    for tok, tid in vocab.items():
        if _DISCARD_RE.match(tok):
            s = int(tok.split("_")[1]); discard.setdefault(s, []).append(tid)
        elif _TAKE_SELF_RE.match(tok) or _PASS_SELF_RE.match(tok):
            s = int(tok.split("_")[2]); selfg.setdefault(s, []).append(tid)
        elif _TAKE_REACT_RE.match(tok):
            s = int(tok.split("_")[2]); react.setdefault(s, []).append(tid)
        elif _PASS_REACT_RE.match(tok):
            m = _PASS_REACT_RE.match(tok)
            if m.group(2) == "voluntary":
                s = int(tok.split("_")[2]); react.setdefault(s, []).append(tid)
        elif tok in ("chi_pos_high", "chi_pos_mid", "chi_pos_low"):
            chi_pos.append(tid)
        elif tok in ("red_used", "red_not_used"):
            red.append(tid)
        elif _TILE_RE.match(tok):
            kan_tile.append(tid)
    srt = lambda d: {k: sorted(v) for k, v in d.items()}
    return {"discard": srt(discard), "self": srt(selfg), "react": srt(react),
            "chi_pos": sorted(chi_pos), "red": sorted(red), "kan_tile": sorted(kan_tile)}


def candidates_for(groups, dtype_code, seat):
    name = DTYPE_NAMES[dtype_code]
    g = groups[name]
    if isinstance(g, dict):
        return g.get(seat, [])
    return g
