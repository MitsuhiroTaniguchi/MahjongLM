from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from transformers import PreTrainedTokenizerBase


FINAL_RANK_RE = re.compile(r"^final_rank_(?P<seat>[0-3])_(?P<place>[1-4])$")


@dataclass(frozen=True)
class FinalRankToken:
    token_id: int
    token: str
    seat: int
    place: int


def extract_final_rank_token_ids(
    input_ids: Sequence[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    seat_count: int,
) -> list[int]:
    ranks = extract_final_rank_tokens(input_ids, tokenizer=tokenizer, seat_count=seat_count)
    return [rank.token_id for rank in ranks]


def extract_final_rank_tokens(
    input_ids: Sequence[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    seat_count: int,
) -> list[FinalRankToken]:
    if seat_count not in {3, 4}:
        raise ValueError(f"seat_count must be 3 or 4, got {seat_count}")

    ranks: list[FinalRankToken] = []
    for token_id in input_ids:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if not isinstance(token, str):
            continue
        match = FINAL_RANK_RE.match(token)
        if match is None:
            continue
        ranks.append(
            FinalRankToken(
                token_id=int(token_id),
                token=token,
                seat=int(match.group("seat")),
                place=int(match.group("place")),
            )
        )

    if len(ranks) != seat_count:
        raise ValueError(f"expected {seat_count} final_rank tokens, found {len(ranks)}")

    seats = [rank.seat for rank in ranks]
    expected_seats = list(range(seat_count))
    if seats != expected_seats:
        raise ValueError(f"final_rank seats must be {expected_seats}, got {seats}")

    places = [rank.place for rank in ranks]
    expected_places = set(range(1, seat_count + 1))
    if set(places) != expected_places:
        raise ValueError(f"final_rank places must be {sorted(expected_places)}, got {places}")

    return ranks


def build_outcome_conditioned_input_ids(
    input_ids: Sequence[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    seat_count: int,
) -> list[int]:
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must define bos_token_id and eos_token_id")
    final_rank_ids = extract_final_rank_token_ids(input_ids, tokenizer=tokenizer, seat_count=seat_count)
    return [int(bos_token_id), *final_rank_ids, *[int(token_id) for token_id in input_ids], int(eos_token_id)]


REINJECT_NONE = "none"
REINJECT_ROUND = "round"
REINJECT_TURN = "turn"

LOSS_SCOPE_FULL = "full"
LOSS_SCOPE_ACTIONS = "actions"

_TILE_RE = re.compile(r"^(?:[mps][0-9]|z[1-7])$")
_DISCARD_RE = re.compile(r"^discard_(\d+)_")
_TAKE_SELF_RE = re.compile(r"^take_self_(\d+)_(\w+)$")
_PASS_SELF_RE = re.compile(r"^pass_self_(\d+)_")
_TAKE_REACT_RE = re.compile(r"^take_react_(\d+)_(\w+)$")
_PASS_REACT_RE = re.compile(r"^pass_react_(\d+)_\w+_(voluntary|forced_priority)$")
_DETAIL_TOKENS = {"chi_pos_high", "chi_pos_mid", "chi_pos_low", "red_used", "red_not_used"}
_SELF_TILE_OPTS = {"ankan", "kakan", "penuki"}


def is_viewer_action_token(
    token: str,
    *,
    viewer_seat: int,
    state: dict,
) -> bool:
    """Stateful classifier: is ``token`` a *decision* made by the viewer?

    Decisions are: discards, self take/pass (riichi/kan/tsumo/...), reaction
    take / voluntary-pass (chi/pon/kan/ron), and the action sub-choices that
    follow them (chi meld position, red-five used/not-used, kan/penuki target
    tile). Everything else — option offerings (``opt_*``), forced-priority
    passes, draws (luck), opponents' public tiles, dora, scores — is *not* a
    viewer decision and is excluded from the loss.

    ``state`` is a mutable dict carried across the token stream; callers reset
    it once per sequence.
    """
    if token == "round_start":
        state["viewer_seat_round"] = None
        state["last_take_viewer"] = False
        state["expect_self_tile"] = False
        return False
    if token.startswith("draw_") and not token.endswith("_hidden"):
        state["viewer_seat_round"] = int(token.split("_")[1])
        state["expect_self_tile"] = False
        return False

    vs = state.get("viewer_seat_round")

    # Sub-choice: kan/penuki target tile immediately after a viewer take_self.
    if state.get("expect_self_tile"):
        state["expect_self_tile"] = False
        if _TILE_RE.match(token):
            return True

    # Sub-choices for chi/pon: meld position and red-five usage.
    if token in _DETAIL_TOKENS:
        return bool(state.get("last_take_viewer"))

    if vs is None:
        return False

    m = _DISCARD_RE.match(token)
    if m:
        return int(m.group(1)) == vs

    m = _TAKE_SELF_RE.match(token)
    if m:
        is_viewer = int(m.group(1)) == vs
        state["last_take_viewer"] = is_viewer
        state["expect_self_tile"] = is_viewer and m.group(2) in _SELF_TILE_OPTS
        return is_viewer

    m = _PASS_SELF_RE.match(token)
    if m:
        return int(m.group(1)) == vs

    m = _TAKE_REACT_RE.match(token)
    if m:
        is_viewer = int(m.group(1)) == vs
        state["last_take_viewer"] = is_viewer
        return is_viewer

    m = _PASS_REACT_RE.match(token)
    if m:
        return int(m.group(1)) == vs and m.group(2) == "voluntary"

    return False


def build_outcome_conditioned_v2(
    input_ids: Sequence[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    seat_count: int,
    viewer_seat: int,
    reinject: str = REINJECT_TURN,
    loss_scope: str = LOSS_SCOPE_FULL,
) -> tuple[list[int], list[bool]]:
    """Improved outcome conditioning that makes the target outcome reachable
    at every decision and trains a *conditional* model ``P(trajectory | outcome)``.

    Compared to :func:`build_outcome_conditioned_input_ids` this:

    1. Prepends the full final-rank vector after ``<bos>`` (same as v1).
    2. Re-injects the *viewer's* final-rank token close to each decision:
       - ``round``: once after every ``round_start``.
       - ``turn`` : right before every viewer (non-hidden) draw — i.e. adjacent
         to every discard/riichi/call decision the viewer makes.
    3. Returns a ``label_mask`` (``True`` -> keep label, ``False`` -> set to
       ``-100``) so the conditioning tokens (BOS, prefix ranks, re-injected
       ranks) are *not* predicted. The model only learns to generate the game
       conditioned on the outcome, which is the whole point and — per
       Russo (2026) — is what lets conditioning actually move the policy.

    The diagnosis behind this design: the v1 adapter was outcome-*insensitive*
    because the condition sat only at the very start (hundreds of tokens from
    each decision) and the loss also asked the model to predict the prepended
    ranks. See docs/research/outcome_conditioned_policy_analysis.md.
    """
    if reinject not in (REINJECT_NONE, REINJECT_ROUND, REINJECT_TURN):
        raise ValueError(f"invalid reinject mode: {reinject}")
    if loss_scope not in (LOSS_SCOPE_FULL, LOSS_SCOPE_ACTIONS):
        raise ValueError(f"invalid loss_scope: {loss_scope}")
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must define bos_token_id and eos_token_id")

    ranks = extract_final_rank_tokens(input_ids, tokenizer=tokenizer, seat_count=seat_count)
    rank_ids = [rank.token_id for rank in ranks]
    viewer_ranks = [rank for rank in ranks if rank.seat == viewer_seat]
    if len(viewer_ranks) != 1:
        raise ValueError(f"expected exactly one final_rank for viewer seat {viewer_seat}, found {len(viewer_ranks)}")
    viewer_rank_id = viewer_ranks[0].token_id

    tokens = tokenizer.convert_ids_to_tokens([int(t) for t in input_ids])

    out_ids: list[int] = [int(bos_token_id), *rank_ids]
    label_mask: list[bool] = [False] * len(out_ids)  # condition prefix is not predicted

    action_state: dict = {}
    for token_id, token in zip(input_ids, tokens):
        token_id = int(token_id)
        is_viewer_draw = isinstance(token, str) and token.startswith("draw_") and not token.endswith("_hidden")
        if reinject == REINJECT_TURN and is_viewer_draw:
            out_ids.append(viewer_rank_id)
            label_mask.append(False)
        if loss_scope == LOSS_SCOPE_ACTIONS:
            keep = is_viewer_action_token(token, viewer_seat=viewer_seat, state=action_state)
        else:
            keep = True
        out_ids.append(token_id)
        label_mask.append(keep)
        if reinject == REINJECT_ROUND and token == "round_start":
            out_ids.append(viewer_rank_id)
            label_mask.append(False)

    out_ids.append(int(eos_token_id))
    label_mask.append(loss_scope == LOSS_SCOPE_FULL)
    return out_ids, label_mask


def build_outcome_conditioned_v3(
    input_ids: Sequence[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    seat_count: int,
    viewer_seat: int,
    reinject: str = REINJECT_TURN,
    loss_scope: str = LOSS_SCOPE_ACTIONS,
    override_bucket: str | None = None,
    round_buckets: Sequence[str] | None = None,
) -> tuple[list[int], list[bool]]:
    """v3 = **proximal** outcome conditioning.

    The v2 dissection found terminal final-rank conditioning carries ~0.003
    nats about per-move actions (rank is too distal). v3 instead injects, near
    every viewer decision, the bucketed *score delta the viewer realised in the
    current round* — a proximal outcome with far higher action-influence.

    Sequence: ``<bos> final_rank... <round_oc_b> <body...> <eos>`` where the
    round-outcome bucket token is re-injected before every viewer draw (``turn``)
    or once per ``round_start`` (``round``). The terminal final-rank block is
    kept as cheap global (placement) context. All conditioning tokens are masked
    from the loss; ``loss_scope='actions'`` keeps loss on viewer decisions only.

    ``override_bucket`` forces every round to a given bucket token (for
    inference / counterfactual probing, e.g. condition every round on
    ``round_oc_bigwin``). When ``None`` the true per-round buckets are used.
    """
    from gpt2.round_outcome import viewer_round_deltas

    if reinject not in (REINJECT_NONE, REINJECT_ROUND, REINJECT_TURN):
        raise ValueError(f"invalid reinject mode: {reinject}")
    if loss_scope not in (LOSS_SCOPE_FULL, LOSS_SCOPE_ACTIONS):
        raise ValueError(f"invalid loss_scope: {loss_scope}")
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must define bos_token_id and eos_token_id")

    tokens = tokenizer.convert_ids_to_tokens([int(t) for t in input_ids])
    ranks = extract_final_rank_tokens(input_ids, tokenizer=tokenizer, seat_count=seat_count)
    rank_ids = [rank.token_id for rank in ranks]

    round_info = viewer_round_deltas(tokens, viewer_seat=viewer_seat, seat_count=seat_count)
    if round_buckets is not None:
        if len(round_buckets) != len(round_info):
            raise ValueError(f"round_buckets length {len(round_buckets)} != rounds {len(round_info)}")
        buckets = list(round_buckets)
    else:
        buckets = [override_bucket or info["bucket"] for info in round_info]
    bucket_ids = [tokenizer.convert_tokens_to_ids(b) for b in buckets]
    unk_id = tokenizer.unk_token_id
    if any(bid is None or bid == unk_id for bid in bucket_ids):
        raise ValueError("round-outcome bucket tokens are not in the tokenizer; extend it for v3")

    out_ids: list[int] = [int(bos_token_id), *rank_ids]
    label_mask: list[bool] = [False] * len(out_ids)

    action_state: dict = {}
    ridx = -1
    for token_id, token in zip(input_ids, tokens):
        token_id = int(token_id)
        if token == "round_start":
            ridx += 1
        cur_bucket = bucket_ids[ridx] if 0 <= ridx < len(bucket_ids) else None
        is_viewer_draw = isinstance(token, str) and token.startswith("draw_") and not token.endswith("_hidden")

        # turn-level: inject the current round's bucket right before each viewer draw
        if reinject == REINJECT_TURN and is_viewer_draw and cur_bucket is not None:
            out_ids.append(cur_bucket)
            label_mask.append(False)

        if loss_scope == LOSS_SCOPE_ACTIONS:
            keep = is_viewer_action_token(token, viewer_seat=viewer_seat, state=action_state)
        else:
            keep = True
        out_ids.append(token_id)
        label_mask.append(keep)

        # round-level: inject the bucket immediately after round_start
        if reinject == REINJECT_ROUND and token == "round_start" and cur_bucket is not None:
            out_ids.append(cur_bucket)
            label_mask.append(False)

    out_ids.append(int(eos_token_id))
    label_mask.append(loss_scope == LOSS_SCOPE_FULL)
    return out_ids, label_mask


def extend_tokenizer_for_v3(tokenizer: PreTrainedTokenizerBase) -> int:
    """Add the round-outcome bucket tokens to the tokenizer (idempotent).

    Returns the number of tokens actually added. Call ``model.resize_token_
    embeddings(len(tokenizer))`` afterwards. Token ids are assigned in fixed
    order so train and eval/probe tokenizers stay consistent.
    """
    from gpt2.round_outcome import ROUND_OC_TOKENS

    missing = [t for t in ROUND_OC_TOKENS if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing:
        tokenizer.add_tokens(missing)
    return len(missing)
