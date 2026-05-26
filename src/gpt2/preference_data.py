from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tenhou_tokenizer.views import TILE_TOKENS
from tenhou_tokenizer.viewspec import (
    TOKEN_VIEW_COMPLETE,
    TOKEN_VIEW_OMNISCIENT,
    imperfect_view_token,
)


FINAL_RANK_RE = re.compile(r"^final_rank_(?P<seat>[0-3])_(?P<place>[1-4])$")
RULE_PLAYER_RE = re.compile(r"^rule_player_(?P<count>[34])$")
KYOKU_RE = re.compile(r"^kyoku_(?P<kyoku>[0-3])$")
SEAT_TOKEN_RE = re.compile(
    r"^(?:haipai|hidden_haipai|draw|score|final_score|rank|final_rank)_(?P<seat>[0-3])(?:_|$)"
)


@dataclass(frozen=True)
class GeneratedGame:
    tokens: tuple[str, ...]
    generation_index: int
    seed_id: str = ""
    rule_key: str = ""


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    seed_id: str
    rule_key: str
    viewer_seat: int
    chosen_rank: int
    rejected_rank: int
    chosen_generation_index: int
    rejected_generation_index: int

    def as_record(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "seed_id": self.seed_id,
            "rule_key": self.rule_key,
            "viewer_seat": self.viewer_seat,
            "chosen_rank": self.chosen_rank,
            "rejected_rank": self.rejected_rank,
            "chosen_generation_index": self.chosen_generation_index,
            "rejected_generation_index": self.rejected_generation_index,
        }

    def as_tokenized_record(self, tokenizer) -> dict[str, object]:
        chosen_tokens = self.chosen.split()
        rejected_tokens = self.rejected.split()
        return {
            "prompt_input_ids": tokenizer.convert_tokens_to_ids(self.prompt.split()),
            "chosen_input_ids": tokenizer.convert_tokens_to_ids(chosen_tokens),
            "rejected_input_ids": tokenizer.convert_tokens_to_ids(rejected_tokens),
            "chosen_loss_mask": completion_loss_mask(chosen_tokens, self.viewer_seat),
            "rejected_loss_mask": completion_loss_mask(rejected_tokens, self.viewer_seat),
            "seed_id": self.seed_id,
            "rule_key": self.rule_key,
            "viewer_seat": self.viewer_seat,
            "chosen_rank": self.chosen_rank,
            "rejected_rank": self.rejected_rank,
            "chosen_generation_index": self.chosen_generation_index,
            "rejected_generation_index": self.rejected_generation_index,
        }


def token_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def dpo_prompt_and_completion(tokens: Sequence[str]) -> tuple[str, str]:
    normalized = ensure_bos_eos(tokens)
    split_idx = _dpo_prompt_end(normalized)
    return token_text(normalized[:split_idx]), token_text(normalized[split_idx:])


def ensure_bos_eos(tokens: Sequence[str]) -> tuple[str, ...]:
    out = list(tokens)
    if not out or out[0] != "<bos>":
        out.insert(0, "<bos>")
    if out[-1] != "<eos>":
        out.append("<eos>")
    return tuple(out)


def _dpo_prompt_end(tokens: Sequence[str]) -> int:
    try:
        game_start_idx = tokens.index("game_start")
    except ValueError as exc:
        raise ValueError("DPO sequence must contain game_start") from exc
    return game_start_idx + 1


def infer_seat_count(tokens: Sequence[str]) -> int:
    for token in tokens:
        match = RULE_PLAYER_RE.match(token)
        if match:
            return int(match.group("count"))
    ranks = extract_final_ranks(tokens, allow_partial=True)
    if ranks:
        return max(ranks) + 1
    max_seat = -1
    for token in tokens:
        match = SEAT_TOKEN_RE.match(token)
        if match:
            max_seat = max(max_seat, int(match.group("seat")))
        for prefix in (
            "discard_",
            "opt_self_",
            "pass_self_",
            "take_self_",
            "opt_react_",
            "pass_react_",
            "take_react_",
        ):
            if token.startswith(prefix):
                max_seat = max(max_seat, _seat_from_action_token(token, prefix))
                break
    if max_seat >= 0:
        return max_seat + 1
    raise ValueError("could not infer seat count from tokens")


def extract_final_ranks(tokens: Sequence[str], *, allow_partial: bool = False) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for token in tokens:
        match = FINAL_RANK_RE.match(token)
        if not match:
            continue
        ranks[int(match.group("seat"))] = int(match.group("place"))
    if allow_partial:
        return ranks
    seat_count = infer_seat_count(tokens) if not ranks else max(ranks) + 1
    if len(ranks) < seat_count:
        raise ValueError(f"missing final rank tokens: expected {seat_count}, got {len(ranks)}")
    return {seat: ranks[seat] for seat in range(seat_count)}


def omniscient_to_imperfect_tokens(tokens: Sequence[str], viewer_seat: int) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    seat_count = infer_seat_count(tokens)
    viewer_round_seat = viewer_seat % seat_count
    while idx < len(tokens):
        token = tokens[idx]
        kyoku_match = KYOKU_RE.match(token)
        if kyoku_match:
            viewer_round_seat = _viewer_round_seat(viewer_seat, int(kyoku_match.group("kyoku")), seat_count)
            out.append(token)
            idx += 1
            continue
        if token in {TOKEN_VIEW_OMNISCIENT, TOKEN_VIEW_COMPLETE} or token.startswith("view_imperfect_"):
            out.append(imperfect_view_token(viewer_seat))
            idx += 1
            continue
        if token == "wall":
            idx += 1
            while idx < len(tokens) and tokens[idx] in TILE_TOKENS:
                idx += 1
            continue
        if token.startswith("haipai_"):
            seat = int(token.split("_")[1])
            if seat == viewer_round_seat:
                out.append(token)
                out.extend(tokens[idx + 1 : idx + 14])
            else:
                out.append(f"hidden_haipai_{seat}")
            idx += 14
            continue
        if token.startswith("draw_"):
            parts = token.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                seat = int(parts[1])
                out.append(token if seat == viewer_round_seat else f"draw_{seat}_hidden")
                idx += 1
                continue
        if token.startswith("opt_self_"):
            keep = _seat_from_action_token(token, "opt_self_") == viewer_round_seat
            idx = _copy_self_option_payload(tokens, idx, out if keep else None)
            continue
        if token.startswith("pass_self_"):
            keep = _seat_from_action_token(token, "pass_self_") == viewer_round_seat
            idx = _copy_optional_tile_payload(tokens, idx, out if keep else None)
            continue
        if token.startswith("opt_react_"):
            if _seat_from_action_token(token, "opt_react_") == viewer_round_seat:
                out.append(token)
            idx += 1
            continue
        if token.startswith("pass_react_"):
            if _seat_from_action_token(token, "pass_react_") == viewer_round_seat:
                out.append(token)
            idx += 1
            continue
        out.append(token)
        idx += 1
    return ensure_bos_eos(out)


def build_preference_pairs(
    games: Sequence[GeneratedGame],
    *,
    rng: random.Random | None = None,
) -> list[PreferencePair]:
    if not games:
        return []
    rng = rng or random.Random()
    seat_count = infer_seat_count(games[0].tokens)
    valid_games: list[GeneratedGame] = []
    rank_vectors: list[tuple[int, ...]] = []
    for game in games:
        try:
            ranks = extract_final_ranks(game.tokens)
        except ValueError:
            continue
        if any(seat not in ranks for seat in range(seat_count)):
            continue
        valid_games.append(game)
        rank_vectors.append(tuple(ranks[seat] for seat in range(seat_count)))
    games = valid_games
    if len(games) < 2:
        return []
    if len(set(rank_vectors)) <= 1:
        return []

    pairs: list[PreferencePair] = []
    for seat in range(seat_count):
        min_rank = min(vector[seat] for vector in rank_vectors)
        max_rank = max(vector[seat] for vector in rank_vectors)
        if min_rank == max_rank:
            continue
        chosen_candidates = [game for game, vector in zip(games, rank_vectors) if vector[seat] == min_rank]
        rejected_candidates = [game for game, vector in zip(games, rank_vectors) if vector[seat] == max_rank]
        chosen = rng.choice(chosen_candidates)
        rejected = rng.choice(rejected_candidates)
        chosen_tokens = omniscient_to_imperfect_tokens(chosen.tokens, seat)
        rejected_tokens = omniscient_to_imperfect_tokens(rejected.tokens, seat)
        prompt, chosen_text = dpo_prompt_and_completion(chosen_tokens)
        rejected_prompt, rejected_text = dpo_prompt_and_completion(rejected_tokens)
        if prompt != rejected_prompt:
            raise ValueError("chosen/rejected prompts diverged after BOS normalization")
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen_text,
                rejected=rejected_text,
                seed_id=chosen.seed_id or rejected.seed_id,
                rule_key=chosen.rule_key or rejected.rule_key,
                viewer_seat=seat,
                chosen_rank=min_rank,
                rejected_rank=max_rank,
                chosen_generation_index=chosen.generation_index,
                rejected_generation_index=rejected.generation_index,
            )
        )
    return pairs


def read_generation_batches(path: Path, *, tokenizer=None) -> Iterable[list[GeneratedGame]]:
    with path.open("r", encoding="utf-8") as handle:
        pending: dict[str, list[GeneratedGame]] = {}
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "generations" in record:
                yield [
                    GeneratedGame(
                        tokens=_generation_tokens(item, tokenizer=tokenizer),
                        generation_index=int(item.get("generation_index", idx)),
                        seed_id=str(record.get("seed_id", "")),
                        rule_key=str(record.get("rule_key", "")),
                    )
                    for idx, item in enumerate(record["generations"])
                ]
                continue
            batch_id = str(record.get("batch_id", record.get("seed_id", line_number)))
            games = pending.setdefault(batch_id, [])
            games.append(
                GeneratedGame(
                    tokens=_generation_tokens(record, tokenizer=tokenizer),
                    generation_index=int(record.get("generation_index", len(games))),
                    seed_id=str(record.get("seed_id", "")),
                    rule_key=str(record.get("rule_key", "")),
                )
            )
        yield from pending.values()


def _generation_tokens(record: dict[str, object], *, tokenizer=None) -> tuple[str, ...]:
    tokens = record.get("tokens")
    if tokens is not None:
        return tuple(str(token) for token in tokens)
    input_ids = record.get("input_ids")
    if input_ids is None:
        raise ValueError("generation record must contain either tokens or input_ids")
    if tokenizer is None:
        raise ValueError("tokenizer is required to read input_ids generation records")
    return tuple(tokenizer.convert_ids_to_tokens(list(input_ids)))


def completion_loss_mask(tokens: Sequence[str], viewer_seat: int) -> list[int]:
    try:
        seat_count = infer_seat_count(tokens)
    except ValueError:
        seat_count = 4
    viewer_round_seat = viewer_seat % seat_count
    mask: list[int] = []
    for token in tokens:
        kyoku_match = KYOKU_RE.match(token)
        if kyoku_match:
            viewer_round_seat = _viewer_round_seat(viewer_seat, int(kyoku_match.group("kyoku")), seat_count)
            mask.append(0)
            continue
        mask.append(1 if _is_round_seat_decision_token(token, viewer_round_seat) else 0)
    return mask


def is_viewer_decision_token(token: str, viewer_seat: int) -> bool:
    return _is_round_seat_decision_token(token, viewer_seat)


def _is_round_seat_decision_token(token: str, round_seat: int) -> bool:
    if token.startswith("discard_"):
        parts = token.split("_")
        return len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) == round_seat
    for prefix in ("take_self_", "pass_self_", "take_react_", "pass_react_"):
        if token.startswith(prefix):
            return _seat_from_action_token(token, prefix) == round_seat
    return False


def _viewer_round_seat(viewer_player: int, kyoku: int, seat_count: int) -> int:
    return (viewer_player - kyoku) % seat_count


def _seat_from_action_token(token: str, prefix: str) -> int:
    return int(token.removeprefix(prefix).split("_", 1)[0])


def _copy_self_option_payload(tokens: Sequence[str], idx: int, out: list[str] | None) -> int:
    if out is not None:
        out.append(tokens[idx])
    idx += 1
    while idx < len(tokens) and tokens[idx] in TILE_TOKENS:
        if out is not None:
            out.append(tokens[idx])
        idx += 1
    return idx


def _copy_optional_tile_payload(tokens: Sequence[str], idx: int, out: list[str] | None) -> int:
    if out is not None:
        out.append(tokens[idx])
    idx += 1
    if idx < len(tokens) and tokens[idx] in TILE_TOKENS:
        if out is not None:
            out.append(tokens[idx])
        idx += 1
    return idx
