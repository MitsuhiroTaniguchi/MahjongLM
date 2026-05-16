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


def token_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def dpo_prompt_and_completion(tokens: Sequence[str]) -> tuple[str, str]:
    normalized = ensure_bos_eos(tokens)
    return normalized[0], token_text(normalized[1:])


def ensure_bos_eos(tokens: Sequence[str]) -> tuple[str, ...]:
    out = list(tokens)
    if not out or out[0] != "<bos>":
        out.insert(0, "<bos>")
    if out[-1] != "<eos>":
        out.append("<eos>")
    return tuple(out)


def infer_seat_count(tokens: Sequence[str]) -> int:
    for token in tokens:
        match = RULE_PLAYER_RE.match(token)
        if match:
            return int(match.group("count"))
    ranks = extract_final_ranks(tokens, allow_partial=True)
    if ranks:
        return max(ranks) + 1
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
    while idx < len(tokens):
        token = tokens[idx]
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
            if seat == viewer_seat:
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
                out.append(token if seat == viewer_seat else f"draw_{seat}_hidden")
                idx += 1
                continue
        if token.startswith("opt_self_"):
            keep = _seat_from_action_token(token, "opt_self_") == viewer_seat
            idx = _copy_self_option_payload(tokens, idx, out if keep else None)
            continue
        if token.startswith("pass_self_"):
            keep = _seat_from_action_token(token, "pass_self_") == viewer_seat
            idx = _copy_optional_tile_payload(tokens, idx, out if keep else None)
            continue
        if token.startswith("opt_react_"):
            if _seat_from_action_token(token, "opt_react_") == viewer_seat:
                out.append(token)
            idx += 1
            continue
        if token.startswith("pass_react_"):
            if _seat_from_action_token(token, "pass_react_") == viewer_seat:
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
    rank_vectors = [tuple(extract_final_ranks(game.tokens)[seat] for seat in range(seat_count)) for game in games]
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


def read_generation_batches(path: Path) -> Iterable[list[GeneratedGame]]:
    with path.open("r", encoding="utf-8") as handle:
        pending: dict[str, list[GeneratedGame]] = {}
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "generations" in record:
                yield [
                    GeneratedGame(
                        tokens=tuple(item["tokens"]),
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
                    tokens=tuple(record["tokens"]),
                    generation_index=int(record.get("generation_index", len(games))),
                    seed_id=str(record.get("seed_id", "")),
                    rule_key=str(record.get("rule_key", "")),
                )
            )
        yield from pending.values()


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
