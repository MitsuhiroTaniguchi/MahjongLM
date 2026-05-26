from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from gpt2.preference_data import (
    GeneratedGame,
    completion_loss_mask,
    dpo_prompt_and_completion,
    extract_final_ranks,
    infer_seat_count,
    omniscient_to_imperfect_tokens,
)


FINAL_SCORE_RE = re.compile(r"^final_score_(?P<seat>[0-3])$")
TENBO_RE = re.compile(r"^TENBO_(?P<amount>[0-9]+)$")


@dataclass(frozen=True)
class RolloutSeatRecord:
    prompt: str
    completion: str
    seed_id: str
    rule_key: str
    generation_index: int
    viewer_seat: int
    rank: int
    final_score: int | None
    reward: float
    group_reward_mean: float
    group_reward_std: float
    advantage: float

    def as_tokenized_record(self, tokenizer) -> dict[str, object]:
        completion_tokens = self.completion.split()
        return {
            "prompt_input_ids": tokenizer.convert_tokens_to_ids(self.prompt.split()),
            "completion_input_ids": tokenizer.convert_tokens_to_ids(completion_tokens),
            "loss_mask": completion_loss_mask(completion_tokens, self.viewer_seat),
            "reward": float(self.reward),
            "group_reward_mean": float(self.group_reward_mean),
            "group_reward_std": float(self.group_reward_std),
            "advantage": float(self.advantage),
            "seed_id": self.seed_id,
            "rule_key": self.rule_key,
            "generation_index": self.generation_index,
            "viewer_seat": self.viewer_seat,
            "rank": self.rank,
            "final_score": self.final_score,
        }


def build_group_rl_records(
    games: Sequence[GeneratedGame],
    *,
    placement_weight: float = 1.0,
    score_weight: float = 0.2,
    normalize_advantage: bool = True,
) -> list[RolloutSeatRecord]:
    if not games:
        return []
    seat_count = infer_seat_count(games[0].tokens)
    valid: list[tuple[GeneratedGame, dict[int, int], dict[int, int]]] = []
    for game in games:
        try:
            ranks = extract_final_ranks(game.tokens)
        except ValueError:
            continue
        if any(seat not in ranks for seat in range(seat_count)):
            continue
        if any(ranks[seat] < 1 or ranks[seat] > seat_count for seat in range(seat_count)):
            continue
        valid.append((game, ranks, extract_final_scores(game.tokens)))
    if len(valid) < 2:
        return []

    rewards_by_seat: dict[int, list[float]] = {seat: [] for seat in range(seat_count)}
    payloads: list[tuple[GeneratedGame, int, int, int | None, float]] = []
    for game, ranks, scores in valid:
        for seat in range(seat_count):
            reward = placement_weight * placement_reward(ranks[seat], seat_count)
            if seat in scores:
                reward += score_weight * score_reward(scores[seat], seat_count)
            rewards_by_seat[seat].append(reward)
            payloads.append((game, seat, ranks[seat], scores.get(seat), reward))

    stats_by_seat: dict[int, tuple[float, float]] = {}
    for seat, rewards in rewards_by_seat.items():
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        std = math.sqrt(max(variance, 0.0))
        stats_by_seat[seat] = (mean, std)

    records: list[RolloutSeatRecord] = []
    for game, seat, rank, final_score, reward in payloads:
        mean, std = stats_by_seat[seat]
        advantage = reward - mean
        if normalize_advantage and std > 1e-8:
            advantage /= std
        imperfect_tokens = omniscient_to_imperfect_tokens(game.tokens, seat)
        prompt, completion = dpo_prompt_and_completion(imperfect_tokens)
        records.append(
            RolloutSeatRecord(
                prompt=prompt,
                completion=completion,
                seed_id=game.seed_id,
                rule_key=game.rule_key,
                generation_index=game.generation_index,
                viewer_seat=seat,
                rank=rank,
                final_score=final_score,
                reward=reward,
                group_reward_mean=mean,
                group_reward_std=std,
                advantage=advantage,
            )
        )
    return records


def placement_reward(rank: int, seat_count: int) -> float:
    if seat_count == 4:
        return {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}[rank]
    if seat_count == 3:
        return {1: 1.0, 2: 0.0, 3: -1.0}[rank]
    raise ValueError(f"seat_count must be 3 or 4, got {seat_count}")


def score_reward(score: int, seat_count: int) -> float:
    baseline = 35000 if seat_count == 3 else 25000
    return (score - baseline) / 50000.0


def extract_final_scores(tokens: Sequence[str]) -> dict[int, int]:
    scores: dict[int, int] = {}
    idx = 0
    while idx < len(tokens):
        match = FINAL_SCORE_RE.match(tokens[idx])
        if not match:
            idx += 1
            continue
        seat = int(match.group("seat"))
        idx += 1
        sign = 1
        total = 0
        while idx < len(tokens):
            token = tokens[idx]
            if FINAL_SCORE_RE.match(token) or token.startswith("final_rank_") or token == "<eos>":
                break
            if token == "TENBO_PLUS":
                sign = 1
            elif token == "TENBO_MINUS":
                sign = -1
            else:
                amount_match = TENBO_RE.match(token)
                if amount_match:
                    total += int(amount_match.group("amount"))
            idx += 1
        scores[seat] = sign * total
    return scores
