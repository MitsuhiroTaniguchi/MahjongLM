from __future__ import annotations

from gpt2.preference_data import GeneratedGame
from gpt2.rl_data import (
    build_group_rl_records,
    extract_final_scores,
    placement_reward,
    score_reward,
)


def _tokens(*, ranks: tuple[int, int, int, int], scores: tuple[int, int, int, int]) -> tuple[str, ...]:
    score_tokens: list[str] = []
    for seat, score in enumerate(scores):
        score_tokens.append(f"final_score_{seat}")
        score_tokens.append("TENBO_PLUS" if score >= 0 else "TENBO_MINUS")
        remaining = abs(score)
        for amount in (10000, 5000, 1000, 500, 100):
            while remaining >= amount:
                score_tokens.append(f"TENBO_{amount}")
                remaining -= amount
    return (
        "<bos>",
        "rule_player_4",
        "rule_length_tonpu",
        "view_omniscient",
        "game_start",
        "round_start",
        "wall",
        *(["m1"] * 136),
        "haipai_0",
        *(["m1"] * 13),
        "haipai_1",
        *(["p1"] * 13),
        "haipai_2",
        *(["s1"] * 13),
        "haipai_3",
        *(["z1"] * 13),
        "draw_0_m2",
        "discard_0_m2_tedashi",
        "draw_1_p2",
        "discard_1_p2_tedashi",
        "round_end",
        "game_end",
        *score_tokens,
        *(f"final_rank_{seat}_{rank}" for seat, rank in enumerate(ranks)),
        "<eos>",
    )


def test_extract_final_scores_from_tenbo_tokens() -> None:
    scores = extract_final_scores(_tokens(ranks=(1, 2, 3, 4), scores=(25500, -1200, 35000, 0)))

    assert scores == {0: 25500, 1: -1200, 2: 35000, 3: 0}


def test_placement_and_score_rewards() -> None:
    assert placement_reward(1, 4) == 1.0
    assert placement_reward(4, 4) == -1.0
    assert placement_reward(2, 3) == 0.0
    assert score_reward(30000, 4) == 0.1
    assert score_reward(25000, 3) == -0.2


def test_build_group_rl_records_emits_group_advantages() -> None:
    games = [
        GeneratedGame(tokens=_tokens(ranks=(1, 4, 2, 3), scores=(40000, 10000, 30000, 20000)), generation_index=0),
        GeneratedGame(tokens=_tokens(ranks=(4, 1, 3, 2), scores=(10000, 40000, 20000, 30000)), generation_index=1),
    ]

    rows = build_group_rl_records(games, score_weight=0.0)
    seat0 = [row for row in rows if row.viewer_seat == 0]

    assert len(rows) == 8
    assert {row.advantage for row in seat0} == {1.0, -1.0}
    assert seat0[0].prompt == "<bos> rule_player_4 rule_length_tonpu view_imperfect_0 game_start"
    assert seat0[0].completion.startswith("round_start ")
