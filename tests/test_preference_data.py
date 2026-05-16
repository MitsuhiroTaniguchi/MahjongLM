from __future__ import annotations

import random

from gpt2.preference_data import (
    GeneratedGame,
    build_preference_pairs,
    extract_final_ranks,
    omniscient_to_imperfect_tokens,
)


WALL = ["m1"] * 136


def _tokens(*, ranks: tuple[int, int, int, int], draw1: str = "p5") -> tuple[str, ...]:
    return (
        "<bos>",
        "view_omniscient",
        "rule_player_4",
        "rule_length_hanchan",
        "game_start",
        "round_start",
        "wall",
        *WALL,
        "haipai_0",
        *["m1"] * 13,
        "haipai_1",
        *["p1"] * 13,
        "haipai_2",
        *["s1"] * 13,
        "haipai_3",
        *["z1"] * 13,
        "draw_1_" + draw1,
        "opt_self_1_riichi",
        "take_self_1_riichi",
        "draw_0_m2",
        "opt_self_0_riichi",
        "pass_self_0_riichi",
        "game_end",
        *(f"final_rank_{seat}_{rank}" for seat, rank in enumerate(ranks)),
        "<eos>",
    )


def test_extract_final_ranks() -> None:
    assert extract_final_ranks(_tokens(ranks=(1, 4, 2, 3))) == {0: 1, 1: 4, 2: 2, 3: 3}


def test_omniscient_to_imperfect_masks_private_information() -> None:
    tokens = omniscient_to_imperfect_tokens(_tokens(ranks=(1, 4, 2, 3)), viewer_seat=0)
    assert tokens[0] == "<bos>"
    assert "view_imperfect_0" in tokens
    assert "view_omniscient" not in tokens
    assert "wall" not in tokens
    assert "haipai_0" in tokens
    assert "haipai_1" not in tokens
    assert "hidden_haipai_1" in tokens
    assert "draw_1_hidden" in tokens
    assert "draw_1_p5" not in tokens
    assert "opt_self_1_riichi" not in tokens
    assert "take_self_1_riichi" in tokens
    assert "opt_self_0_riichi" in tokens


def test_build_preference_pairs_discards_same_rank_batch() -> None:
    games = [GeneratedGame(tokens=_tokens(ranks=(1, 2, 3, 4)), generation_index=i) for i in range(2)]
    assert build_preference_pairs(games, rng=random.Random(0)) == []


def test_build_preference_pairs_selects_better_rank_per_viewer() -> None:
    games = [
        GeneratedGame(tokens=_tokens(ranks=(1, 4, 2, 3)), generation_index=0, seed_id="seed", rule_key="rule_player_4"),
        GeneratedGame(tokens=_tokens(ranks=(4, 1, 3, 2), draw1="p6"), generation_index=1, seed_id="seed", rule_key="rule_player_4"),
    ]
    pairs = build_preference_pairs(games, rng=random.Random(0))
    assert len(pairs) == 4
    seat0 = next(pair for pair in pairs if pair.viewer_seat == 0)
    assert seat0.chosen_generation_index == 0
    assert seat0.rejected_generation_index == 1
    assert seat0.prompt == "<bos>"
    assert seat0.chosen.startswith("view_imperfect_0 ")
    assert seat0.chosen.endswith("<eos>")
