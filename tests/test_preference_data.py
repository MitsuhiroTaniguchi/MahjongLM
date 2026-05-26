from __future__ import annotations

import random

from gpt2.preference_data import (
    GeneratedGame,
    build_preference_pairs,
    completion_loss_mask,
    extract_final_ranks,
    omniscient_to_imperfect_tokens,
)


WALL = ["m1"] * 136


def _tokens(*, ranks: tuple[int, int, int, int], draw1: str = "p5") -> tuple[str, ...]:
    return (
        "<bos>",
        "rule_player_4",
        "rule_length_hanchan",
        "view_omniscient",
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
        "discard_0_m2_tedashi",
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


def test_omniscient_to_imperfect_tracks_qijia_player_across_round_rotation() -> None:
    source = (
        "<bos>",
        "rule_player_4",
        "rule_length_hanchan",
        "view_omniscient",
        "game_start",
        "round_start",
        "wall",
        *WALL,
        "bakaze_0",
        "kyoku_1",
        "honba",
        "TENBO_ZERO",
        "riichi_sticks",
        "TENBO_ZERO",
        "dora",
        "m1",
        "haipai_0",
        *["m1"] * 13,
        "haipai_1",
        *["p1"] * 13,
        "haipai_2",
        *["s1"] * 13,
        "haipai_3",
        *["z1"] * 13,
        "draw_3_p5",
        "opt_self_3_riichi",
        "pass_self_3_riichi",
        "discard_3_p5_tedashi",
        "draw_0_m2",
        "opt_self_0_riichi",
        "pass_self_0_riichi",
        "discard_0_m2_tedashi",
        "game_end",
        *(f"final_rank_{seat}_{rank}" for seat, rank in enumerate((1, 2, 3, 4))),
        "<eos>",
    )

    tokens = omniscient_to_imperfect_tokens(source, viewer_seat=0)

    assert "haipai_3" in tokens
    assert "hidden_haipai_0" in tokens
    assert "draw_3_p5" in tokens
    assert "draw_0_hidden" in tokens
    assert "opt_self_3_riichi" in tokens
    assert "opt_self_0_riichi" not in tokens


def test_final_score_and_final_rank_remain_player_coordinates() -> None:
    source = (
        "<bos>",
        "rule_player_4",
        "rule_length_hanchan",
        "view_omniscient",
        "game_start",
        "round_start",
        "wall",
        *WALL,
        "bakaze_0",
        "kyoku_1",
        "honba",
        "TENBO_ZERO",
        "riichi_sticks",
        "TENBO_ZERO",
        "dora",
        "m1",
        "haipai_3",
        *["z1"] * 13,
        "discard_3_p5_tedashi",
        "game_end",
        "final_score_0",
        "TENBO_30000",
        "final_score_1",
        "TENBO_25000",
        "final_score_2",
        "TENBO_20000",
        "final_score_3",
        "TENBO_15000",
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "final_rank_3_4",
        "<eos>",
    )

    tokens = omniscient_to_imperfect_tokens(source, viewer_seat=0)

    assert tokens[tokens.index("final_score_0") : tokens.index("final_rank_0_1")] == (
        "final_score_0",
        "TENBO_30000",
        "final_score_1",
        "TENBO_25000",
        "final_score_2",
        "TENBO_20000",
        "final_score_3",
        "TENBO_15000",
    )
    assert tokens[tokens.index("final_rank_0_1") : tokens.index("<eos>")] == (
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "final_rank_3_4",
    )


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
    assert seat0.prompt == "<bos> rule_player_4 rule_length_hanchan view_imperfect_0 game_start"
    assert seat0.chosen.startswith("round_start ")
    assert seat0.chosen.endswith("<eos>")


def test_preference_pair_can_emit_tokenized_record() -> None:
    games = [
        GeneratedGame(tokens=_tokens(ranks=(1, 4, 2, 3)), generation_index=0, seed_id="seed", rule_key="rule_player_4"),
        GeneratedGame(tokens=_tokens(ranks=(4, 1, 3, 2), draw1="p6"), generation_index=1, seed_id="seed", rule_key="rule_player_4"),
    ]
    pair = next(pair for pair in build_preference_pairs(games, rng=random.Random(0)) if pair.viewer_seat == 0)
    record = pair.as_tokenized_record(_TokenIdMap())

    assert set(record) >= {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
    assert len(record["prompt_input_ids"]) == 5
    assert isinstance(record["chosen_input_ids"][0], int)
    assert isinstance(record["rejected_input_ids"][0], int)
    assert set(record) >= {"chosen_loss_mask", "rejected_loss_mask"}
    assert sum(record["chosen_loss_mask"]) == 2
    assert len(record["chosen_loss_mask"]) == len(record["chosen_input_ids"])


def test_completion_loss_mask_keeps_only_viewer_decisions() -> None:
    tokens = [
        "round_start",
        "draw_0_m1",
        "opt_self_0_riichi",
        "pass_self_0_riichi",
        "take_self_1_riichi",
        "discard_0_m1_tedashi",
        "discard_1_p1_tsumogiri",
        "take_react_0_ron",
        "pass_react_2_pon_voluntary",
    ]

    assert completion_loss_mask(tokens, viewer_seat=0) == [0, 0, 0, 1, 0, 1, 0, 1, 0]


def test_completion_loss_mask_rotates_qijia_player_by_kyoku() -> None:
    tokens = [
        "round_start",
        "kyoku_1",
        "draw_3_p5",
        "pass_self_3_riichi",
        "discard_3_p5_tedashi",
        "pass_self_0_riichi",
        "discard_0_m2_tedashi",
        "game_end",
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "final_rank_3_4",
        "<eos>",
    ]

    assert completion_loss_mask(tokens, viewer_seat=0) == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_completion_loss_mask_rotates_three_player_qijia_player() -> None:
    tokens = [
        "round_start",
        "kyoku_2",
        "discard_1_p5_tedashi",
        "discard_2_m2_tedashi",
        "game_end",
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "<eos>",
    ]

    assert completion_loss_mask(tokens, viewer_seat=0) == [0, 0, 1, 0, 0, 0, 0, 0, 0]


class _TokenIdMap:
    def __init__(self) -> None:
        self.ids = {"<bos>": 1, "<eos>": 2}
        self.next_id = 3

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._id(tokens)
        return [self._id(token) for token in tokens]

    def _id(self, token: str) -> int:
        if token not in self.ids:
            self.ids[token] = self.next_id
            self.next_id += 1
        return self.ids[token]
