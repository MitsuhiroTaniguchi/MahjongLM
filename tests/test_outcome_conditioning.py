from __future__ import annotations

import pytest

from gpt2.outcome_conditioning import (
    build_outcome_conditioned_input_ids,
    extract_final_rank_token_ids,
)
from tenhou_tokenizer.huggingface import MahjongTokenizerFast


@pytest.fixture()
def tokenizer() -> MahjongTokenizerFast:
    return MahjongTokenizerFast.from_pretrained("tokenizer")


def ids(tokenizer: MahjongTokenizerFast, tokens: list[str]) -> list[int]:
    return tokenizer.convert_tokens_to_ids(tokens)


def test_build_outcome_conditioned_ids_adds_bos_rank_prefix_and_eos(tokenizer: MahjongTokenizerFast) -> None:
    original_tokens = [
        "view_imperfect_0",
        "game_start",
        "final_score_0",
        "final_score_1",
        "final_score_2",
        "final_score_3",
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "final_rank_3_4",
    ]
    original_ids = ids(tokenizer, original_tokens)

    conditioned = build_outcome_conditioned_input_ids(original_ids, tokenizer=tokenizer, seat_count=4)

    assert conditioned == ids(
        tokenizer,
        [
            "<bos>",
            "final_rank_0_1",
            "final_rank_1_2",
            "final_rank_2_3",
            "final_rank_3_4",
            *original_tokens,
            "<eos>",
        ],
    )
    assert original_ids == ids(tokenizer, original_tokens)


def test_extract_final_rank_ids_accepts_sanma(tokenizer: MahjongTokenizerFast) -> None:
    input_ids = ids(
        tokenizer,
        [
            "view_imperfect_2",
            "game_start",
            "final_rank_0_2",
            "final_rank_1_3",
            "final_rank_2_1",
        ],
    )

    assert extract_final_rank_token_ids(input_ids, tokenizer=tokenizer, seat_count=3) == ids(
        tokenizer,
        ["final_rank_0_2", "final_rank_1_3", "final_rank_2_1"],
    )


def test_extract_final_rank_ids_rejects_duplicate_places(tokenizer: MahjongTokenizerFast) -> None:
    input_ids = ids(
        tokenizer,
        [
            "view_imperfect_0",
            "final_rank_0_1",
            "final_rank_1_1",
            "final_rank_2_3",
            "final_rank_3_4",
        ],
    )

    with pytest.raises(ValueError, match="final_rank places"):
        extract_final_rank_token_ids(input_ids, tokenizer=tokenizer, seat_count=4)


def test_extract_final_rank_ids_rejects_extra_yonma_rank_for_sanma(tokenizer: MahjongTokenizerFast) -> None:
    input_ids = ids(
        tokenizer,
        [
            "view_imperfect_0",
            "final_rank_0_1",
            "final_rank_1_2",
            "final_rank_2_3",
            "final_rank_3_4",
        ],
    )

    with pytest.raises(ValueError, match="expected 3 final_rank tokens"):
        extract_final_rank_token_ids(input_ids, tokenizer=tokenizer, seat_count=3)
