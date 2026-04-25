from __future__ import annotations

import json
import zipfile

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer import TenhouTokenizer
from tests.dataset_sample import (
    DATASET_2023,
    DATASET_2023_CURATED_GAME_IDS,
    get_dataset_2023_curated_zip,
)
from tests.validation_helpers import (
    validate_round_stepwise,
    validate_score_rotation,
    validate_token_stream,
)


CURATED_EXPECTED_TOKENS = {
    "2023/2023021622gm-00a9-0000-75ec253d.txt": {"pingju_kyushukyuhai", "take_self_2_ankan"},
    "2023/2023021117gm-00a9-0000-d0a6d793.txt": {"pingju_sukantsu", "take_self_0_kakan"},
    "2023/2023052310gm-00a9-0000-2faa0711.txt": {"pingju_sanchahou", "take_react_1_ron"},
    "2023/2023070721gm-00a9-0000-d6a86063.txt": {"take_react_1_minkan", "pingju_kyushukyuhai"},
    "2023/2023073004gm-00a9-0000-ddb6dc93.txt": {"take_self_0_ankan", "take_react_2_minkan"},
    "2023/2023080301gm-00a9-0000-d28602aa.txt": {"pingju_nagashimangan", "take_react_0_ron"},
    "2023/2023081821gm-00a9-0000-7cf4f26e.txt": {"pingju_suuchariichi", "take_self_3_riichi"},
    "2023/2023091610gm-00a9-0000-bdcf1ac5.txt": {"take_self_0_kakan", "take_self_0_ankan"},
    "2023/2023091810gm-00a9-0000-1428c108.txt": {"take_react_1_ron", "take_react_3_ron"},
    "2023/2023112221gm-00a9-0000-6e1a680f.txt": {"take_self_0_kakan", "take_self_1_tsumo"},
    "2023/2023092616gm-00a9-0000-cedf6b55.txt": {"take_react_2_ron"},
    "2023/2023081200gm-00a9-0000-b14c9329.txt": {"take_react_0_ron"},
    "2023/2023102817gm-00a9-0000-3abd1803.txt": {"take_react_1_ron"},
}


@pytest.mark.slow
def test_curated_corpus_games_cover_expected_rare_paths() -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    with zipfile.ZipFile(get_dataset_2023_curated_zip()) as zf:
        assert set(zf.namelist()) == set(DATASET_2023_CURATED_GAME_IDS)
        for game_id in DATASET_2023_CURATED_GAME_IDS:
            game = json.load(zf.open(game_id))
            tokens = TenhouTokenizer().tokenize_game(game)
            validate_token_stream(tokens)
            validate_score_rotation(game)
            for round_data in game.get("log", []):
                validate_round_stepwise(round_data)
            expected = CURATED_EXPECTED_TOKENS[game_id]
            assert expected.issubset(set(tokens)), game_id


def test_curated_game_ids_are_unique() -> None:
    assert len(DATASET_2023_CURATED_GAME_IDS) == len(set(DATASET_2023_CURATED_GAME_IDS))
