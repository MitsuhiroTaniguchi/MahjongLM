from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer import TenhouTokenizer
from tests.dataset_sample import DATASET_2023, get_dataset_2023_sample_zip
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event
from tests.validation_helpers import (
    validate_round_stepwise,
    validate_score_rotation,
    validate_token_stream,
)

ROOT = Path(__file__).resolve().parents[1]


def test_validation_helpers_accept_minimal_game() -> None:
    game = minimal_game([qipai_event(), pingju_event()])
    tokens = TenhouTokenizer().tokenize_game(game)

    validate_token_stream(tokens)
    validate_score_rotation(game)
    validate_round_stepwise(game["log"][0])


def test_stepwise_validation_accepts_call_then_discard_round() -> None:
    game = minimal_game(
        [
            qipai_event(hands=["m1456789p1234s11", "m23p123456s123z11", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "s2"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"fulou": {"l": 1, "m": "m1-23"}},
            {"dapai": {"l": 1, "p": "z1"}},
            pingju_event(),
        ]
    )

    validate_round_stepwise(game["log"][0])


def test_stepwise_validation_accepts_ankan_and_rinshan_round() -> None:
    game = minimal_game(
        [
            qipai_event(hands=["m111p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "m1"}},
            {"gang": {"l": 0, "m": "m1111"}},
            {"gangzimo": {"l": 0, "p": "p5"}},
            {"dapai": {"l": 0, "p": "p5_"}},
            pingju_event(),
        ]
    )

    validate_round_stepwise(game["log"][0])


@pytest.mark.slow
def test_dataset_validation_first_500_games() -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    with zipfile.ZipFile(get_dataset_2023_sample_zip()) as zf:
        for name in zf.namelist():
            game = json.load(zf.open(name))
            tokens = TenhouTokenizer().tokenize_game(game)
            validate_token_stream(tokens)
            validate_score_rotation(game)
            for round_data in game.get("log", []):
                validate_round_stepwise(round_data)
