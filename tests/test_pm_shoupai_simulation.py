from __future__ import annotations

import json
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer import TenhouTokenizer
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event

ROOT = Path(__file__).resolve().parents[1]
DATASET_2023 = ROOT / "data" / "raw" / "tenhou" / "data2023.zip"


@contextmanager
def _simulation_flags(enabled: bool) -> Iterator[None]:
    prev_stateless = engine.PM_STATELESS_SIMULATION_API_AVAILABLE
    prev_sim = engine.PM_SIMULATION_API_AVAILABLE
    prev_shoupai = engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE
    engine.PM_STATELESS_SIMULATION_API_AVAILABLE = enabled
    engine.PM_SIMULATION_API_AVAILABLE = enabled
    engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE = enabled
    try:
        yield
    finally:
        engine.PM_STATELESS_SIMULATION_API_AVAILABLE = prev_stateless
        engine.PM_SIMULATION_API_AVAILABLE = prev_sim
        engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE = prev_shoupai


def _tokenize(game: dict, *, simulation_api: bool) -> list[str]:
    with _simulation_flags(simulation_api):
        return TenhouTokenizer().tokenize_game(game)


@pytest.mark.skipif(
    not engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE,
    reason="pymahjong shoupai simulation APIs are unavailable",
)
@pytest.mark.parametrize(
    "game",
    [
        minimal_game([qipai_event(), pingju_event()]),
        minimal_game(
            [
                qipai_event(
                    hands=[
                        "m1456789p1234s11",
                        "m23p123456s123z11",
                        "m123456789p1234",
                        "m123456789p1234",
                    ]
                ),
                {"zimo": {"l": 0, "p": "s2"}},
                {"dapai": {"l": 0, "p": "m1"}},
                {"fulou": {"l": 1, "m": "m1-23"}},
                {"dapai": {"l": 1, "p": "z1"}},
                pingju_event(),
            ]
        ),
        minimal_game(
            [
                qipai_event(
                    hands=[
                        "m111p123s123z1122",
                        "m123456789p1234",
                        "m123456789p1234",
                        "m123456789p1234",
                    ]
                ),
                {"zimo": {"l": 0, "p": "m1"}},
                {"gang": {"l": 0, "m": "m1111"}},
                {"gangzimo": {"l": 0, "p": "p5"}},
                {"dapai": {"l": 0, "p": "p5_"}},
                pingju_event(),
            ]
        ),
        minimal_game(
            [
                qipai_event(
                    hands=[
                        "m666p123s123z1122",
                        "m123456789p1234",
                        "m123456789p1234",
                        "m123456789p1234",
                    ]
                ),
                {"zimo": {"l": 3, "p": "p1"}},
                {"dapai": {"l": 3, "p": "m6"}},
                {"fulou": {"l": 0, "m": "m6666-"}},
                {"gangzimo": {"l": 0, "p": "p1"}},
                {"dapai": {"l": 0, "p": "p1_"}},
                {"kaigang": {"baopai": "z1"}},
                pingju_event(),
            ]
        ),
    ],
)
def test_shoupai_simulation_matches_python_fallback(game: dict) -> None:
    baseline = _tokenize(game, simulation_api=False)
    accelerated = _tokenize(game, simulation_api=True)

    assert accelerated == baseline


@pytest.mark.slow
@pytest.mark.skipif(
    not engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE,
    reason="pymahjong shoupai simulation APIs are unavailable",
)
def test_shoupai_simulation_matches_python_fallback_on_real_games() -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    with zipfile.ZipFile(DATASET_2023) as zf:
        for name in zf.namelist()[:25]:
            game = json.load(zf.open(name))
            baseline = _tokenize(game, simulation_api=False)
            accelerated = _tokenize(game, simulation_api=True)
            assert accelerated == baseline, name
