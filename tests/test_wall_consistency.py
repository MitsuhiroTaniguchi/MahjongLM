from __future__ import annotations

import pytest

from tenhou_tokenizer.engine import TokenizeError
from tenhou_tokenizer.wall import assert_wall_consistent_with_game


def _canonical_wall_tokens() -> list[str]:
    tiles: list[str] = []
    for suit in ("m", "p", "s"):
        for number in range(1, 10):
            if number == 5:
                tiles.extend([f"{suit}0", f"{suit}5", f"{suit}5", f"{suit}5"])
            else:
                tiles.extend([f"{suit}{number}"] * 4)
    for number in range(1, 8):
        tiles.extend([f"z{number}"] * 4)
    assert len(tiles) == 136
    return tiles


def test_wall_assertion_rejects_invalid_red_five_multiset() -> None:
    wall = _canonical_wall_tokens()
    wall[wall.index("m0")] = "m5"
    game = {"log": [[{"qipai": {"shoupai": ["m123456789p1234"] * 4, "baopai": "m1"}}]]}

    with pytest.raises(TokenizeError, match="wall tile multiset is invalid"):
        assert_wall_consistent_with_game(game, [wall])


def test_wall_assertion_checks_dora_and_ura_indicators() -> None:
    wall = _canonical_wall_tokens()
    hands = ["m123406789p1234", *["m123456789p1234"] * 3]
    game = {
        "log": [
            [
                {"qipai": {"shoupai": hands, "baopai": "z7"}},
                {"hule": {"fubaopai": ["z7"]}},
            ]
        ]
    }

    assert_wall_consistent_with_game(game, [wall])

    impossible_indicators = {
        "log": [
            [
                {"qipai": {"shoupai": hands, "baopai": "z7"}},
                {"hule": {"fubaopai": ["z7", "z7", "z7", "z7"]}},
            ]
        ]
    }
    with pytest.raises(TokenizeError, match="observed tile contradicts reconstructed wall"):
        assert_wall_consistent_with_game(impossible_indicators, [wall])


def test_wall_assertion_counts_shared_double_ron_ura_indicators_once() -> None:
    wall = _canonical_wall_tokens()
    game = {
        "log": [
            [
                {"qipai": {"shoupai": ["p2", "", "", ""], "baopai": "m1"}},
                {"zimo": {"l": 0, "p": "p2"}},
                {"zimo": {"l": 0, "p": "p2"}},
                {"hule": {"l": 1, "fubaopai": ["p2"]}},
                {"hule": {"l": 2, "fubaopai": ["p2"]}},
            ]
        ]
    }

    assert_wall_consistent_with_game(game, [wall])
