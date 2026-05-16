from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from tenhou_tokenizer.engine import TokenizeError
from tenhou_tokenizer.wall import assert_wall_consistent_with_game, generate_tenhou_wall_tokens, tile_id_to_token


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


def _ordered_wall_tokens(
    hands: list[str],
    *,
    draws: list[str] | None = None,
    dora: list[str] | None = None,
    ura: list[str] | None = None,
) -> list[str]:
    from tenhou_tokenizer.engine import _parse_tiles

    seat_count = len(hands)
    wall_len = 108 if seat_count == 3 else 136
    wall: list[str | None] = [None] * wall_len
    cursor = 0
    hand_tiles = [_parse_tiles(hand, stop_at_comma=True, context="test_ordered_wall") for hand in hands]
    hand_offsets = [0] * seat_count
    for _chunk in range(3):
        for seat in range(seat_count):
            tiles = hand_tiles[seat][hand_offsets[seat] : hand_offsets[seat] + 4]
            wall[cursor : cursor + 4] = tiles
            cursor += 4
            hand_offsets[seat] += 4
    for seat in range(seat_count):
        tiles = hand_tiles[seat][hand_offsets[seat] : hand_offsets[seat] + 1]
        wall[cursor : cursor + 1] = tiles
        cursor += 1
    for tile in draws or []:
        wall[cursor] = tile
        cursor += 1
    for index, tile in enumerate(dora or []):
        wall[wall_len - 6 - 2 * index] = tile
    for index, tile in enumerate(ura or []):
        wall[wall_len - 5 - 2 * index] = tile

    remaining = Counter(_canonical_wall_tokens())
    for tile in wall:
        if tile is not None:
            remaining[tile] -= 1
    fill = list(remaining.elements())
    for index, tile in enumerate(wall):
        if tile is None:
            wall[index] = fill.pop()
    assert not fill
    return [str(tile) for tile in wall]


def _sanma_raw_id_to_token(tile_id: int) -> str:
    if 0 <= tile_id <= 3:
        compact = tile_id
    elif 32 <= tile_id <= 35:
        compact = 4 + (tile_id - 32)
    elif 36 <= tile_id <= 71:
        compact = 8 + (tile_id - 36)
    elif 72 <= tile_id <= 107:
        compact = 44 + (tile_id - 72)
    elif 108 <= tile_id <= 135:
        compact = 80 + (tile_id - 108)
    else:
        raise ValueError(tile_id)
    return tile_id_to_token(compact, seat_count=3)


def _tokens_to_hand(tokens: list[str]) -> str:
    return "".join(tokens)


def _qipai_payload(hands: list[str], baopai: str) -> dict:
    from tenhou_tokenizer.engine import _parse_tiles

    return {
        "shoupai": hands,
        "baopai": baopai,
        "_ordered_shoupai_tokens": [
            _parse_tiles(hand, stop_at_comma=True, context="test_ordered_qipai")
            for hand in hands
        ],
    }


def test_wall_assertion_rejects_invalid_red_five_multiset() -> None:
    wall = _canonical_wall_tokens()
    wall[wall.index("m0")] = "m5"
    hands = ["m123456789p1234"] * 4
    game = {"log": [[{"qipai": _qipai_payload(hands, "m1")}]]}

    with pytest.raises(TokenizeError, match="wall tile multiset is invalid"):
        assert_wall_consistent_with_game(game, [wall])


def test_wall_assertion_checks_dora_and_ura_indicators() -> None:
    hands = ["m123406789p1234", *["m123456789p1234"] * 3]
    wall = _ordered_wall_tokens(hands, dora=["z7"], ura=["z7"])
    game = {
        "log": [
            [
                {"qipai": _qipai_payload(hands, "z7")},
                {"hule": {"fubaopai": ["z7"]}},
            ]
        ]
    }

    assert_wall_consistent_with_game(game, [wall])

    impossible_indicators = {
        "log": [
            [
                {"qipai": _qipai_payload(hands, "z7")},
                {"hule": {"fubaopai": ["z7", "z7", "z7", "z7"]}},
            ]
        ]
    }
    with pytest.raises(TokenizeError, match="observed tile contradicts reconstructed wall|ura indicator contradicts"):
        assert_wall_consistent_with_game(impossible_indicators, [wall])


def test_wall_assertion_counts_shared_double_ron_ura_indicators_once() -> None:
    hands = [
        "m123406789p1345",
        "p123406789s1345",
        "s123406789z1234",
        "z1234567m111p11s1",
    ]
    wall = _ordered_wall_tokens(hands, draws=["p2", "p2"], dora=["z7"], ura=["p2"])
    game = {
        "log": [
            [
                {"qipai": _qipai_payload(hands, "z7")},
                {"zimo": {"l": 0, "p": "p2"}},
                {"zimo": {"l": 0, "p": "p2"}},
                {"hule": {"l": 1, "fubaopai": ["p2"]}},
                {"hule": {"l": 2, "fubaopai": ["p2"]}},
            ]
        ]
    }

    assert_wall_consistent_with_game(game, [wall])


def test_wall_assertion_rejects_wrong_order_even_when_multiset_matches() -> None:
    hands = ["m123406789p1234", *["m123456789p1234"] * 3]
    wall = _ordered_wall_tokens(hands, dora=["z7"])
    wall[0], wall[1] = wall[1], wall[0]
    game = {"log": [[{"qipai": _qipai_payload(hands, "z7")}]]}

    with pytest.raises(TokenizeError, match="wall order mismatch"):
        assert_wall_consistent_with_game(game, [wall])


def test_sanma_wall_uses_108_tiles_and_checks_rinshan_order_from_seed() -> None:
    raw_path = Path(__file__).resolve().parent / "fixtures" / "tenhou" / "2014091101gm-00b9-0000-5ca6b487.txt"
    raw = raw_path.read_text(encoding="utf-8")
    seed = re.search(r'SHUFFLE\s+[^>]*seed="mt19937ar-sha512-n288-base64,([^"]+)"', raw).group(1)
    wall = generate_tenhou_wall_tokens(seed, 1, seat_count=3)[0]
    assert len(wall) == 108
    assert not any(token in wall for token in {"m0", "m2", "m3", "m4", "m5", "m6", "m7", "m8"})

    init = re.search(r"<INIT ([^>]*)/>", raw).group(1)

    def attr(name: str) -> str:
        return re.search(name + r'="([^"]*)"', init).group(1)

    hands = [
        _tokens_to_hand([_sanma_raw_id_to_token(int(tile_id)) for tile_id in attr(f"hai{seat}").split(",")])
        for seat in range(3)
    ]
    game = {
        "title": "三鳳南喰赤",
        "log": [
            [
                {
                    "qipai": {
                        "shoupai": hands,
                        "baopai": _sanma_raw_id_to_token(75),
                        "_ordered_shoupai_tokens": [
                            [_sanma_raw_id_to_token(int(tile_id)) for tile_id in attr(f"hai{seat}").split(",")]
                            for seat in range(3)
                        ],
                        "_wall_oya": int(attr("oya")),
                    }
                },
                {"zimo": {"l": 0, "p": _sanma_raw_id_to_token(66)}},
                {"penuki": {"l": 0, "p": "z4"}},
                {"zimo": {"l": 0, "p": _sanma_raw_id_to_token(1)}},
                {"zimo": {"l": 1, "p": _sanma_raw_id_to_token(33)}},
            ]
        ],
    }

    assert_wall_consistent_with_game(game, [wall])
