from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer.engine import (
    TokenizeError,
    classify_fulou,
    classify_gang,
    parse_hand_counts,
    parse_meld_tiles,
    parse_meld_token_tiles_and_called,
    tile_to_index,
    token_tile,
)


@pytest.mark.parametrize(
    ("meld", "expected_tiles", "expected_called_index"),
    [
        ("m1-23", ["m1", "m2", "m3"], 1),
        ("m12-3", ["m1", "m2", "m3"], 2),
        ("m123-", ["m1", "m2", "m3"], 2),
        ("m05+5", ["m0", "m5", "m5"], 2),
        ("z666=6", ["z6", "z6", "z6", "z6"], 3),
    ],
)
def test_parse_meld_token_tiles_and_called_matrix(
    meld: str,
    expected_tiles: list[str],
    expected_called_index: int,
) -> None:
    tiles, called_index = parse_meld_token_tiles_and_called(meld)

    assert tiles == expected_tiles
    assert called_index == expected_called_index


@pytest.mark.parametrize(
    ("meld", "expected"),
    [
        ("m1-23", "chi"),
        ("m111+", "pon"),
        ("z1111", "minkan"),
        ("m1111", "ankan"),
        ("m1111+", "kakan"),
        ("z666=6", "kakan"),
    ],
)
def test_classify_meld_matrix(meld: str, expected: str) -> None:
    if expected in {"chi", "pon", "minkan"}:
        assert classify_fulou(parse_meld_tiles(meld)) == expected
    else:
        assert classify_gang(meld) == expected


@pytest.mark.parametrize("tile", ["x1", "m", "m10", "z8", "m*", "11"])
def test_token_tile_rejects_invalid_formats(tile: str) -> None:
    with pytest.raises(TokenizeError):
        token_tile(tile)


@pytest.mark.parametrize("hand", ["123m", "m12x", "abc"])
def test_parse_hand_counts_rejects_invalid_hand_text(hand: str) -> None:
    with pytest.raises(TokenizeError):
        parse_hand_counts(hand)


@pytest.mark.parametrize("meld", ["", "123", "x123", "m", "+++", "5-67"])
def test_parse_meld_tiles_rejects_invalid_text(meld: str) -> None:
    with pytest.raises(TokenizeError):
        parse_meld_tiles(meld)


def test_tile_to_index_normalizes_red_five() -> None:
    assert tile_to_index("m0") == tile_to_index("m5")
