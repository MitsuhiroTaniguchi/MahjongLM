from __future__ import annotations

from tenhou_tokenizer.engine import TenhouTokenizer
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event
from tests.validation_helpers import validate_token_stream


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
    return tiles


def test_token_stream_validator_accepts_omniscient_wall_block() -> None:
    tokens = TenhouTokenizer().tokenize_game(minimal_game([qipai_event(), pingju_event()]))
    tokens.insert(1, "view_omniscient")
    round_start = tokens.index("round_start")
    tokens[round_start + 1 : round_start + 1] = ["wall", *_canonical_wall_tokens()]

    validate_token_stream(tokens)
