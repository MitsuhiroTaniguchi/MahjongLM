from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer.engine import TenhouTokenizer, TokenizeError, encode_tenbo_tokens
from tests.fixtures.synthetic_logs import qipai_payload


def test_encode_tenbo_tokens_decomposes_by_stick_units() -> None:
    assert encode_tenbo_tokens(25000) == [
        "TENBO_PLUS",
        "TENBO_10000",
        "TENBO_10000",
        "TENBO_5000",
    ]
    assert encode_tenbo_tokens(-3900) == [
        "TENBO_MINUS",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
    ]
    assert encode_tenbo_tokens(0) == ["TENBO_ZERO"]


def test_encode_tenbo_tokens_rejects_non_100_multiple() -> None:
    with pytest.raises(TokenizeError):
        encode_tenbo_tokens(12345)


def test_qipai_emits_score_as_tenbo_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    i = tokenizer.tokens.index("score_0")
    assert tokenizer.tokens[i : i + 5] == [
        "score_0",
        "TENBO_PLUS",
        "TENBO_10000",
        "TENBO_10000",
        "TENBO_5000",
    ]


def test_result_emits_score_delta_as_tenbo_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer._on_hule({"l": 0, "fenpei": [-3900, 3900, 0, 0]})

    i0 = tokenizer.tokens.index("score_delta_0")
    assert tokenizer.tokens[i0 + 1 : i0 + 14] == [
        "TENBO_MINUS",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
    ]

    i2 = tokenizer.tokens.index("score_delta_2")
    assert tokenizer.tokens[i2 + 1] == "TENBO_ZERO"
