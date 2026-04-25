from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import PlayerState, TenhouTokenizer, TokenizeError, parse_hand_counts, tile_to_index
from tests.fixtures.synthetic_logs import qipai_payload


def _player_state(hand: str) -> PlayerState:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload(hands=[hand, "m123456789p1234", "m123456789p1234", "m123456789p1234"]))
    return tokenizer.players[0]


def test_consume_unspecified_tile_prefers_normal_five_before_red() -> None:
    tokenizer = TenhouTokenizer()
    player = _player_state("m055p123s123z1122")

    tokenizer._consume_unspecified_tile(player, tile_to_index("m5"), n=1)
    assert player.concealed[tile_to_index("m5")] == 2
    assert player.red_fives["m"] == 1

    tokenizer._consume_unspecified_tile(player, tile_to_index("m5"), n=1)
    assert player.concealed[tile_to_index("m5")] == 1
    assert player.red_fives["m"] == 1

    tokenizer._consume_unspecified_tile(player, tile_to_index("m5"), n=1)
    assert player.concealed[tile_to_index("m5")] == 0
    assert player.red_fives["m"] == 0


def test_consume_concealed_token_rejects_missing_red_tile() -> None:
    tokenizer = TenhouTokenizer()
    player = _player_state("m55p123s123z11223")

    with pytest.raises(TokenizeError):
        tokenizer._consume_concealed_token(player, "m0")


def test_infer_consumed_meld_tokens_skips_called_tile_copy() -> None:
    tokenizer = TenhouTokenizer()
    player = _player_state("m056p123s123z1122")

    consumed = tokenizer._infer_consumed_meld_tokens(player, ["m0", "m5", "m5"], called_index=2)

    assert consumed == ["m0", "m5"]


def test_infer_consumed_meld_tokens_raises_when_hand_cannot_supply_tiles() -> None:
    tokenizer = TenhouTokenizer()
    player = _player_state("m1p123456s123z112")

    with pytest.raises(TokenizeError):
        tokenizer._infer_consumed_meld_tokens(player, ["m1", "m2", "m3"], called_index=2)


def test_red_choice_token_only_emits_when_choice_exists() -> None:
    tokenizer = TenhouTokenizer()
    pre_counts = parse_hand_counts("m05p123s123z11223")
    pre_red = {"m": 1, "p": 0, "s": 0}

    token = tokenizer._red_choice_token(
        action="pon",
        consumed_tokens=["m0", "m5"],
        pre_counts=pre_counts,
        pre_red_fives=pre_red,
    )

    assert token is None


def test_red_choice_token_distinguishes_used_vs_not_used() -> None:
    tokenizer = TenhouTokenizer()
    pre_counts = parse_hand_counts("m055p123s123z1122")
    pre_red = {"m": 1, "p": 0, "s": 0}

    used = tokenizer._red_choice_token(
        action="pon",
        consumed_tokens=["m0", "m5"],
        pre_counts=pre_counts,
        pre_red_fives=pre_red,
    )
    not_used = tokenizer._red_choice_token(
        action="pon",
        consumed_tokens=["m5", "m5"],
        pre_counts=pre_counts,
        pre_red_fives=pre_red,
    )

    assert used == "red_used"
    assert not_used == "red_not_used"


def test_fulou_fallback_path_emits_unknown_detail_and_still_consumes() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m055p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("m5"),
        options_by_player={0: {"pon"}},
    )

    original = tokenizer._infer_consumed_meld_tokens

    def fail_once(*_args, **_kwargs):
        tokenizer._infer_consumed_meld_tokens = original
        raise TokenizeError("forced fallback")

    tokenizer._infer_consumed_meld_tokens = fail_once  # type: ignore[method-assign]
    before = sum(tokenizer.players[0].concealed)

    tokenizer._on_fulou({"l": 0, "m": "m05+5"})

    after = sum(tokenizer.players[0].concealed)
    assert "event_unknown_fulou_detail" in tokenizer.tokens
    assert "take_react_0_pon" in tokenizer.tokens
    assert after == before - 2


def test_fulou_rejects_missing_pending_reaction() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m055p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )

    with pytest.raises(TokenizeError):
        tokenizer._on_fulou({"l": 0, "m": "m05+5"})


def test_fulou_rejects_unoffered_action() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m055p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("m5"),
        options_by_player={0: {"chi"}},
    )

    with pytest.raises(TokenizeError):
        tokenizer._on_fulou({"l": 0, "m": "m05+5"})


def test_kakan_consumes_normal_five_when_open_pon_already_used_red() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    actor = 0
    tile = tile_to_index("m5")
    player = tokenizer.players[actor]
    player.concealed = [0] * 34
    player.red_fives = {"m": 1, "p": 0, "s": 0}
    player.concealed[tile] = 2
    player.open_pons[tile] = 1
    player.open_pons_red[tile] = 1
    player.melds = [("pon", tile)]
    tokenizer.pending_self = engine.SelfDecision(actor=actor, options={"kakan"})
    tokenizer.expected_discard_actor = actor

    tokenizer._on_gang({"l": actor, "m": "m5550+"})

    assert player.concealed[tile] == 1
    assert player.red_fives["m"] == 1
    assert tile not in player.open_pons
    assert tile not in player.open_pons_red
    assert player.melds == [("minkan", tile)]


def test_kakan_consumes_red_five_when_added_tile_is_red() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    actor = 0
    tile = tile_to_index("m5")
    player = tokenizer.players[actor]
    player.concealed = [0] * 34
    player.red_fives = {"m": 1, "p": 0, "s": 0}
    player.concealed[tile] = 1
    player.open_pons[tile] = 1
    player.melds = [("pon", tile)]
    tokenizer.pending_self = engine.SelfDecision(actor=actor, options={"kakan"})
    tokenizer.expected_discard_actor = actor

    tokenizer._on_gang({"l": actor, "m": "m5550+"})

    assert player.concealed[tile] == 0
    assert player.red_fives["m"] == 0
    assert tile not in player.open_pons
    assert tile not in player.open_pons_red
    assert player.melds == [("minkan", tile)]


def test_kakan_fallback_consumes_unspecified_tile_when_red_bookkeeping_is_out_of_sync() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    actor = 0
    tile = tile_to_index("m5")
    player = tokenizer.players[actor]
    player.concealed = [0] * 34
    player.red_fives = {"m": 1, "p": 0, "s": 0}
    player.concealed[tile] = 1
    player.open_pons[tile] = 1
    player.open_pons_red[tile] = 0
    player.melds = [("pon", tile)]
    tokenizer.pending_self = engine.SelfDecision(actor=actor, options={"kakan"})
    tokenizer.expected_discard_actor = actor

    tokenizer._on_gang({"l": actor, "m": "m5550+"})

    assert player.concealed[tile] == 0
    assert player.red_fives["m"] == 0
    assert player.melds == [("minkan", tile)]


def test_gang_rejects_missing_pending_self() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload(hands=["m1111p123s123z112", "m123456789p1234", "m123456789p1234", "m123456789p1234"]))

    with pytest.raises(TokenizeError):
        tokenizer._on_gang({"l": 0, "m": "m1111"})


def test_gang_rejects_unoffered_action() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload(hands=["m1111p123s123z112", "m123456789p1234", "m123456789p1234", "m123456789p1234"]))
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"riichi"})

    with pytest.raises(TokenizeError):
        tokenizer._on_gang({"l": 0, "m": "m1111"})
