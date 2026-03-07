from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import TenhouTokenizer, TokenizeError, encode_tenbo_tokens, tile_to_index
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event, qipai_payload


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
        "TENBO_500",
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


def test_qipai_emits_honba_and_riichi_sticks_as_tenbo_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    honba_idx = tokenizer.tokens.index("honba")
    riichi_idx = tokenizer.tokens.index("riichi_sticks")

    assert tokenizer.tokens[honba_idx : honba_idx + 2] == ["honba", "TENBO_ZERO"]
    assert tokenizer.tokens[riichi_idx : riichi_idx + 2] == ["riichi_sticks", "TENBO_ZERO"]


def test_qipai_emits_nonzero_honba_and_riichi_sticks_as_tenbo_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        {
            **qipai_payload(),
            "changbang": 2,
            "lizhibang": 3,
        }
    )

    honba_idx = tokenizer.tokens.index("honba")
    riichi_idx = tokenizer.tokens.index("riichi_sticks")

    assert tokenizer.tokens[honba_idx : honba_idx + 4] == [
        "honba",
        "TENBO_PLUS",
        "TENBO_100",
        "TENBO_100",
    ]
    assert tokenizer.tokens[riichi_idx : riichi_idx + 5] == [
        "riichi_sticks",
        "TENBO_PLUS",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_1000",
    ]


def test_result_emits_score_delta_as_tenbo_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})
    tokenizer._on_hule({"l": 0, "fenpei": [-3900, 3900, 0, 0]})

    i0 = tokenizer.tokens.index("score_delta_0")
    assert tokenizer.tokens[i0 + 1 : i0 + 10] == [
        "TENBO_MINUS",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_1000",
        "TENBO_500",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
        "TENBO_100",
    ]

    i2 = tokenizer.tokens.index("score_delta_2")
    assert tokenizer.tokens[i2 + 1] == "TENBO_ZERO"


def test_hule_emits_ron_then_score_deltas_in_seat_order() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={2: {"ron"}},
    )
    tokenizer._on_hule({"l": 2, "baojia": 0, "fenpei": [-1000, 2000, -500, -500]})

    ron_idx = tokenizer.tokens.index("ron_from_2_0")
    delta_positions = [tokenizer.tokens.index(f"score_delta_{seat}") for seat in range(4)]
    assert ron_idx < delta_positions[0] < delta_positions[1] < delta_positions[2] < delta_positions[3]


def test_hule_rejects_ron_without_pending_reaction() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    with pytest.raises(TokenizeError):
        tokenizer._on_hule({"l": 2, "baojia": 0, "fenpei": [0, 2000, -1000, -1000]})


def test_hule_fenpei_must_use_integer_scores() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    with pytest.raises(TokenizeError):
        tokenizer._on_hule({"l": 0, "fenpei": [0, "1000", -500, -500]})


def test_pingju_emits_draw_name_before_score_deltas() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer._on_pingju({"name": "流局", "fenpei": [1000, -1000, 0, 0]})

    draw_idx = tokenizer.tokens.index("pingju_ryukyoku")
    delta_positions = [tokenizer.tokens.index(f"score_delta_{seat}") for seat in range(4)]
    assert draw_idx < delta_positions[0] < delta_positions[1] < delta_positions[2] < delta_positions[3]


def test_pingju_unknown_name_is_rejected() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    with pytest.raises(TokenizeError):
        tokenizer._on_pingju({"name": "流局 123!", "fenpei": [0, 0, 0, 0]})


def test_normal_pingju_closes_pending_self_before_result_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"riichi"})

    tokens = TenhouTokenizer().tokenize_game(
        minimal_game(
            [
                qipai_event(),
                {"zimo": {"l": 0, "p": "m1"}},
                pingju_event(),
            ]
        )
    )

    pass_idx = tokens.index("pass_self_0_riichi")
    draw_idx = tokens.index("pingju_ryukyoku")
    delta_idx = tokens.index("score_delta_0")
    assert pass_idx < draw_idx < delta_idx


def test_pingju_closes_pending_reaction_before_result_tokens() -> None:
    def fake_reaction(
        _self: TenhouTokenizer,
        discarder: int,
        tile_idx: int,
    ) -> engine.ReactionDecision:
        return engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        )

    tokenizer = TenhouTokenizer()
    tokenizer._compute_reaction_options = fake_reaction.__get__(tokenizer, TenhouTokenizer)
    tokenizer._compute_self_options = lambda *_args, **_kwargs: set()
    tokens = tokenizer.tokenize_game(
        minimal_game(
            [
                qipai_event(),
                {"zimo": {"l": 0, "p": "m1"}},
                {"dapai": {"l": 0, "p": "m1"}},
                pingju_event(),
            ]
        )
    )

    pass_idx = tokens.index("pass_react_1_ron_forced_rule")
    draw_idx = tokens.index("pingju_ryukyoku")
    delta_idx = tokens.index("score_delta_0")
    assert pass_idx < draw_idx < delta_idx


def test_next_draw_closes_riichi_ron_window_before_draw_token() -> None:
    def fake_reaction(
        _self: TenhouTokenizer,
        discarder: int,
        tile_idx: int,
    ) -> engine.ReactionDecision:
        return engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        )

    tokenizer = TenhouTokenizer()
    tokenizer._compute_reaction_options = fake_reaction.__get__(tokenizer, TenhouTokenizer)
    tokenizer._compute_self_options = lambda *_args, **_kwargs: {"riichi"} if _args[1] == 0 else set()
    tokens = tokenizer.tokenize_game(
        minimal_game(
            [
                qipai_event(),
                {"zimo": {"l": 0, "p": "m1"}},
                {"dapai": {"l": 0, "p": "m1*"}},
                {"zimo": {"l": 1, "p": "m2"}},
                pingju_event(),
            ]
        )
    )

    pass_idx = tokens.index("pass_react_1_ron_voluntary")
    draw_idx = tokens.index("draw_1_m2")
    assert pass_idx < draw_idx
    assert tokenizer.players[0].score == 24000


def test_pingju_does_not_deduct_riichi_stick_when_closing_ron_window() -> None:
    def fake_reaction(
        _self: TenhouTokenizer,
        discarder: int,
        tile_idx: int,
    ) -> engine.ReactionDecision:
        return engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        )

    tokenizer = TenhouTokenizer()
    tokenizer._compute_reaction_options = fake_reaction.__get__(tokenizer, TenhouTokenizer)
    tokenizer._compute_self_options = lambda *_args, **_kwargs: {"riichi"} if _args[1] == 0 else set()
    tokens = tokenizer.tokenize_game(
        minimal_game(
            [
                qipai_event(),
                {"zimo": {"l": 0, "p": "m1"}},
                {"dapai": {"l": 0, "p": "m1*"}},
                pingju_event(),
            ]
        )
    )

    pass_idx = tokens.index("pass_react_1_ron_forced_rule")
    draw_idx = tokens.index("pingju_ryukyoku")
    assert pass_idx < draw_idx
    assert tokenizer.players[0].score == 25000


def test_multiple_hule_preserves_event_order_until_reaction_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}, 2: {"ron"}},
            trigger="discard",
        ),
    )
    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"hule": {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]}},
            {"hule": {"l": 2, "baojia": 0, "fenpei": [0, 0, 1000, -1000]}},
        ]
    )

    tokenizer = TenhouTokenizer()
    tokens = tokenizer.tokenize_game(game)

    first_ron = tokens.index("ron_from_1_0")
    second_ron = tokens.index("ron_from_2_0")
    take_ron_1 = tokens.index("take_react_1_ron")
    take_ron_2 = tokens.index("take_react_2_ron")

    assert first_ron < second_ron < take_ron_1 < take_ron_2


def test_multiple_hule_emits_score_deltas_immediately_after_each_ron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}, 2: {"ron"}},
            trigger="discard",
        ),
    )
    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"hule": {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]}},
            {"hule": {"l": 2, "baojia": 0, "fenpei": [0, 0, 1000, -1000]}},
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)

    first_ron = tokens.index("ron_from_1_0")
    first_delta_0 = tokens.index("score_delta_0", first_ron)
    second_ron = tokens.index("ron_from_2_0")
    second_delta_0 = tokens.index("score_delta_0", second_ron)
    take_ron_1 = tokens.index("take_react_1_ron")

    assert first_ron < first_delta_0 < second_ron < second_delta_0 < take_ron_1


def test_multiple_hule_requires_same_baojia_for_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}, 2: {"ron"}},
            trigger="discard",
        ),
    )
    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"hule": {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]}},
            {"hule": {"l": 2, "baojia": 1, "fenpei": [0, 0, 1000, -1000]}},
        ]
    )

    with pytest.raises(engine.TokenizeError):
        TenhouTokenizer().tokenize_game(game)


def test_round_rejects_non_hule_event_after_ron_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        ),
    )
    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"hule": {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]}},
            pingju_event(),
        ]
    )

    with pytest.raises(engine.TokenizeError):
        TenhouTokenizer().tokenize_game(game)


def test_round_rejects_non_dict_hule_after_ron_without_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        ),
    )
    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"hule": "bad"},
        ]
    )

    with pytest.raises(engine.TokenizeError):
        TenhouTokenizer().tokenize_game(game)
