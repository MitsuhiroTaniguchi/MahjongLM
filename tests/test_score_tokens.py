from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import TenhouTokenizer, TokenizeError, encode_tenbo_tokens, tile_to_index
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event, qipai_payload
from tests.validation_helpers import validate_token_stream


def test_encode_tenbo_tokens_decomposes_by_stick_units() -> None:
    assert encode_tenbo_tokens(25000) == [
        "TENBO_PLUS",
        "TENBO_20000",
        "TENBO_5000",
    ]
    assert encode_tenbo_tokens(-3900) == [
        "TENBO_MINUS",
        "TENBO_3000",
        "TENBO_500",
        "TENBO_300",
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
    assert tokenizer.tokens[i : i + 4] == [
        "score_0",
        "TENBO_PLUS",
        "TENBO_20000",
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

    assert tokenizer.tokens[honba_idx : honba_idx + 3] == [
        "honba",
        "TENBO_PLUS",
        "TENBO_200",
    ]
    assert tokenizer.tokens[riichi_idx : riichi_idx + 3] == [
        "riichi_sticks",
        "TENBO_PLUS",
        "TENBO_3000",
    ]


def test_result_emits_score_delta_as_tenbo_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})
    tokenizer._on_hule({"l": 0, "fenpei": [-3900, 3900, 0, 0]})

    i0 = tokenizer.tokens.index("score_delta_0")
    assert tokenizer.tokens[i0 + 1 : i0 + 6] == [
        "TENBO_MINUS",
        "TENBO_3000",
        "TENBO_500",
        "TENBO_300",
        "TENBO_100",
    ]

    i2 = tokenizer.tokens.index("score_delta_2")
    assert tokenizer.tokens[i2 + 1] == "TENBO_ZERO"


def test_result_emits_round_rank_tokens_after_score_deltas() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})
    tokenizer._on_hule({"l": 0, "fenpei": [-3900, 3900, 0, 0]})

    delta_idx = tokenizer.tokens.index("score_delta_3")
    rank_tokens = tokenizer.tokens[delta_idx + 2 : delta_idx + 6]
    assert rank_tokens == ["rank_0_4", "rank_1_1", "rank_2_2", "rank_3_3"]


def test_hule_emits_ron_then_score_deltas_in_seat_order() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={2: {"ron"}},
    )
    tokenizer._on_hule({"l": 2, "baojia": 0, "fenpei": [-1000, 2000, -500, -500]})

    take_idx = tokenizer.tokens.index("take_react_2_ron")
    delta_positions = [tokenizer.tokens.index(f"score_delta_{seat}") for seat in range(4)]
    assert take_idx < delta_positions[0] < delta_positions[1] < delta_positions[2] < delta_positions[3]


def test_hule_closes_competing_reactions_before_ron_result_tokens() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("p7"),
        options_by_player={0: {"ron"}, 2: {"chi"}},
    )

    tokenizer._on_hule({"l": 0, "baojia": 1, "fenpei": [15600, -12500, 0, 0]})

    pass_idx = tokenizer.tokens.index("pass_react_2_chi_forced_priority")
    delta_idx = tokenizer.tokens.index("score_delta_0")
    take_idx = tokenizer.tokens.index("take_react_0_ron")
    assert take_idx < pass_idx < delta_idx


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


def test_pingju_emits_round_rank_tokens_after_score_deltas() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer._on_pingju({"name": "流局", "fenpei": [1000, -1000, 0, 0]})

    delta_idx = tokenizer.tokens.index("score_delta_3")
    rank_tokens = tokenizer.tokens[delta_idx + 2 : delta_idx + 6]
    assert rank_tokens == ["rank_0_1", "rank_1_4", "rank_2_2", "rank_3_3"]


def test_tokenize_game_emits_final_scores_and_final_ranks_before_game_end() -> None:
    tokens = TenhouTokenizer().tokenize_game(
        {
            "qijia": 0,
            "defen": [25000, 25000, 25000, 25000],
            "rank": [1, 2, 3, 4],
            "log": [[qipai_event(), pingju_event()]],
        }
    )

    final_score_idx = tokens.index("final_score_0")
    final_rank_idx = tokens.index("final_rank_0_1")
    assert tokens[final_score_idx : final_score_idx + 3] == [
        "final_score_0",
        "TENBO_PLUS",
        "TENBO_20000",
    ]
    assert final_score_idx < final_rank_idx < len(tokens) - 1
    assert tokens[final_rank_idx : final_rank_idx + 4] == [
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "final_rank_3_4",
    ]
    assert tokens[-1] == "game_end"


def test_tokenize_game_uses_top_level_final_defen_for_final_score_block() -> None:
    tokens = TenhouTokenizer().tokenize_game(
        {
            "qijia": 0,
            "defen": [26000, 24000, 25000, 25000],
            "rank": [1, 4, 2, 3],
            "log": [[qipai_event(), pingju_event()]],
        }
    )
    final_score_idx = tokens.index("final_score_0")
    assert tokens[final_score_idx : final_score_idx + 4] == [
        "final_score_0",
        "TENBO_PLUS",
        "TENBO_20000",
        "TENBO_5000",
    ]
    assert "TENBO_1000" in tokens[final_score_idx : tokens.index("final_score_1")]


def test_tokenize_game_omits_final_suffix_without_top_level_defen() -> None:
    tokens = TenhouTokenizer().tokenize_game({"log": [[qipai_event(), pingju_event()]]})
    assert "final_score_0" not in tokens
    assert "final_rank_0_1" not in tokens
    assert tokens[-1] == "game_end"


def test_tokenize_game_requires_qijia_for_tied_final_rank_reconstruction() -> None:
    with pytest.raises(TokenizeError, match="game.qijia"):
        TenhouTokenizer().tokenize_game(
            {
                "defen": [25000, 25000, 24000, 26000],
                "rank": [2, 3, 4, 1],
                "log": [[qipai_event(), pingju_event()]],
            }
        )


def test_tokenize_game_validates_top_level_final_rank() -> None:
    with pytest.raises(TokenizeError, match="game.rank"):
        TenhouTokenizer().tokenize_game(
            {
                "qijia": 0,
                "defen": [25000, 25000, 25000, 25000],
                "rank": [4, 3, 2, 1],
                "log": [[qipai_event(), pingju_event()]],
            }
        )


def test_round_rank_tie_break_uses_rotated_seat_order() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.initial_qijia = 1
    tokenizer._on_qipai(qipai_payload(jushu=2))
    tokenizer._on_pingju({"name": "流局", "fenpei": [0, 0, 0, 0]})

    rank_tokens = tokenizer.tokens[-4:]
    assert rank_tokens == ["rank_0_3", "rank_1_4", "rank_2_1", "rank_3_2"]


def test_multi_ron_emits_round_rank_tokens_only_after_last_hule() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}, 2: {"ron"}},
        trigger="discard",
    )

    tokenizer._on_hule(
        {"l": 1, "baojia": 0, "fenpei": [0, -1000, 1000, 0]},
        more_ron_expected=True,
        remaining_ron_winners={1, 2},
    )
    assert not any(token.startswith("rank_") for token in tokenizer.tokens)

    tokenizer._on_hule(
        {"l": 2, "baojia": 0, "fenpei": [0, 0, 1000, -1000]},
        more_ron_expected=False,
        remaining_ron_winners={2},
    )
    assert tokenizer.tokens[-4:] == ["rank_0_2", "rank_1_3", "rank_2_1", "rank_3_4"]


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

    pass_idx = tokens.index("pass_react_1_ron_voluntary")
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


def test_pingju_deducts_riichi_stick_when_closing_ron_window_without_win() -> None:
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

    pass_idx = tokens.index("pass_react_1_ron_voluntary")
    draw_idx = tokens.index("pingju_ryukyoku")
    assert pass_idx < draw_idx
    assert tokenizer.players[0].score == 24000


def test_multiple_hule_emits_take_before_each_ron(
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

    take_ron_1 = tokens.index("take_react_1_ron")
    take_ron_2 = tokens.index("take_react_2_ron")
    first_delta_0 = tokens.index("score_delta_0", take_ron_1)
    second_delta_0 = tokens.index("score_delta_0", take_ron_2)

    assert take_ron_1 < first_delta_0 < take_ron_2 < second_delta_0


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

    take_ron_1 = tokens.index("take_react_1_ron")
    first_delta_0 = tokens.index("score_delta_0", take_ron_1)
    take_ron_2 = tokens.index("take_react_2_ron")
    second_delta_0 = tokens.index("score_delta_0", take_ron_2)
    take_ron_1 = tokens.index("take_react_1_ron")

    assert take_ron_1 < first_delta_0 < take_ron_2 < second_delta_0


def test_multiple_hule_emits_declined_ron_pass_before_first_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}, 2: {"ron"}, 3: {"ron"}},
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

    pass_idx = tokens.index("pass_react_3_ron_voluntary")
    second_take = tokens.index("take_react_2_ron")
    first_take = tokens.index("take_react_1_ron")
    assert first_take < pass_idx < second_take
    validate_token_stream(tokens)


def test_multiple_hule_delays_later_winner_non_ron_pass_until_their_ron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: engine.ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}, 2: {"pon", "ron"}, 3: {"ron"}},
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

    take_ron_2 = tokens.index("take_react_2_ron")
    pass_pon_2 = tokens.index("pass_react_2_pon_voluntary")
    first_delta_0 = tokens.index("score_delta_0", tokens.index("take_react_1_ron"))
    second_delta_0 = tokens.index("score_delta_0", take_ron_2)
    assert first_delta_0 < take_ron_2 < pass_pon_2 < second_delta_0
    assert "pass_react_2_pon_forced_priority" not in tokens


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


def test_validate_token_stream_rejects_score_block_without_take_ron() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "opt_react_0_ron",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_pass_react_after_result_start() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "opt_react_0_ron",
                "opt_react_2_chi",
                "take_react_0_ron",
                "score_delta_0",
                "TENBO_ZERO",
                "pass_react_2_chi_forced_priority",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_draw_between_take_ron_and_ron_from() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "opt_react_1_ron",
                "take_react_1_ron",
                "draw_0_m1",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_duplicate_score_block_without_new_take() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "opt_react_1_ron",
                "take_react_1_ron",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_mid_round_score_delta_block() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "draw_0_m1",
                "score_delta_0",
                "TENBO_ZERO",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_mid_round_haipai_block() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "discard_0_m1",
                "haipai_0",
                *["m1"] * 13,
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_mid_round_bakaze_token() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "draw_0_m1",
                "bakaze_1",
                "pingju_ryukyoku",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_mid_round_dora_block() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "draw_0_m1",
                "dora",
                "m2",
                "pingju_ryukyoku",
                "score_delta_0",
                "TENBO_ZERO",
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_next_round_before_result_completes() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "pingju_ryukyoku",
                "score_delta_0",
                "TENBO_ZERO",
                "round_start",
                "game_end",
            ]
        )


def test_validate_token_stream_rejects_game_end_before_round_result() -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(
            [
                "game_start",
                "round_start",
                "bakaze_0",
                "kyoku_0",
                "honba",
                "TENBO_ZERO",
                "riichi_sticks",
                "TENBO_ZERO",
                "dora",
                "m1",
                "score_0",
                "TENBO_ZERO",
                "score_1",
                "TENBO_ZERO",
                "score_2",
                "TENBO_ZERO",
                "score_3",
                "TENBO_ZERO",
                "haipai_0",
                *["m1"] * 13,
                "haipai_1",
                *["m1"] * 13,
                "haipai_2",
                *["m1"] * 13,
                "haipai_3",
                *["m1"] * 13,
                "draw_0_m1",
                "game_end",
            ]
        )


@pytest.mark.parametrize(
    "tokens",
    [
        [
            "game_start",
            "round_start",
            "bakaze_0",
            "kyoku_0",
            "honba",
            "TENBO_ZERO",
            "riichi_sticks",
            "TENBO_ZERO",
            "dora",
            "m1",
            "score_0",
            "TENBO_ZERO",
            "score_1",
            "TENBO_ZERO",
            "score_2",
            "TENBO_ZERO",
            "score_3",
            "TENBO_ZERO",
            "haipai_0",
            *["m1"] * 13,
            "haipai_1",
            *["m1"] * 13,
            "haipai_2",
            *["m1"] * 13,
            "haipai_3",
            *["m1"] * 13,
            "pingju_ryukyoku",
            "score_delta_1",
            "TENBO_ZERO",
            "game_end",
        ],
        [
            "game_start",
            "round_start",
            "bakaze_0",
            "kyoku_0",
            "honba",
            "TENBO_ZERO",
            "riichi_sticks",
            "TENBO_ZERO",
            "dora",
            "m1",
            "score_0",
            "TENBO_ZERO",
            "score_1",
            "TENBO_ZERO",
            "score_2",
            "TENBO_ZERO",
            "score_3",
            "TENBO_ZERO",
            "haipai_0",
            *["m1"] * 13,
            "haipai_1",
            *["m1"] * 13,
            "haipai_2",
            *["m1"] * 13,
            "haipai_3",
            *["m1"] * 13,
            "pingju_ryukyoku",
            "score_delta_0",
            "TENBO_ZERO",
            "score_delta_2",
            "TENBO_ZERO",
            "game_end",
        ],
    ],
)
def test_validate_token_stream_rejects_malformed_score_delta_blocks(tokens: list[str]) -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(tokens)


@pytest.mark.parametrize(
    "tokens",
    [
        ["game_start", "round_start", "dora", "round_start", "game_end"],
        ["game_start", "round_start", "score_0", "round_start", "game_end"],
        ["game_start", "round_start", "haipai_0", "m1", "round_start", "game_end"],
        ["game_start", "round_start", "discard_0_m1", "round_start", "tedashi", "game_end"],
        ["game_start", "round_start", "take_react_2_chi", "game_end"],
        ["game_start", "round_start", "opt_self_0_ankan", "round_start", "game_end"],
        ["game_start", "round_start", "take_self_0_kakan", "game_end"],
    ],
)
def test_validate_token_stream_rejects_malformed_payload_sequences(tokens: list[str]) -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(tokens)


@pytest.mark.parametrize(
    "tokens",
    [
        ["game_start", "bakaze_0", "game_end"],
        ["game_start", "round_start", "kyoku_0", "game_end"],
        ["game_start", "round_start", "bakaze_0", "honba", "TENBO_ZERO", "game_end"],
        [
            "game_start",
            "round_start",
            "bakaze_0",
            "kyoku_0",
            "honba",
            "TENBO_ZERO",
            "riichi_sticks",
            "TENBO_ZERO",
            "dora",
            "m1",
            "score_1",
            "TENBO_ZERO",
            "game_end",
        ],
        [
            "game_start",
            "round_start",
            "bakaze_0",
            "kyoku_0",
            "honba",
            "TENBO_ZERO",
            "riichi_sticks",
            "TENBO_ZERO",
            "dora",
            "m1",
            "score_0",
            "TENBO_ZERO",
            "score_1",
            "TENBO_ZERO",
            "score_2",
            "TENBO_ZERO",
            "score_3",
            "TENBO_ZERO",
            "haipai_1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "m1",
            "game_end",
        ],
    ],
)
def test_validate_token_stream_rejects_malformed_round_prelude(tokens: list[str]) -> None:
    with pytest.raises(AssertionError):
        validate_token_stream(tokens)


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
