from __future__ import annotations

from pathlib import Path

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


def test_tokenize_game_emits_rule_tokens_before_game_start() -> None:
    tokens = TenhouTokenizer().tokenize_game(
        {
            "title": "四鳳東喰赤速",
            "log": [[qipai_event(), pingju_event()]],
        }
    )
    assert tokens[:4] == ["rule_player_4", "rule_length_tonpu", "game_start", "round_start"]


def test_tokenize_game_emits_three_player_rule_token_from_title() -> None:
    tokens = TenhouTokenizer().tokenize_game(
        {
            "title": "三鳳南喰赤",
            "log": [[qipai_event(seat_count=3), pingju_event(seat_count=3)]],
        }
    )
    assert tokens[:4] == ["rule_player_3", "rule_length_hanchan", "game_start", "round_start"]


def test_tokenize_game_emits_inferred_three_player_rule_token_without_title() -> None:
    tokens = TenhouTokenizer().tokenize_game(
        {
            "log": [[qipai_event(seat_count=3), pingju_event(seat_count=3)]],
        }
    )
    assert tokens[:3] == ["rule_player_3", "game_start", "round_start"]


def test_sanma_multi_player_simulation_is_enabled_when_three_player_api_exists() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3

    assert tokenizer._use_multi_player_simulation() is (
        engine.PM_STATELESS_SIMULATION_API_AVAILABLE and engine.PM_THREE_PLAYER_API_AVAILABLE
    )


def test_sanma_simulation_stays_disabled_when_only_shoupai_helpers_exist() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3

    original_stateless = engine.PM_STATELESS_SIMULATION_API_AVAILABLE
    original_sim = engine.PM_SIMULATION_API_AVAILABLE
    original_shoupai = engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE
    original_three_player = engine.PM_THREE_PLAYER_API_AVAILABLE
    engine.PM_STATELESS_SIMULATION_API_AVAILABLE = False
    engine.PM_SIMULATION_API_AVAILABLE = True
    engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE = True
    engine.PM_THREE_PLAYER_API_AVAILABLE = True
    try:
        assert tokenizer._use_multi_player_simulation() is False
    finally:
        engine.PM_STATELESS_SIMULATION_API_AVAILABLE = original_stateless
        engine.PM_SIMULATION_API_AVAILABLE = original_sim
        engine.PM_SHOUPAI_SIMULATION_API_AVAILABLE = original_shoupai
        engine.PM_THREE_PLAYER_API_AVAILABLE = original_three_player


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


def test_hule_emits_yaku_summary_tokens_before_score_deltas() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [
                {"name": "立直", "fanshu": 1},
                {"name": "門前清自摸和", "fanshu": 1},
                {"name": "ドラ", "fanshu": 2},
            ],
            "fanshu": 4,
            "fu": 30,
        }
    )

    tsumo_idx = tokenizer.tokens.index("take_self_0_tsumo")
    yaku_idx = tokenizer.tokens.index("yaku_riichi")
    han_idx = tokenizer.tokens.index("han_4")
    fu_idx = tokenizer.tokens.index("fu_30")
    delta_idx = tokenizer.tokens.index("score_delta_0")
    assert tsumo_idx < yaku_idx < han_idx < fu_idx < delta_idx
    assert "yaku_menzen_tsumo" in tokenizer.tokens
    assert tokenizer.tokens.count("yaku_dora") == 2


def test_hule_repeats_each_dora_family_token_by_fanshu() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [
                {"name": "立直", "fanshu": 1},
                {"name": "ドラ", "fanshu": 2},
                {"name": "赤ドラ", "fanshu": 2},
                {"name": "裏ドラ", "fanshu": 3},
            ],
            "fanshu": 8,
            "fu": 30,
        }
    )

    assert tokenizer.tokens.count("yaku_riichi") == 1
    assert tokenizer.tokens.count("yaku_dora") == 2
    assert tokenizer.tokens.count("yaku_aka_dora") == 2
    assert tokenizer.tokens.count("yaku_ura_dora") == 3


def test_hule_emits_ura_dora_reveal_tiles_for_riichi_win() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [
                {"name": "立直", "fanshu": 1},
                {"name": "ドラ", "fanshu": 1},
                {"name": "裏ドラ", "fanshu": 2},
            ],
            "fubaopai": ["m1", "p9"],
            "fanshu": 4,
            "fu": 30,
        }
    )

    ura_idx = tokenizer.tokens.index("ura_dora")
    assert tokenizer.tokens[ura_idx : ura_idx + 3] == ["ura_dora", "m1", "p9"]
    assert tokenizer.tokens.count("yaku_ura_dora") == 2


def test_hule_emits_opened_winning_hand_with_winning_tile_at_end_on_tsumo() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [{"name": "立直", "fanshu": 1}],
            "shoupai": "p40888s34099z333p3,b,z111=,s9999-",
            "fanshu": 1,
            "fu": 30,
        }
    )

    opened_idx = tokenizer.tokens.index("opened_hand_0")
    assert tokenizer.tokens[opened_idx : opened_idx + 15] == [
        "opened_hand_0",
        "p4",
        "p0",
        "p8",
        "p8",
        "p8",
        "s3",
        "s4",
        "s0",
        "s9",
        "s9",
        "z3",
        "z3",
        "z3",
        "p3",
    ]
    assert "z1" not in tokenizer.tokens[opened_idx : opened_idx + 20]


def test_hule_emits_opened_ron_hand_without_winning_tile() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
    )

    tokenizer._on_hule(
        {
            "l": 1,
            "baojia": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [{"name": "立直", "fanshu": 1}],
            "shoupai": "p1123445678p3,z666-",
            "fanshu": 1,
            "fu": 30,
        }
    )

    opened_idx = tokenizer.tokens.index("opened_hand_1")
    assert tokenizer.tokens[opened_idx : opened_idx + 11] == [
        "opened_hand_1",
        "p1",
        "p1",
        "p2",
        "p3",
        "p4",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
    ]


def test_hule_does_not_emit_ura_dora_reveal_without_riichi() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [
                {"name": "門前清自摸和", "fanshu": 1},
                {"name": "裏ドラ", "fanshu": 2},
            ],
            "fubaopai": ["m1", "p9"],
            "fanshu": 3,
            "fu": 30,
        }
    )

    assert "ura_dora" not in tokenizer.tokens
    assert tokenizer.tokens.count("yaku_ura_dora") == 2


def test_pingju_emits_opened_hands_for_non_empty_shoupai_entries_only() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer._on_pingju(
        {
            "name": "流局",
            "fenpei": [1000, -1000, 0, 0],
            "shoupai": ["m123p456s789z11,p777-", "", "p4067z123,b", ""],
        }
    )

    first_idx = tokenizer.tokens.index("opened_hand_0")
    second_idx = tokenizer.tokens.index("opened_hand_2")
    assert first_idx < second_idx
    assert tokenizer.tokens[first_idx : first_idx + 12] == [
        "opened_hand_0",
        "m1",
        "m2",
        "m3",
        "p4",
        "p5",
        "p6",
        "s7",
        "s8",
        "s9",
        "z1",
        "z1",
    ]
    assert tokenizer.tokens[second_idx : second_idx + 8] == [
        "opened_hand_2",
        "p4",
        "p0",
        "p6",
        "p7",
        "z1",
        "z2",
        "z3",
    ]
    assert "opened_hand_1" not in tokenizer.tokens
    assert "opened_hand_3" not in tokenizer.tokens


def test_pingju_accepts_false_for_hidden_shoupai_entries() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_event(seat_count=3)["qipai"])

    tokenizer._on_pingju(
        {
            "name": "流局",
            "fenpei": [1000, -2000, 1000],
            "shoupai": ["p12334056789s77,b", False, "p3466789s456z555,b,b"],
        }
    )

    assert "pingju_ryukyoku" in tokenizer.tokens
    assert "opened_hand_0" in tokenizer.tokens
    assert "opened_hand_2" in tokenizer.tokens
    assert "opened_hand_1" not in tokenizer.tokens


def test_nagashimangan_does_not_emit_opened_hands() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer._on_pingju(
        {
            "name": "流し満貫",
            "fenpei": [4000, -2000, -1000, -1000],
            "shoupai": ["m123p456s789z11", "", "", ""],
        }
    )

    assert "pingju_nagashimangan" in tokenizer.tokens
    assert all(not token.startswith("opened_hand_") for token in tokenizer.tokens)


def test_suufonrenda_does_not_emit_opened_hands() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer._on_pingju(
        {
            "name": "四風連打",
            "fenpei": [0, 0, 0, 0],
            "shoupai": ["m123", "p456", "", ""],
        }
    )

    assert "pingju_sufurenda" in tokenizer.tokens
    assert all(not token.startswith("opened_hand_") for token in tokenizer.tokens)


def test_sukantsu_does_not_emit_opened_hands() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer._on_pingju(
        {
            "name": "四槓散了",
            "fenpei": [0, 0, 0, 0],
            "shoupai": ["m123", "", "s789", ""],
        }
    )

    assert "pingju_sukantsu" in tokenizer.tokens
    assert all(not token.startswith("opened_hand_") for token in tokenizer.tokens)


def test_suuchariichi_still_emits_opened_hands() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer._on_pingju(
        {
            "name": "四家立直",
            "fenpei": [0, 0, 0, 0],
            "shoupai": [
                "p22245679s56777",
                "m333340567s3367",
                "m44p40678s340,s8888",
                "p1233388999s123",
            ],
        }
    )

    assert "pingju_suuchariichi" in tokenizer.tokens
    assert tokenizer.tokens.count("opened_hand_0") == 1
    assert tokenizer.tokens.count("opened_hand_1") == 1
    assert tokenizer.tokens.count("opened_hand_2") == 1
    assert tokenizer.tokens.count("opened_hand_3") == 1


def test_kyushukyuhai_emits_opened_hand_when_shoupai_is_present() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=1, options={"kyushukyuhai"})

    tokenizer._on_pingju(
        {
            "name": "九種九牌",
            "fenpei": [0, 0, 0, 0],
            "shoupai": ["", "m19p19s19z1234", "", ""],
        }
    )

    opened_idx = tokenizer.tokens.index("opened_hand_1")
    assert tokenizer.tokens[opened_idx : opened_idx + 11] == [
        "opened_hand_1",
        "m1",
        "m9",
        "p1",
        "p9",
        "s1",
        "s9",
        "z1",
        "z2",
        "z3",
        "z4",
    ]


def test_kyushukyuhai_opened_hand_uses_only_declaring_actor_on_converter_variant() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=2, options={"kyushukyuhai"})

    tokenizer._on_pingju(
        {
            "name": "九種九牌",
            "fenpei": [0, 0, 0, 0],
            "shoupai": ["m1", "", "m19p19s19z1234", ""],
        }
    )

    assert "opened_hand_0" not in tokenizer.tokens
    assert "opened_hand_1" not in tokenizer.tokens
    assert tokenizer.tokens.count("opened_hand_2") == 1
    assert "opened_hand_3" not in tokenizer.tokens


def test_hule_emits_yakuman_summary_token() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
    )

    tokenizer._on_hule(
        {
            "l": 1,
            "baojia": 0,
            "fenpei": [-16000, 16000, 0, 0],
            "hupai": [{"name": "国士無双", "fanshu": "*"}],
            "damanguan": 1,
        }
    )

    assert "yaku_kokushi_musou" in tokenizer.tokens
    assert "yakuman_1" in tokenizer.tokens
    assert "han_13" not in tokenizer.tokens
    assert "fu_30" not in tokenizer.tokens


def test_hule_clamps_13_plus_han_to_han_13() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-16000, 16000, 0, 0],
            "hupai": [{"name": "ドラ", "fanshu": 13}],
            "fanshu": 13,
            "fu": 30,
        }
    )

    assert "han_13" in tokenizer.tokens
    assert "yakuman_1" not in tokenizer.tokens
    assert "fu_30" in tokenizer.tokens
    assert "han_15" not in tokenizer.tokens


def test_hule_rejects_unknown_yaku_name() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    with pytest.raises(TokenizeError, match="unknown hule.hupai name"):
        tokenizer._on_hule(
            {
                "l": 0,
                "fenpei": [-3900, 3900, 0, 0],
                "hupai": [{"name": "未知役", "fanshu": 1}],
                "fanshu": 1,
                "fu": 30,
            }
        )


def test_three_player_qipai_initializes_sanma_live_draw_count() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))

    assert tokenizer.live_draws_left == 55


def test_hule_skips_zero_han_hupai_entries() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [
                {"name": "立直", "fanshu": 1},
                {"name": "裏ドラ", "fanshu": 0},
            ],
            "fanshu": 1,
            "fu": 30,
        }
    )

    assert "yaku_riichi" in tokenizer.tokens
    assert "yaku_ura_dora" not in tokenizer.tokens
    assert "ura_dora" not in tokenizer.tokens


def test_hule_skips_blank_converter_placeholder_name() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    tokenizer._on_hule(
        {
            "l": 0,
            "fenpei": [-3900, 3900, 0, 0],
            "hupai": [
                {"name": "", "fanshu": 1},
                {"name": "立直", "fanshu": 1},
            ],
            "fanshu": 1,
            "fu": 30,
        }
    )

    assert "yaku_riichi" in tokenizer.tokens


def test_three_player_self_options_respect_simulated_penuki_mask() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer.players = [engine.PlayerState(concealed=[0] * 34, score=25000) for _ in range(3)]
    tokenizer.players[0].concealed[tile_to_index("z4")] = 1
    tokenizer.live_draws_left = 10

    original_simulation = engine.PM_SIMULATION_API_AVAILABLE
    original_three_player = engine.PM_THREE_PLAYER_API_AVAILABLE
    original_supports = engine.PM_STATELESS_SIMULATION_API_AVAILABLE
    original_pm = engine.pm
    engine.PM_SIMULATION_API_AVAILABLE = True
    engine.PM_THREE_PLAYER_API_AVAILABLE = True
    engine.PM_STATELESS_SIMULATION_API_AVAILABLE = True

    class _FakePm:
        @staticmethod
        def compute_self_option_mask(*args):
            return 0

    engine.pm = _FakePm()
    try:
        options = tokenizer._compute_self_options(actor=0, drawn_tile=tile_to_index("m1"), is_gangzimo=False)
    finally:
        engine.pm = original_pm
        engine.PM_SIMULATION_API_AVAILABLE = original_simulation
        engine.PM_THREE_PLAYER_API_AVAILABLE = original_three_player
        engine.PM_STATELESS_SIMULATION_API_AVAILABLE = original_supports

    assert "penuki" not in options


def test_three_player_fallback_self_options_offer_penuki() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer.players = [engine.PlayerState(concealed=[0] * 34, score=25000) for _ in range(3)]
    tokenizer.players[0].concealed[tile_to_index("z4")] = 1
    tokenizer.live_draws_left = 10

    original_simulation = engine.PM_SIMULATION_API_AVAILABLE
    original_three_player = engine.PM_THREE_PLAYER_API_AVAILABLE
    engine.PM_SIMULATION_API_AVAILABLE = False
    engine.PM_THREE_PLAYER_API_AVAILABLE = False
    try:
        options = tokenizer._compute_self_options(actor=0, drawn_tile=tile_to_index("m1"), is_gangzimo=False)
    finally:
        engine.PM_SIMULATION_API_AVAILABLE = original_simulation
        engine.PM_THREE_PLAYER_API_AVAILABLE = original_three_player

    assert "penuki" in options


def test_penuki_marks_first_turn_open_call_seen() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    tokenizer._on_draw({"l": 0, "p": "z4"}, is_gangzimo=False)

    assert tokenizer.first_turn_open_calls_seen is False

    tokenizer._on_penuki({"l": 0, "p": "z4"})

    assert tokenizer.first_turn_open_calls_seen is True


def test_penuki_opens_ron_reaction_window_before_replacement_draw() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    tokenizer._on_draw({"l": 0, "p": "z4"}, is_gangzimo=False)

    reaction = engine.ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("z4"),
        options_by_player={1: {"ron"}},
        trigger="penuki",
    )
    tokenizer._compute_penuki_reaction_options = lambda actor, tile_idx: reaction  # type: ignore[method-assign]

    tokenizer._on_penuki({"l": 0, "p": "z4"})

    assert tokenizer.pending_reaction is reaction
    assert tokenizer.expected_draw_actor is None
    assert tokenizer.pending_dead_wall_draw is True
    assert "opt_react_1_ron" in tokenizer.tokens


def test_riichi_penuki_does_not_mark_persistent_furiten_when_tsumo_is_also_offered() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    tokenizer.players[0].is_riichi = True
    tokenizer.pending_self = engine.SelfDecision(
        actor=0,
        options={"tsumo", "penuki"},
        option_tiles={"penuki": ["z4"]},
    )

    tokenizer._finalize_self({"penuki"}, actor=0, chosen_tiles={"penuki": "z4"})

    assert tokenizer.players[0].riichi_furiten is False
    assert "take_self_0_penuki" in tokenizer.tokens
    assert "pass_self_0_tsumo" in tokenizer.tokens


def test_vocab_includes_penuki_self_action_tokens() -> None:
    vocab = (Path(__file__).resolve().parents[1] / "tokenizer" / "vocab.txt").read_text(encoding="utf-8")

    for seat in range(4):
        assert f"opt_self_{seat}_penuki" in vocab
        assert f"take_self_{seat}_penuki" in vocab
        assert f"pass_self_{seat}_penuki" in vocab


def test_hule_accepts_140_fu_non_yakuman_ron_with_actual_hand_and_yaku() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload(jushu=0))
    tokenizer.pending_reaction = engine.ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("p8"),
        options_by_player={0: {"ron"}},
    )

    tokenizer._on_hule(
        {
            "l": 0,
            "baojia": 1,
            "fenpei": [18000, -18000, 0, 0],
            "hupai": [
                {"name": "三槓子", "fanshu": 2},
                {"name": "三暗刻", "fanshu": 2},
                {"name": "役牌 中", "fanshu": 1},
                {"name": "混全帯幺九", "fanshu": 2},
            ],
            "fanshu": 7,
            "fu": 140,
            "shoupai": "p79z11p8,m9999,z7777,p1111",
        }
    )
    assert "take_react_0_ron" in tokenizer.tokens
    assert "fu_140" in tokenizer.tokens


def test_hule_rejects_fu_above_standard_non_yakuman_maximum() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"tsumo"})

    with pytest.raises(TokenizeError, match="hule.fu must be 25 or a multiple of 10 between 20 and 140"):
        tokenizer._on_hule(
            {
                "l": 0,
                "fenpei": [-3900, 3900, 0, 0],
                "hupai": [{"name": "立直", "fanshu": 1}],
                "fanshu": 1,
                "fu": 150,
            }
        )


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


def test_tokenize_game_emits_final_scores_and_final_ranks_before_round_end_and_game_end() -> None:
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
    round_end_idx = tokens.index("round_end")
    assert final_score_idx < final_rank_idx < round_end_idx < len(tokens) - 1
    assert tokens[final_rank_idx : final_rank_idx + 4] == [
        "final_rank_0_1",
        "final_rank_1_2",
        "final_rank_2_3",
        "final_rank_3_4",
    ]
    assert tokens[-2:] == ["round_end", "game_end"]


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
    assert tokens[-2:] == ["round_end", "game_end"]


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
                "discard_0_m1_tedashi",
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
                "score_delta_1",
                "TENBO_ZERO",
                "score_delta_2",
                "TENBO_ZERO",
                "score_delta_3",
                "TENBO_ZERO",
                "rank_0_1",
                "rank_1_2",
                "rank_2_3",
                "rank_3_4",
                "round_end",
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
        ["game_start", "rule_player_4", "dora", "round_start", "game_end"],
        ["game_start", "round_start", "dora", "round_start", "game_end"],
        ["game_start", "round_start", "score_0", "round_start", "game_end"],
        ["game_start", "round_start", "haipai_0", "m1", "round_start", "game_end"],
        ["game_start", "round_start", "discard_0_m1_tedashi", "round_start", "game_end"],
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
