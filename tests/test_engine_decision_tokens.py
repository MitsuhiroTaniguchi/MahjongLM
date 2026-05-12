from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import (
    ReactionDecision,
    SelfDecision,
    TenhouTokenizer,
    parse_hand_counts,
    tile_to_index,
)
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event, qipai_payload


@pytest.fixture(autouse=True)
def disable_pm_simulation_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)


def test_permanent_furiten_blocks_ron_for_all_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    offered_tile = tile_to_index("m2")
    discarded_wait = tile_to_index("m1")

    tokenizer.players[1].furiten_tiles.add(discarded_wait)
    monkeypatch.setattr(
        engine,
        "_pm_wait_mask",
        lambda *_args, **_kwargs: (1 << discarded_wait) | (1 << offered_tile),
    )
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: True)

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered_tile)

    assert reaction is not None
    assert "ron" not in reaction.options_by_player.get(1, set())


def test_hule_with_baojia_is_classified_as_ron() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={2: {"ron"}},
    )

    tokenizer._on_hule({"l": 2, "baojia": 0, "fenpei": [0, 0, 0, 0]})
    tokenizer._flush_pending()

    assert "take_react_2_ron" in tokenizer.tokens
    assert "win_ron_2_from_0" not in tokenizer.tokens
    assert "win_tsumo_2" not in tokenizer.tokens


def test_kaigang_does_not_force_self_pass_before_tsumo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_self_options",
        lambda _self, _actor, _tile_idx, is_gangzimo=False: {"tsumo"} if is_gangzimo else {"ankan"},
    )

    game = minimal_game(
        [
            qipai_event(hands=["m111p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "m1"}},
            {"gang": {"l": 0, "m": "m1111"}},
            {"gangzimo": {"l": 0, "p": "m1"}},
            {"kaigang": {"baopai": "p5"}},
            {"hule": {"l": 0, "fenpei": [0, 0, 0, 0]}},
        ]
    )

    tokenizer = TenhouTokenizer()
    tokens = tokenizer.tokenize_game(game)

    assert "opt_self_0_tsumo" in tokens
    assert "take_self_0_tsumo" in tokens
    assert "pass_self_0_tsumo" not in tokens
    ankan_idx = tokens.index("take_self_0_ankan")
    dora_idx = tokens.index("dora", ankan_idx)
    gang_draw_idx = tokens.index("draw_0_m1", dora_idx)
    assert ankan_idx < dora_idx < gang_draw_idx
    assert "win_tsumo_0" not in tokens
    assert "score_delta_0" in tokens
    assert "TENBO_ZERO" in tokens


def test_riichi_take_does_not_emit_riichi_event_token(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"riichi"})
    monkeypatch.setattr(TenhouTokenizer, "_compute_reaction_options", lambda *_args, **_kwargs: None)

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1*"})

    assert "take_self_0_riichi" in tokenizer.tokens
    assert "riichi_0" not in tokenizer.tokens


def test_riichi_discard_ron_does_not_deduct_stick(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"riichi"})
    monkeypatch.setattr(
        TenhouTokenizer,
        "_compute_reaction_options",
        lambda _self, discarder, tile_idx: ReactionDecision(
            discarder=discarder,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="discard",
        ),
    )

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1*"})
    assert tokenizer.players[0].score == 25000

    tokenizer._on_hule({"l": 1, "baojia": 0, "fenpei": [0, 0, 0, 0]})
    tokenizer._flush_pending()
    assert tokenizer.players[0].score == 25000


def test_riichi_stick_is_deducted_when_no_ron_window(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"riichi"})
    monkeypatch.setattr(TenhouTokenizer, "_compute_reaction_options", lambda *_args, **_kwargs: None)

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1*"})

    assert tokenizer.players[0].score == 24000


def test_riichi_stick_is_deducted_when_ron_options_all_pass() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
        trigger="discard",
    )
    tokenizer.pending_riichi_actor = 0

    tokenizer._finalize_reaction(close_reason="voluntary")
    assert tokenizer.players[0].score == 24000


@pytest.mark.parametrize(
    ("chosen", "close_reason", "expected_score"),
    [
        ({1: "ron"}, "voluntary", 25000),
        ({}, "voluntary", 24000),
    ],
)
def test_riichi_stick_resolution_matrix(
    chosen: dict[int, str],
    close_reason: str,
    expected_score: int,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
        chosen=chosen,
        trigger="discard",
    )
    tokenizer.pending_riichi_actor = 0

    tokenizer._finalize_reaction(close_reason=close_reason)

    assert tokenizer.players[0].score == expected_score
    assert tokenizer.pending_riichi_actor is None


def test_kakan_generates_reaction_decision_and_rob_kan_take(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tile = tile_to_index("m1")
    actor = 0
    p = tokenizer.players[actor]
    p.concealed[tile] = max(p.concealed[tile], 1)
    p.open_pons[tile] = 1
    p.melds = [("pon", tile)]
    tokenizer.pending_self = SelfDecision(actor=actor, options={"kakan"})
    tokenizer.expected_discard_actor = actor

    def fake_kakan_reaction(self: TenhouTokenizer, actor: int, tile_idx: int) -> ReactionDecision:
        return ReactionDecision(
            discarder=actor,
            discard_tile=tile_idx,
            options_by_player={1: {"ron"}},
            trigger="kakan",
        )

    monkeypatch.setattr(TenhouTokenizer, "_compute_kakan_reaction_options", fake_kakan_reaction)

    tokenizer._on_gang({"l": actor, "m": "m1111+"})
    tokenizer._on_hule({"l": 1, "baojia": actor, "fenpei": [0, 0, 0, 0]})
    tokenizer._flush_pending()

    assert "take_self_0_kakan" in tokenizer.tokens
    assert "opt_react_1_ron" in tokenizer.tokens
    assert "take_react_1_ron" in tokenizer.tokens


def test_passing_ron_by_taking_other_call_does_not_set_temporary_furiten() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"chi", "ron"}},
        chosen={1: "chi"},
    )
    tokenizer._finalize_reaction()

    assert tokenizer.players[1].temporary_furiten is False
    assert "take_react_1_chi" in tokenizer.tokens
    assert "pass_react_1_ron_voluntary" in tokenizer.tokens


def test_pass_react_forced_priority_when_pon_blocks_chi() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer.pending_reaction = ReactionDecision(
        discarder=3,
        discard_tile=tile_to_index("m4"),
        options_by_player={0: {"chi"}, 1: {"pon"}},
        chosen={1: "pon"},
    )
    tokenizer._finalize_reaction()

    assert "take_react_1_pon" in tokenizer.tokens
    assert "pass_react_0_chi_forced_priority" in tokenizer.tokens


def test_forced_rule_close_is_rejected_for_reactions() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
    )
    with pytest.raises(engine.TokenizeError, match="forced_rule reaction close"):
        tokenizer._finalize_reaction(close_reason="forced_rule")


@pytest.mark.parametrize(
    (
        "options_by_player",
        "chosen",
        "riichi_seats",
        "close_reason",
        "expected_tokens",
        "temporary_furiten_seats",
        "riichi_furiten_seats",
    ),
    [
        (
            {0: {"chi"}, 1: {"pon"}},
            {1: "pon"},
            set(),
            "voluntary",
            {"take_react_1_pon", "pass_react_0_chi_forced_priority"},
            set(),
            set(),
        ),
        (
            {0: {"pon"}, 1: {"ron"}},
            {1: "ron"},
            set(),
            "voluntary",
            {"take_react_1_ron", "pass_react_0_pon_forced_priority"},
            set(),
            set(),
        ),
        (
            {0: {"minkan"}, 1: {"pon"}},
            {1: "pon"},
            set(),
            "voluntary",
            {"take_react_1_pon", "pass_react_0_minkan_forced_priority"},
            set(),
            set(),
        ),
        (
            {1: {"pon", "ron"}},
            {1: "pon"},
            set(),
            "voluntary",
            {"take_react_1_pon", "pass_react_1_ron_voluntary"},
            set(),
            set(),
        ),
        (
            {1: {"minkan", "ron"}},
            {1: "minkan"},
            set(),
            "voluntary",
            {"take_react_1_minkan", "pass_react_1_ron_voluntary"},
            set(),
            set(),
        ),
        (
            {0: {"pon"}, 1: {"pon"}},
            {1: "pon"},
            set(),
            "voluntary",
            {"take_react_1_pon", "pass_react_0_pon_forced_priority"},
            set(),
            set(),
        ),
        (
            {0: {"minkan"}, 1: {"minkan"}},
            {1: "minkan"},
            set(),
            "voluntary",
            {"take_react_1_minkan", "pass_react_0_minkan_forced_priority"},
            set(),
            set(),
        ),
        (
            {0: {"pon"}, 1: {"minkan"}},
            {1: "minkan"},
            set(),
            "voluntary",
            {"take_react_1_minkan", "pass_react_0_pon_forced_priority"},
            set(),
            set(),
        ),
        (
            {0: {"chi"}, 1: {"pon"}, 2: {"ron"}},
            {2: "ron"},
            set(),
            "voluntary",
            {"take_react_2_ron", "pass_react_0_chi_forced_priority", "pass_react_1_pon_forced_priority"},
            set(),
            set(),
        ),
        (
            {0: {"chi"}, 1: {"minkan"}, 2: {"pon"}},
            {1: "minkan"},
            set(),
            "voluntary",
            {"take_react_1_minkan", "pass_react_0_chi_forced_priority", "pass_react_2_pon_forced_priority"},
            set(),
            set(),
        ),
        (
            {1: {"ron"}},
            {},
            {1},
            "voluntary",
            {"pass_react_1_ron_voluntary"},
            set(),
            {1},
        ),
        (
            {1: {"chi"}},
            {},
            set(),
            "voluntary",
            {"pass_react_1_chi_voluntary"},
            set(),
            set(),
        ),
    ],
)
def test_reaction_resolution_matrix(
    options_by_player: dict[int, set[str]],
    chosen: dict[int, str],
    riichi_seats: set[int],
    close_reason: str,
    expected_tokens: set[str],
    temporary_furiten_seats: set[int],
    riichi_furiten_seats: set[int],
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    for seat in riichi_seats:
        tokenizer.players[seat].is_riichi = True

    tokenizer.pending_reaction = ReactionDecision(
        discarder=3,
        discard_tile=tile_to_index("m4"),
        options_by_player=options_by_player,
        chosen=chosen,
    )
    tokenizer._finalize_reaction(close_reason=close_reason)

    for token in expected_tokens:
        assert token in tokenizer.tokens
    for seat in range(4):
        assert tokenizer.players[seat].temporary_furiten is (seat in temporary_furiten_seats)
        assert tokenizer.players[seat].riichi_furiten is (seat in riichi_furiten_seats)


def test_self_resolution_follows_option_order() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_self = SelfDecision(
        actor=0,
        options={"tsumo", "ankan", "riichi"},
        option_tiles={"ankan": ["m1"]},
    )

    tokenizer._finalize_self({"riichi"})

    tail = tokenizer.tokens[-3:]
    assert tail == [
        "pass_self_0_tsumo",
        "take_self_0_riichi",
        "pass_self_0_ankan",
    ]


def test_reaction_resolution_emits_take_and_pass_in_priority_order() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = ReactionDecision(
        discarder=3,
        discard_tile=tile_to_index("m4"),
        options_by_player={0: {"chi"}, 1: {"pon", "minkan"}, 2: {"ron"}},
        chosen={1: "pon"},
    )

    tokenizer._finalize_reaction()

    tail = tokenizer.tokens[-4:]
    assert tail == [
        "pass_react_2_ron_voluntary",
        "take_react_1_pon",
        "pass_react_1_minkan_voluntary",
        "pass_react_0_chi_forced_priority",
    ]


def test_fulou_resolution_keeps_take_at_opt_order_position() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.players[0].concealed[tile_to_index("m5")] = 1
    tokenizer.players[0].concealed[tile_to_index("m6")] = 1
    tokenizer.pending_reaction = ReactionDecision(
        discarder=3,
        discard_tile=tile_to_index("m4"),
        options_by_player={0: {"chi"}, 1: {"pon"}},
    )

    tokenizer._on_fulou({"l": 0, "m": "m4-56"})

    pass_idx = tokenizer.tokens.index("pass_react_1_pon_voluntary")
    take_idx = tokenizer.tokens.index("take_react_0_chi")
    detail_idx = tokenizer.tokens.index("chi_pos_low")
    assert pass_idx < take_idx < detail_idx


def test_reaction_option_block_orders_pon_before_minkan_for_same_player() -> None:
    tokenizer = TenhouTokenizer()

    block = tokenizer._build_reaction_option_block({1: {"minkan", "pon"}})

    assert block == [
        "opt_react_1_pon",
        "opt_react_1_minkan",
    ]


def test_houtei_ron_option_uses_haidi_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = tile_to_index("m1")
    tokenizer.live_draws_left = 0

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(
        engine,
        "_pm_has_hupai_multi",
        lambda cases: [bool(case[8]) for case in cases],  # is_haidi
    )

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered)
    assert reaction is not None
    assert "ron" in reaction.options_by_player.get(1, set())


def test_kakan_ron_option_uses_qianggang_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = tile_to_index("m1")

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(
        engine,
        "_pm_has_hupai_multi",
        lambda cases: [bool(case[10]) for case in cases],  # is_qianggang
    )

    reaction = tokenizer._compute_kakan_reaction_options(actor=0, tile_idx=offered)
    assert reaction is not None
    assert "ron" in reaction.options_by_player.get(1, set())


def test_penuki_ron_option_uses_qianggang_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    offered = tile_to_index("z4")

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(
        engine,
        "_pm_has_hupai_multi",
        lambda cases: [bool(case[10]) for case in cases],  # is_qianggang
    )

    reaction = tokenizer._compute_penuki_reaction_options(actor=0, tile_idx=offered)
    assert reaction is not None
    assert "ron" in reaction.options_by_player.get(1, set())


def test_ankan_never_opens_reaction_window_under_tenhou_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = tile_to_index("z7")

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True] * len(cases))

    tokenizer.players[1].concealed = parse_hand_counts("m119p19s19z123456")
    reaction = tokenizer._compute_ankan_reaction_options(actor=0, tile_idx=offered)

    assert reaction is None

    tokenizer.players[1].concealed = parse_hand_counts("m123456789p1234")
    tokenizer._invalidate_wait_mask(1)
    reaction = tokenizer._compute_ankan_reaction_options(actor=0, tile_idx=offered)

    assert reaction is None


def test_gangzimo_last_tile_draw_eval_uses_lingshang_and_haidi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.live_draws_left = 1
    seen: dict[str, bool] = {}

    def fake_eval_draw(*, is_haidi: bool, is_lingshang: bool, **_kwargs: object) -> tuple[bool, bool]:
        seen["is_haidi"] = is_haidi
        seen["is_lingshang"] = is_lingshang
        return True, False

    monkeypatch.setattr(engine, "_pm_evaluate_draw", fake_eval_draw)

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=True)
    assert seen == {"is_haidi": False, "is_lingshang": True}
    assert "opt_self_0_tsumo" in tokenizer.tokens


def test_last_rinshan_discard_sets_houtei_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.live_draws_left = 1
    offered = tile_to_index("m2")
    seen: list[tuple[bool, bool, bool]] = []

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)

    def fake_has_hupai_multi(cases: list[tuple[object, ...]]) -> list[bool]:
        seen.extend((bool(case[8]), bool(case[9]), bool(case[10])) for case in cases)
        return [False] * len(cases)

    monkeypatch.setattr(engine, "_pm_has_hupai_multi", fake_has_hupai_multi)

    tokenizer._on_draw({"l": 0, "p": "m2"}, is_gangzimo=True)
    tokenizer._on_discard({"l": 0, "p": "m2_"})

    assert seen
    assert all(flags == (True, False, False) for flags in seen)


def test_last_penuki_replacement_discard_sets_houtei_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    tokenizer.live_draws_left = 1
    tokenizer.players[0].concealed[tile_to_index("z4")] += 1
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"penuki"}, option_tiles={"penuki": ["z4"]})
    offered = tile_to_index("m2")
    seen: list[tuple[bool, bool, bool]] = []

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)

    def fake_has_hupai_multi(cases: list[tuple[object, ...]]) -> list[bool]:
        seen.extend((bool(case[8]), bool(case[9]), bool(case[10])) for case in cases)
        return [False] * len(cases)

    monkeypatch.setattr(engine, "_pm_has_hupai_multi", fake_has_hupai_multi)

    tokenizer._on_penuki({"l": 0, "p": "z4"})
    tokenizer._on_draw({"l": 0, "p": "m2"}, is_gangzimo=True, is_replacement_draw=True)
    tokenizer.pending_dead_wall_draw = False
    tokenizer._on_discard({"l": 0, "p": "m2_"})

    assert tokenizer.live_draws_left == 0
    assert seen
    assert all(flags == (True, False, False) for flags in seen)


def test_sanchahou_takes_all_offered_ron(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())

    def fake_reaction(_self: TenhouTokenizer, _discarder: int, _tile_idx: int) -> ReactionDecision:
        return ReactionDecision(
            discarder=0,
            discard_tile=tile_to_index("m1"),
            options_by_player={0: {"chi", "ron"}, 1: {"ron"}, 2: {"ron"}},
        )

    monkeypatch.setattr(TenhouTokenizer, "_compute_reaction_options", fake_reaction)

    game = minimal_game(
        [
            qipai_event(),
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"pingju": {"name": "三家和了", "fenpei": [0, 0, 0, 0], "shoupai": ["", "", "", ""]}},
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)
    assert "take_react_0_ron" in tokens
    assert "take_react_1_ron" in tokens
    assert "take_react_2_ron" in tokens
    assert "pass_react_0_chi_voluntary" in tokens
    assert "pingju_sanchahou" in tokens


def test_riichi_ankan_requires_waits_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.players[0].is_riichi = True
    tile = tile_to_index("m1")
    tokenizer.players[0].concealed[tile] = 4

    def same_waits_mask(_counts: list[int], meld_count: int, three_player: bool = False) -> int:
        if meld_count in {0, 1}:
            return (1 << tile_to_index("m1")) | (1 << tile_to_index("m2"))
        return 0

    monkeypatch.setattr(engine, "_pm_wait_mask", same_waits_mask)
    opts = tokenizer._compute_self_options(actor=0, drawn_tile=tile)
    assert "ankan" in opts

    def changed_waits_mask(_counts: list[int], meld_count: int, three_player: bool = False) -> int:
        if meld_count == 0:
            return (1 << tile_to_index("m1")) | (1 << tile_to_index("m2"))
        if meld_count == 1:
            return 1 << tile_to_index("m3")
        return 0

    monkeypatch.setattr(engine, "_pm_wait_mask", changed_waits_mask)
    opts = tokenizer._compute_self_options(actor=0, drawn_tile=tile)
    assert "ankan" not in opts


def test_riichi_okurigang_is_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    actor = 0
    draw_tile = tile_to_index("p3")
    kan_tile = tile_to_index("m1")
    p = tokenizer.players[actor]
    p.is_riichi = True
    p.concealed = [0] * 34
    p.concealed[kan_tile] = 4
    p.concealed[draw_tile] = 1

    monkeypatch.setattr(TenhouTokenizer, "_evaluate_draw", lambda *_args, **_kwargs: (False, False))

    opts = tokenizer._compute_self_options(actor=actor, drawn_tile=draw_tile)
    assert "ankan" not in opts


def test_riichi_ankan_uses_pre_draw_waits_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    actor = 0
    draw_tile = tile_to_index("m1")
    p = tokenizer.players[actor]
    p.is_riichi = True
    p.concealed = [0] * 34
    p.concealed[draw_tile] = 4

    waits_13_mask = (1 << tile_to_index("m6")) | (1 << tile_to_index("p5"))
    waits_14_mask = waits_13_mask | (1 << tile_to_index("p4"))

    def fake_waits_mask(counts: list[int], meld_count: int, three_player: bool = False) -> int:
        if meld_count == 0:
            # Distinguish pre-draw(13) from post-draw(14) baseline by drawn tile count.
            return waits_13_mask if counts[draw_tile] == 3 else waits_14_mask
        if meld_count == 1:
            return waits_13_mask
        return 0

    monkeypatch.setattr(engine, "_pm_wait_mask", fake_waits_mask)
    monkeypatch.setattr(TenhouTokenizer, "_evaluate_draw", lambda *_args, **_kwargs: (False, False))

    opts = tokenizer._compute_self_options(actor=actor, drawn_tile=draw_tile)
    assert "ankan" in opts


def test_kaigang_after_minkan_discard_keeps_discard_reaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)

    hands = [
        "m666p123s123z1122",
        "m123456789p1234",
        "m123456789p1234",
        "m123456789p1234",
    ]
    game = minimal_game(
        [
            qipai_event(hands=hands),
            {"zimo": {"l": 3, "p": "p1"}},
            {"dapai": {"l": 3, "p": "m6"}},
            {"fulou": {"l": 0, "m": "m6666-"}},
            {"gangzimo": {"l": 0, "p": "p1"}},
            {"dapai": {"l": 0, "p": "p1_"}},
            {"kaigang": {"baopai": "z1"}},
            pingju_event(),
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)

    assert "opt_react_0_minkan" in tokens
    assert "take_react_0_minkan" in tokens
    dora_tiles = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "dora"]
    assert "z1" in dora_tiles
    discard_idx = tokens.index("discard_0_p1_tsumogiri")
    dora_idx = tokens.index("dora", discard_idx)
    pass_idx = tokens.index("pass_react_1_chi_voluntary")
    assert discard_idx < dora_idx < pass_idx


def test_delayed_kaigang_before_next_kan_draw_emits_before_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"kakan"})
    monkeypatch.setattr(TenhouTokenizer, "_compute_kakan_reaction_options", lambda *_args, **_kwargs: None)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=[
                "m123456789p1234",
                "m888p123s123z1112",
                "m123456789p1234",
                "m123456789p1234",
            ]
        )
    )
    p = tokenizer.players[1]
    p.concealed = parse_hand_counts("m888p123s123z1112")
    p.open_melds = 1
    p.open_pons[tile_to_index("m8")] = 1
    p.melds = [("pon", tile_to_index("m8"))]
    tokenizer._invalidate_meld_cache(1)
    tokenizer.pending_self = SelfDecision(actor=1, options={"kakan"})
    tokenizer.expected_discard_actor = 1

    tokenizer._on_gang({"l": 1, "m": "m888=8"})
    tokenizer._on_kaigang({"baopai": "m4"})
    tokenizer._on_draw({"l": 1, "p": "m9"}, is_gangzimo=True)

    dora_idx = tokenizer.tokens.index("dora")
    draw_idx = tokenizer.tokens.index("draw_1_m9")
    assert dora_idx < draw_idx


def test_multiple_kaigang_reveals_after_consecutive_kans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"ankan", "kakan"})
    monkeypatch.setattr(TenhouTokenizer, "_compute_ankan_reaction_options", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(TenhouTokenizer, "_compute_kakan_reaction_options", lambda *_args, **_kwargs: None)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=[
                "m123456789p1234",
                "m899p123s123z1111",
                "m123456789p1234",
                "m123456789p1234",
            ]
        )
    )
    p = tokenizer.players[1]
    p.concealed = parse_hand_counts("m899p123s123z1111")
    p.open_melds = 1
    p.open_pons[tile_to_index("m8")] = 1
    p.melds = [("pon", tile_to_index("m8"))]
    tokenizer._invalidate_meld_cache(1)
    tokenizer.pending_self = SelfDecision(actor=1, options={"kakan", "ankan"})
    tokenizer.expected_discard_actor = 1

    tokenizer._on_gang({"l": 1, "m": "m888=8"})
    tokenizer.pending_self = SelfDecision(actor=1, options={"ankan"})
    tokenizer.expected_discard_actor = 1
    tokenizer._on_gang({"l": 1, "m": "z1111"})
    tokenizer._on_kaigang({"baopai": "m4"})
    tokenizer._on_kaigang({"baopai": "m3"})
    tokenizer._on_draw({"l": 1, "p": "m9"}, is_gangzimo=True)
    tokenizer._on_discard({"l": 1, "p": "p1"})

    dora_tiles = [tokenizer.tokens[i + 1] for i, token in enumerate(tokenizer.tokens[:-1]) if token == "dora"]
    assert dora_tiles.count("m4") == 1
    assert dora_tiles.count("m3") == 1
    draw_idx = tokenizer.tokens.index("draw_1_m9")
    assert tokenizer.tokens.index("dora", 0) < draw_idx


def test_last_discard_after_rinshan_still_uses_houtei_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.live_draws_left = 0
    tokenizer.last_draw_was_gangzimo = True

    observed: dict[str, tuple[bool, bool, bool] | None] = {"flags": None}

    def fake_has_hupai_multi(cases: list[tuple[object, ...]]) -> list[bool]:
        first = cases[0]
        observed["flags"] = (bool(first[8]), bool(first[9]), bool(first[10]))
        return [True for _ in cases]

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << tile_to_index("m1"))
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", fake_has_hupai_multi)

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m1"))

    assert reaction is not None
    assert "ron" in reaction.options_by_player[1]
    assert observed["flags"] == (True, False, False)


def test_kyushukyuhai_is_emitted_as_pass_when_not_taken() -> None:
    kyushu_hand = "m1199p19s19z12345"
    game = minimal_game(
        [
            qipai_event(hands=[kyushu_hand, "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "m2"}},
            {"dapai": {"l": 0, "p": "m2_"}},
        ]
    )

    tokenizer = TenhouTokenizer()
    tokens = tokenizer.tokenize_game(game)

    assert "opt_self_0_kyushukyuhai" in tokens
    assert "pass_self_0_kyushukyuhai" in tokens
    assert "take_self_0_kyushukyuhai" not in tokens


def test_kyushukyuhai_is_not_offered_after_other_player_calls() -> None:
    hands = [
        "m1456789p1234s11",
        "m23p123456s123z11",
        "m1199p19s19z12345",
        "m123456789p1234",
    ]
    game = minimal_game(
        [
            qipai_event(hands=hands),
            {"zimo": {"l": 0, "p": "s2"}},
            {"dapai": {"l": 0, "p": "m1"}},
            {"fulou": {"l": 1, "m": "m1-23"}},
            {"dapai": {"l": 1, "p": "z1"}},
            {"zimo": {"l": 2, "p": "m2"}},
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)

    assert "take_react_1_chi" in tokens
    assert "opt_self_2_kyushukyuhai" not in tokens
    assert "pass_self_2_kyushukyuhai" not in tokens


def test_kyushukyuhai_is_emitted_as_take_on_pingju() -> None:
    kyushu_hand = "m1199p19s19z12345"
    game = minimal_game(
        [
            qipai_event(hands=[kyushu_hand, "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 0, "p": "m2"}},
            {"pingju": {"name": "九種九牌", "fenpei": [0, 0, 0, 0], "shoupai": ["m1", "", "", ""]}},
        ]
    )

    tokenizer = TenhouTokenizer()
    tokens = tokenizer.tokenize_game(game)

    assert "opt_self_0_kyushukyuhai" in tokens
    assert "take_self_0_kyushukyuhai" in tokens
    assert "pass_self_0_kyushukyuhai" not in tokens


def test_kyushukyuhai_uses_pending_actor_when_shoupai_has_multiple_non_empty() -> None:
    kyushu_hand = "m1199p19s19z12345"
    game = minimal_game(
        [
            qipai_event(hands=["m123456789p1234", "m123456789p1234", kyushu_hand, "m123456789p1234"]),
            {"zimo": {"l": 2, "p": "m2"}},
            # Converter variant: multiple non-empty shoupai slots.
            {"pingju": {"name": "九種九牌", "fenpei": [0, 0, 0, 0], "shoupai": ["m1", "", "m2", ""]}},
        ]
    )

    tokenizer = TenhouTokenizer()
    tokens = tokenizer.tokenize_game(game)

    assert "opt_self_2_kyushukyuhai" in tokens
    assert "take_self_2_kyushukyuhai" in tokens
    assert "pass_self_2_kyushukyuhai" not in tokens


def test_red_tiles_are_preserved_in_qipai_draw_and_discard_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m067p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m0"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m0_"})

    haipai_idx = tokenizer.tokens.index("haipai_0")
    assert tokenizer.tokens[haipai_idx + 1] == "m0"
    assert "draw_0_m0" in tokenizer.tokens
    assert "discard_0_m0_tsumogiri" in tokenizer.tokens


def test_discard_emits_no_tedashi_marker_when_choice_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m23456789p1234s1", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1_"})

    assert "discard_0_m1_tsumogiri" in tokenizer.tokens


def test_discard_emits_tedashi_when_same_tile_choice_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m167p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1"})

    assert "discard_0_m1_tedashi" in tokenizer.tokens


def test_discard_does_not_emit_marker_when_only_red_five_matches_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m067p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m5"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m5_"})

    assert "discard_0_m5_tsumogiri" in tokenizer.tokens


def test_discard_does_not_emit_marker_when_only_normal_five_matches_red_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m567p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m0"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m0_"})

    assert "discard_0_m0_tsumogiri" in tokenizer.tokens


def test_riichi_discard_emits_tsumogiri_when_same_tile_choice_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"riichi"})
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m167p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1*_"})

    assert "take_self_0_riichi" in tokenizer.tokens
    assert "discard_0_m1_tsumogiri" in tokenizer.tokens


def test_riichi_discard_emits_tedashi_when_same_tile_choice_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"riichi"})
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=["m167p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]
        )
    )

    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1*"})

    assert "take_self_0_riichi" in tokenizer.tokens
    assert "discard_0_m1_tedashi" in tokenizer.tokens


def test_fulou_no_longer_emits_call_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)

    hands = [
        "m067p123s123z1122",
        "m123456789p1234",
        "m123456789p1234",
        "m123456789p1234",
    ]
    game = minimal_game(
        [
            qipai_event(hands=hands),
            {"zimo": {"l": 3, "p": "p1"}},
            {"dapai": {"l": 3, "p": "m7"}},
            {"fulou": {"l": 0, "m": "m06-7"}},
            pingju_event(),
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)

    assert "take_react_0_chi" in tokens
    assert all(not t.startswith("call_") for t in tokens)


@pytest.mark.parametrize(
    ("meld", "discard", "hand", "expected"),
    [
        ("m-456", "m4", "m56p123456s123z11", "chi_pos_low"),
        ("m4-56", "m5", "m46p123456s123z11", "chi_pos_mid"),
        ("m45-6", "m6", "m45p123456s123z11", "chi_pos_high"),
    ],
)
def test_fulou_emits_chi_position_token(
    meld: str,
    discard: str,
    hand: str,
    expected: str,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=[hand, "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )
    tokenizer.pending_reaction = ReactionDecision(
        discarder=3,
        discard_tile=tile_to_index(discard),
        options_by_player={0: {"chi"}},
    )

    tokenizer._on_fulou({"l": 0, "m": meld})

    assert "take_react_0_chi" in tokenizer.tokens
    assert expected in tokenizer.tokens
    assert tokenizer.tokens.index("take_react_0_chi") < tokenizer.tokens.index(expected)


def test_fulou_emits_red_used_when_choice_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)
    game = minimal_game(
        [
            qipai_event(hands=["m056p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 3, "p": "p1"}},
            {"dapai": {"l": 3, "p": "m7"}},
            {"fulou": {"l": 0, "m": "m06-7"}},
            pingju_event(),
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)
    assert "take_react_0_chi" in tokens
    assert "chi_pos_high" in tokens
    assert "red_used" in tokens
    assert tokens.index("take_react_0_chi") < tokens.index("chi_pos_high") < tokens.index("red_used")


def test_fulou_emits_red_not_used_when_choice_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)
    game = minimal_game(
        [
            qipai_event(hands=["m056p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"]),
            {"zimo": {"l": 3, "p": "p1"}},
            {"dapai": {"l": 3, "p": "m7"}},
            {"fulou": {"l": 0, "m": "m56-7"}},
            pingju_event(),
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)
    assert "take_react_0_chi" in tokens
    assert "red_not_used" in tokens
    assert tokens.index("take_react_0_chi") < tokens.index("red_not_used")


def test_fulou_emits_red_used_and_not_used_for_pon() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m055p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )
    tokenizer.pending_reaction = ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("m5"),
        options_by_player={0: {"pon"}},
    )
    tokenizer._on_fulou({"l": 0, "m": "m05+5"})
    assert "take_react_0_pon" in tokenizer.tokens
    assert "red_used" in tokenizer.tokens
    assert tokenizer.tokens.index("take_react_0_pon") < tokenizer.tokens.index("red_used")

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m055p123s123z1122", "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )
    tokenizer.pending_reaction = ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("m5"),
        options_by_player={0: {"pon"}},
    )
    tokenizer._on_fulou({"l": 0, "m": "m55+5"})
    assert "take_react_0_pon" in tokenizer.tokens
    assert "red_not_used" in tokenizer.tokens
    assert tokenizer.tokens.index("take_react_0_pon") < tokenizer.tokens.index("red_not_used")


def test_fulou_emits_red_token_for_any_consumed_five() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m05p123s123z11223", "m123456789p1234", "m123456789p1234", "m123456789p1234"])
    )
    tokenizer.pending_reaction = ReactionDecision(
        discarder=1,
        discard_tile=tile_to_index("m5"),
        options_by_player={0: {"pon"}},
    )
    tokenizer._on_fulou({"l": 0, "m": "m05+5"})
    assert "take_react_0_pon" in tokenizer.tokens
    assert "red_used" in tokenizer.tokens


def test_kakan_no_longer_emits_kan_token(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tile = tile_to_index("m5")
    actor = 0
    p = tokenizer.players[actor]
    p.concealed[tile] = max(p.concealed[tile], 1)
    p.open_pons[tile] = 1
    p.melds = [("pon", tile)]
    monkeypatch.setattr(TenhouTokenizer, "_compute_kakan_reaction_options", lambda *_args, **_kwargs: None)
    tokenizer.pending_self = SelfDecision(actor=actor, options={"kakan"}, option_tiles={"kakan": ["m5"]})
    tokenizer.expected_discard_actor = actor

    tokenizer._on_gang({"l": actor, "m": "m5550+"})

    assert "take_self_0_kakan" in tokenizer.tokens
    take_idx = tokenizer.tokens.index("take_self_0_kakan")
    assert tokenizer.tokens[take_idx + 1] == "m5"
    assert all(not t.startswith("kan_") for t in tokenizer.tokens)


def test_ankan_no_longer_emits_kan_token() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tile = tile_to_index("m5")
    actor = 0
    tokenizer.players[actor].concealed[tile] = 4
    tokenizer.pending_self = SelfDecision(actor=actor, options={"ankan"}, option_tiles={"ankan": ["m5"]})
    tokenizer.expected_discard_actor = actor

    tokenizer._on_gang({"l": actor, "m": "m5550"})

    assert "take_self_0_ankan" in tokenizer.tokens
    take_idx = tokenizer.tokens.index("take_self_0_ankan")
    assert tokenizer.tokens[take_idx + 1] == "m5"
    assert all(not t.startswith("kan_") for t in tokenizer.tokens)


def test_ankan_does_not_emit_reaction_options_even_if_kokushi_could_ron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tile = tile_to_index("m1")
    actor = 0
    p = tokenizer.players[actor]
    p.concealed[tile] = max(p.concealed[tile], 4)
    tokenizer.pending_self = SelfDecision(actor=actor, options={"ankan"}, option_tiles={"ankan": ["m1"]})
    tokenizer.expected_discard_actor = actor

    def fail_ankan_reaction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ankan must not compute reaction options under Tenhou rules")

    monkeypatch.setattr(TenhouTokenizer, "_compute_ankan_reaction_options", fail_ankan_reaction)
    tokenizer._on_gang({"l": actor, "m": "m1111"})
    tokenizer._flush_pending()

    assert "take_self_0_ankan" in tokenizer.tokens
    take_idx = tokenizer.tokens.index("take_self_0_ankan")
    assert tokenizer.tokens[take_idx + 1] == "m1"
    assert all(not token.startswith(("opt_react_", "take_react_", "pass_react_")) for token in tokenizer.tokens)


def test_multiple_ankan_candidates_only_reveal_tile_on_take() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    actor = 0
    tokenizer.players[actor].concealed[tile_to_index("m1")] = 4
    tokenizer.players[actor].concealed[tile_to_index("z1")] = 4
    tokenizer.pending_self = SelfDecision(
        actor=actor,
        options={"ankan"},
        option_tiles={"ankan": ["m1", "z1"]},
    )

    tokenizer._finalize_self({"ankan"}, actor=actor, chosen_tiles={"ankan": "z1"})

    assert tokenizer.tokens[-2:] == [
        "take_self_0_ankan",
        "z1",
    ]


def test_multiple_kakan_candidates_only_reveal_tile_on_take() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    actor = 0
    player = tokenizer.players[actor]
    player.open_pons[tile_to_index("m1")] = 1
    player.open_pons[tile_to_index("z1")] = 1
    player.melds = [("pon", tile_to_index("m1")), ("pon", tile_to_index("z1"))]
    player.concealed[tile_to_index("m1")] = max(player.concealed[tile_to_index("m1")], 1)
    player.concealed[tile_to_index("z1")] = max(player.concealed[tile_to_index("z1")], 1)
    tokenizer.pending_self = SelfDecision(
        actor=actor,
        options={"kakan"},
        option_tiles={"kakan": ["m1", "z1"]},
    )

    tokenizer._finalize_self({"kakan"}, actor=actor, chosen_tiles={"kakan": "m1"})

    assert tokenizer.tokens[-2:] == [
        "take_self_0_kakan",
        "m1",
    ]


def test_multiple_kakan_candidates_do_not_emit_tiles_in_opt_block() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    actor = 0
    player = tokenizer.players[actor]
    player.open_pons[tile_to_index("m1")] = 1
    player.open_pons[tile_to_index("z1")] = 1
    player.melds = [("pon", tile_to_index("m1")), ("pon", tile_to_index("z1"))]
    player.concealed[tile_to_index("m1")] = max(player.concealed[tile_to_index("m1")], 1)
    player.concealed[tile_to_index("z1")] = max(player.concealed[tile_to_index("z1")], 1)
    tokenizer.expected_discard_actor = None

    tokenizer._on_draw({"l": actor, "p": "p1"}, is_gangzimo=False)

    opt_idx = tokenizer.tokens.index("opt_self_0_kakan")
    assert opt_idx + 1 >= len(tokenizer.tokens) or tokenizer.tokens[opt_idx + 1] not in {"m1", "z1"}


@pytest.mark.parametrize(
    ("method_name", "trigger_arg"),
    [
        ("_compute_kakan_reaction_options", {"actor": 0, "tile_idx": tile_to_index("m1")}),
    ],
)
def test_rob_kan_reaction_skips_furiten_players(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    trigger_arg: dict[str, int],
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = trigger_arg["tile_idx"]

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True] * len(cases))

    tokenizer.players[1].temporary_furiten = True
    tokenizer.players[2].riichi_furiten = True
    tokenizer.players[3].furiten_tiles.add(offered)

    reaction = getattr(tokenizer, method_name)(**trigger_arg)
    assert reaction is None


def test_kakan_reaction_only_offers_ron() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.players[1].concealed = [0] * 34
    tokenizer.players[1].concealed[tile_to_index("m1")] = 3

    reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
        trigger="kakan",
    )
    tokenizer.pending_reaction = reaction
    tokenizer._finalize_reaction()

    assert "take_react_1_ron" not in tokenizer.tokens
    assert "pass_react_1_ron_voluntary" in tokenizer.tokens
