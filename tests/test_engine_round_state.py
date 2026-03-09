from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import (
    ReactionDecision,
    SelfDecision,
    TenhouTokenizer,
    TokenizeError,
    parse_hand_counts,
    tile_to_index,
)
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event, qipai_payload


@pytest.fixture(autouse=True)
def disable_pm_simulation_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)


def test_round_start_resets_per_game() -> None:
    game = minimal_game([qipai_event(), pingju_event()])
    tokenizer = TenhouTokenizer()

    first = tokenizer.tokenize_game(game)
    second = tokenizer.tokenize_game(game)

    assert first.count("round_start") == 1
    assert second.count("round_start") == 1


def test_dealer_seat_is_not_derived_from_jushu(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    def fake_has_hupai(*, lunban: int, **_kwargs) -> bool:
        captured["lunban"] = lunban
        return False

    monkeypatch.setattr(engine, "_pm_has_hupai", fake_has_hupai)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload(jushu=3))

    tokenizer._can_win(seat=2, win_tile=tile_to_index("m1"), is_tsumo=False)

    assert tokenizer.dealer_seat == 0
    assert captured["lunban"] == 2


def test_temporary_furiten_sets_on_ron_pass_and_clears_on_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
    )
    tokenizer._finalize_reaction()
    assert tokenizer.players[1].temporary_furiten

    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer._on_draw({"l": 1, "p": "m2"}, is_gangzimo=False)

    assert not tokenizer.players[1].temporary_furiten


def test_call_options_are_blocked_on_last_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.live_draws_left = 0

    m1 = tile_to_index("m1")
    m2 = tile_to_index("m2")
    m3 = tile_to_index("m3")

    tokenizer.players[1].concealed[m1] = 1
    tokenizer.players[1].concealed[m3] = 1
    tokenizer.players[2].concealed[m2] = 2

    monkeypatch.setattr(
        TenhouTokenizer,
        "_wait_mask",
        lambda _self, seat: (1 << m2) if seat in {1, 2} else 0,
    )
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=m2)

    assert reaction is not None
    assert reaction.options_by_player == {1: {"ron"}, 2: {"ron"}}


@pytest.mark.parametrize(
    ("discarder", "tile", "hands", "expected"),
    [
        (
            0,
            "m2",
            ["m123456789p1234", "m13p123456s123z11", "m22p123456s123z11", "m22p123456s123z11"],
            {1: {"chi"}, 2: {"pon"}, 3: {"pon"}},
        ),
        (
            1,
            "m2",
            ["m22p123456s123z11", "m123456789p1234", "m13p123456s123z11", "m22p123456s123z11"],
            {2: {"chi"}, 3: {"pon"}, 0: {"pon"}},
        ),
        (
            2,
            "z1",
            ["z11m12p1234s12344", "z11m12p1234s12344", "m123456789p1234", "z11m12p1234s12344"],
            {3: {"pon"}, 0: {"pon"}, 1: {"pon"}},
        ),
    ],
)
def test_compute_reaction_options_seat_matrix(
    discarder: int,
    tile: str,
    hands: list[str],
    expected: dict[int, set[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload(hands=hands))

    offered = tile_to_index(tile)
    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=discarder, tile_idx=offered)

    assert reaction is not None
    assert reaction.options_by_player == expected


def test_compute_reaction_options_riichi_player_never_gets_call_options(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m123456789p1234", "m13p123456s123z11", "m22p123456s123z11", "m123456789p1234"])
    )
    tokenizer.players[1].is_riichi = True

    offered = tile_to_index("m2")
    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered)

    assert reaction is not None
    assert 1 not in reaction.options_by_player
    assert reaction.options_by_player == {2: {"pon"}}


def test_compute_reaction_options_omits_chi_for_non_shimocha(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m123456789p1234", "m99p123456s123z11", "m123456789p1234", "m22p123456s123z11"])
    )

    offered = tile_to_index("m2")
    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered)

    assert reaction is not None
    assert reaction.options_by_player == {3: {"pon"}}


def test_compute_reaction_options_haitei_allows_only_ron(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(
            hands=[
                "m123456789p1234",
                "m23p123456s123z11",
                "m22p123456s123z11",
                "m222p123456s123z1",
            ]
        )
    )
    tokenizer.live_draws_left = 0

    offered = tile_to_index("m1")
    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True for _ in cases])

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered)

    assert reaction is not None
    assert reaction.options_by_player == {1: {"ron"}, 2: {"ron"}, 3: {"ron"}}


def test_compute_reaction_options_rejects_chi_when_kuikae_leaves_no_legal_discard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m1")] = 1
    p.concealed[tile_to_index("m2")] = 2
    p.concealed[tile_to_index("m3")] = 1

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m2"))

    assert reaction is not None
    assert "chi" not in reaction.options_by_player.get(1, set())


def test_compute_reaction_options_allows_chi_when_only_outer_tile_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m5")] = 1

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m4"))

    assert reaction is not None
    assert "chi" in reaction.options_by_player.get(1, set())


def test_compute_reaction_options_rejects_high_side_chi_when_only_opposite_outer_tile_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m1")] = 1

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m4"))

    assert reaction is None or "chi" not in reaction.options_by_player.get(1, set())


def test_compute_reaction_options_rejects_low_side_chi_when_only_opposite_outer_tile_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m4")] = 1

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m1"))

    assert reaction is None or "chi" not in reaction.options_by_player.get(1, set())


def test_compute_reaction_options_rejects_pon_when_kuikae_leaves_no_legal_discard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[2]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 3

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m2"))

    assert reaction is not None
    assert "pon" not in reaction.options_by_player.get(2, set())


def test_compute_reaction_options_filters_illegal_chi_from_simulation_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m1")] = 1
    p.concealed[tile_to_index("m2")] = 2
    p.concealed[tile_to_index("m3")] = 1

    monkeypatch.setattr(
        engine.pm,
        "compute_reaction_option_masks",
        lambda *_args, **_kwargs: [(1, engine.REACT_OPT_CHI)],
    )

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m2"))

    assert reaction is None


def test_compute_reaction_options_keeps_legal_outer_tile_chi_from_simulation_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m5")] = 1

    monkeypatch.setattr(
        engine.pm,
        "compute_reaction_option_masks",
        lambda *_args, **_kwargs: [(1, engine.REACT_OPT_CHI)],
    )

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m4"))

    assert reaction is not None
    assert reaction.options_by_player == {1: {"chi"}}


def test_compute_reaction_options_filters_illegal_edge_kuikae_chi_from_simulation_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m1")] = 1

    monkeypatch.setattr(
        engine.pm,
        "compute_reaction_option_masks",
        lambda *_args, **_kwargs: [(1, engine.REACT_OPT_CHI)],
    )

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m4"))

    assert reaction is None


def test_compute_reaction_options_pm_stateless_enforces_kuikae_without_tokenizer_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    monkeypatch.setattr(
        tokenizer,
        "_filter_reaction_call_options",
        lambda discarder, tile_idx, options_by_player: options_by_player,
    )

    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m1")] = 1
    p.concealed[tile_to_index("m2")] = 2
    p.concealed[tile_to_index("m3")] = 1

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m2"))
    assert reaction is not None
    assert reaction.options_by_player == {1: {"pon"}}

    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m5")] = 1

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m4"))
    assert reaction is not None
    assert reaction.options_by_player == {1: {"chi"}}


def test_compute_reaction_options_pm_shoupai_enforces_kuikae_without_tokenizer_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    monkeypatch.setattr(
        tokenizer,
        "_filter_reaction_call_options",
        lambda discarder, tile_idx, options_by_player: options_by_player,
    )

    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m1")] = 1
    p.concealed[tile_to_index("m2")] = 2
    p.concealed[tile_to_index("m3")] = 1
    p.sim_shoupai = engine.pm.Shoupai(tuple(p.concealed))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m2"))
    assert reaction is not None
    assert reaction.options_by_player == {1: {"pon"}}

    p.concealed = [0] * 34
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m3")] = 1
    p.concealed[tile_to_index("m5")] = 1
    p.sim_shoupai = engine.pm.Shoupai(tuple(p.concealed))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m4"))
    assert reaction is not None
    assert reaction.options_by_player == {1: {"chi"}}


def test_compute_reaction_options_allows_red_five_outer_tile_chi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("p4")] = 1
    p.concealed[tile_to_index("p5")] = 1
    p.concealed[tile_to_index("p7")] = 1
    p.red_fives["p"] = 1

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("p6"))

    assert reaction is not None
    assert reaction.options_by_player == {1: {"chi"}}


def test_compute_reaction_options_keeps_multiple_legal_chi_shapes_when_one_survives_kuikae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    p = tokenizer.players[1]
    p.concealed = [0] * 34
    p.concealed[tile_to_index("m1")] = 1
    p.concealed[tile_to_index("m2")] = 1
    p.concealed[tile_to_index("m4")] = 1
    p.concealed[tile_to_index("m5")] = 1

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [False] * len(cases))

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m3"))

    assert reaction is not None
    assert reaction.options_by_player == {1: {"chi"}}


def test_compute_kakan_reaction_options_seat_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = tile_to_index("m1")

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True, False, True][: len(cases)])

    reaction = tokenizer._compute_kakan_reaction_options(actor=0, tile_idx=offered)

    assert reaction is not None
    assert reaction.trigger == "kakan"
    assert reaction.options_by_player == {1: {"ron"}, 3: {"ron"}}


def test_compute_kakan_reaction_options_skips_permanent_furiten(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = tile_to_index("m1")

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True] * len(cases))

    tokenizer.players[2].furiten_tiles.add(offered)

    reaction = tokenizer._compute_kakan_reaction_options(actor=0, tile_idx=offered)

    assert reaction is not None
    assert reaction.options_by_player == {1: {"ron"}, 3: {"ron"}}


def test_compute_ankan_reaction_options_requires_kokushi_per_seat(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    offered = tile_to_index("z7")

    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True] * len(cases))

    tokenizer.players[1].concealed = parse_hand_counts("m119p19s19z123456")
    tokenizer.players[2].concealed = parse_hand_counts("m123456789p1234")
    tokenizer.players[3].concealed = parse_hand_counts("m19p19s19z1123456")
    tokenizer._invalidate_wait_mask(1)
    tokenizer._invalidate_wait_mask(2)
    tokenizer._invalidate_wait_mask(3)

    reaction = tokenizer._compute_ankan_reaction_options(actor=0, tile_idx=offered)

    assert reaction is not None
    assert reaction.trigger == "ankan"
    assert reaction.options_by_player == {1: {"ron"}, 3: {"ron"}}


def test_riichi_missed_ron_becomes_persistent_furiten(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.players[1].is_riichi = True

    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
    )
    tokenizer._finalize_reaction()

    assert tokenizer.players[1].riichi_furiten
    assert not tokenizer.players[1].temporary_furiten

    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer._on_draw({"l": 1, "p": "m2"}, is_gangzimo=False)
    assert tokenizer.players[1].riichi_furiten

    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: True)
    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_to_index("m1"))
    assert reaction is None or "ron" not in reaction.options_by_player.get(1, set())


def test_riichi_missed_tsumo_becomes_persistent_furiten(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    seat = 1
    tile_idx = tile_to_index("m1")
    tokenizer.players[seat].is_riichi = True
    tokenizer.pending_self = SelfDecision(actor=seat, options={"tsumo"})

    tokenizer._finalize_self(set(), actor=seat)

    assert tokenizer.players[seat].riichi_furiten
    assert "pass_self_1_tsumo" in tokenizer.tokens

    monkeypatch.setattr(TenhouTokenizer, "_wait_mask", lambda _self, s: (1 << tile_idx) if s == seat else 0)
    monkeypatch.setattr(engine, "_pm_has_hupai_multi", lambda cases: [True] * len(cases))
    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=tile_idx)
    assert reaction is None or "ron" not in reaction.options_by_player.get(seat, set())


def test_wait_mask_is_cached_until_player_state_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    seat = 1
    calls = {"n": 0}

    def fake_wait_mask(_counts: list[int], _meld_count: int) -> int:
        calls["n"] += 1
        return 1 << tile_to_index("m1")

    monkeypatch.setattr(engine, "_pm_wait_mask", fake_wait_mask)

    tokenizer._is_permanent_furiten(seat)
    tokenizer._is_permanent_furiten(seat)
    assert calls["n"] == 1

    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    tokenizer.expected_draw_actor = seat
    tokenizer._on_draw({"l": seat, "p": "m2"}, is_gangzimo=False)
    tokenizer._is_permanent_furiten(seat)
    assert calls["n"] == 2


def test_wait_mask_is_invalidated_by_fulou_and_gang() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(
        qipai_payload(hands=["m123456789p1234", "m23p123456s123z11", "m1111p123s123z112", "m123456789p1234"])
    )

    tokenizer.players[1].wait_mask_cache = 123
    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"chi"}},
    )
    tokenizer._on_fulou({"l": 1, "m": "m1-23"})
    assert tokenizer.players[1].wait_mask_cache is None

    tokenizer.players[2].wait_mask_cache = 456
    tokenizer.pending_self = SelfDecision(actor=2, options={"ankan"})
    tokenizer.expected_discard_actor = 2
    tokenizer._on_gang({"l": 2, "m": "m1111"})
    assert tokenizer.players[2].wait_mask_cache is None


def test_draw_rejects_when_no_live_draws_remain() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.live_draws_left = 0

    with pytest.raises(TokenizeError):
        tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)


def test_draw_rejects_unexpected_actor_turn() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.expected_draw_actor = 0

    with pytest.raises(TokenizeError):
        tokenizer._on_draw({"l": 1, "p": "m1"}, is_gangzimo=False)


def test_discard_rejects_unexpected_actor_turn() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)

    with pytest.raises(TokenizeError):
        tokenizer._on_discard({"l": 1, "p": "m1"})


def test_kaigang_rejects_when_not_expected() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    with pytest.raises(TokenizeError):
        tokenizer._on_kaigang({"baopai": "p1"})


def test_kaigang_rejects_when_minkan_was_only_offered() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"minkan"}},
        trigger="discard",
    )

    with pytest.raises(TokenizeError):
        tokenizer._on_kaigang({"baopai": "p1"})


def test_stateless_simulation_path_does_not_require_shoupai_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine.pm, "Shoupai", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Shoupai should not be used")))
    monkeypatch.setattr(engine.pm, "Action", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Action should not be used")))
    monkeypatch.setattr(engine.pm, "compute_self_option_mask", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine.pm, "compute_reaction_option_masks", lambda *_args, **_kwargs: [])

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer._on_draw({"l": 0, "p": "m1"}, is_gangzimo=False)
    tokenizer._on_discard({"l": 0, "p": "m1"})

    assert all(player.sim_shoupai is None for player in tokenizer.players)


def test_rob_kan_simulation_zero_masks_do_not_create_pending_reaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine, "PM_SHOUPAI_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", True)
    monkeypatch.setattr(engine.pm, "compute_rob_kan_option_masks", lambda *_args, **_kwargs: [(1, 0), (2, 0)])

    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    assert tokenizer._compute_kakan_reaction_options(actor=0, tile_idx=tile_to_index("m1")) is None
    assert tokenizer._compute_ankan_reaction_options(actor=0, tile_idx=tile_to_index("m1")) is None


def test_reaction_ron_shape_gate_skips_hupai_call(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.live_draws_left = 0

    # Make all potential callers unable to chi/pon/minkan so only ron is relevant.
    for seat in range(4):
        tokenizer.players[seat].concealed = [0] * 34

    offered = tile_to_index("m5")
    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 0)

    def fail_has_hupai_multi(*_args, **_kwargs) -> bool:
        raise AssertionError("has_hupai_multi should not be called when discard is outside wait mask")

    monkeypatch.setattr(engine, "_pm_has_hupai_multi", fail_has_hupai_multi)

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered)
    assert reaction is None


def test_compute_self_options_uses_combined_draw_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    actor = 0
    draw_tile = tile_to_index("m1")
    observed: dict[str, object] = {}

    def fake_evaluate_draw(
        counts: list[int],
        melds: list[tuple[str, int]],
        encoded_melds: list[tuple[int, int]],
        win_tile: int,
        is_menqian: bool,
        is_riichi: bool,
        zhuangfeng: int,
        lunban: int,
        closed_kans: int,
        check_riichi_discard: bool,
        is_haidi: bool,
        is_lingshang: bool,
    ) -> tuple[bool, bool]:
        observed["win_tile"] = win_tile
        observed["is_menqian"] = is_menqian
        observed["encoded_melds"] = encoded_melds
        observed["check_riichi_discard"] = check_riichi_discard
        observed["is_haidi"] = is_haidi
        observed["is_lingshang"] = is_lingshang
        return True, True

    monkeypatch.setattr(engine, "_pm_evaluate_draw", fake_evaluate_draw)

    opts = tokenizer._compute_self_options(actor=actor, drawn_tile=draw_tile)
    assert "tsumo" in opts
    assert "riichi" in opts
    assert observed["win_tile"] == draw_tile
    assert observed["is_menqian"] is True
    assert observed["encoded_melds"] == []
    assert observed["check_riichi_discard"] is True
    assert observed["is_haidi"] is False
    assert observed["is_lingshang"] is False


def test_pm_has_hupai_uses_context_enabled_has_hupai(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_has_hupai(
        _hand: tuple[int, ...],
        _melds: list[tuple[int, int]],
        _win_tile: int,
        _is_tsumo: bool,
        _is_menqian: bool,
        _is_riichi: bool,
        _zhuangfeng: int,
        _lunban: int,
        is_haidi: bool,
        is_lingshang: bool,
        is_qianggang: bool,
    ) -> bool:
        called["flags"] = (is_haidi, is_lingshang, is_qianggang)
        return True

    monkeypatch.setattr(engine, "PM_FASTAPI_AVAILABLE", True)
    monkeypatch.setattr(engine.pm, "has_hupai", fake_has_hupai, raising=False)

    out = engine._pm_has_hupai(
        counts=[0] * 34,
        melds=[],
        encoded_melds=None,
        win_tile=0,
        is_tsumo=False,
        is_menqian=True,
        is_riichi=False,
        zhuangfeng=0,
        lunban=0,
        is_haidi=True,
        is_lingshang=False,
        is_qianggang=True,
    )
    assert out is True
    assert called["flags"] == (True, False, True)


@pytest.mark.parametrize(
    ("trigger", "key", "value", "expected"),
    [
        ("discard", "kaigang", {"baopai": "m1"}, True),
        ("discard", "fulou", {"l": 1, "m": "m1-23"}, True),
        ("discard", "hule", {"l": 1, "baojia": 2}, True),
        ("discard", "hule", {"l": 1, "baojia": 1}, False),
        ("kakan", "fulou", {"l": 1, "m": "m1-23"}, False),
        ("kakan", "hule", {"l": 1, "baojia": 2}, True),
        ("ankan", "kaigang", {"baopai": "m1"}, True),
    ],
)
def test_is_reaction_continuation_matrix(
    trigger: str,
    key: str,
    value: dict,
    expected: bool,
) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.pending_reaction = ReactionDecision(
        discarder=2,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
        trigger=trigger,
    )

    assert tokenizer._is_reaction_continuation(key, value) is expected


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("kaigang", {"baopai": "m1"}, True),
        ("dapai", {"l": 1, "p": "m1"}, True),
        ("dapai", {"l": 2, "p": "m1"}, False),
        ("gang", {"l": 1, "m": "m1111"}, True),
        ("gang", {"l": 2, "m": "m1111"}, False),
        ("hule", {"l": 1, "fenpei": [0, 0, 0, 0]}, True),
        ("hule", {"l": 1, "baojia": 2, "fenpei": [0, 0, 0, 0]}, False),
        ("pingju", {"name": "流局", "fenpei": [0, 0, 0, 0]}, False),
        ("pingju", {"name": "九種九牌", "fenpei": [0, 0, 0, 0]}, True),
    ],
)
def test_is_self_resolution_matrix(key: str, value: dict, expected: bool) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.pending_self = SelfDecision(actor=1, options={"riichi"})

    assert tokenizer._is_self_resolution(key, value) is expected


def test_pending_riichi_actor_clears_on_ron_and_voluntary_close() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
        chosen={1: "ron"},
        trigger="discard",
    )
    tokenizer.pending_riichi_actor = 0
    tokenizer._finalize_reaction(close_reason="voluntary")
    assert tokenizer.pending_riichi_actor is None
    assert tokenizer.players[0].score == 25000

    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"ron"}},
        trigger="discard",
    )
    tokenizer.pending_riichi_actor = 0
    tokenizer._finalize_reaction(close_reason="voluntary")
    assert tokenizer.pending_riichi_actor is None
    assert tokenizer.players[0].score == 24000


def test_pm_has_hupai_multi_uses_context_enabled_has_hupai_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def fake_has_hupai_multi(
        cases: list[
            tuple[
                tuple[int, ...],
                list[tuple[int, int]],
                int,
                bool,
                bool,
                bool,
                int,
                int,
                bool,
                bool,
                bool,
                bool,
            ]
        ]
    ) -> list[bool]:
        called["n"] = len(cases)
        return [True] * len(cases)

    monkeypatch.setattr(engine, "PM_MULTI_HUPAI_AVAILABLE", True)
    monkeypatch.setattr(engine.pm, "has_hupai_multi", fake_has_hupai_multi, raising=False)

    out = engine._pm_has_hupai_multi(
        [
            ([0] * 34, [], 0, False, True, False, 0, 0, True, False, False, False),
            ([0] * 34, [], 1, False, True, False, 0, 1, False, True, True, False),
        ]
    )
    assert out == [True, True]
    assert called["n"] == 2


def test_compute_reaction_options_threads_three_player_to_ron_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    tokenizer.live_draws_left = 10

    offered = tile_to_index("m2")
    monkeypatch.setattr(engine, "PM_STATELESS_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "PM_SIMULATION_API_AVAILABLE", False)
    monkeypatch.setattr(engine, "_pm_wait_mask", lambda *_args, **_kwargs: 1 << offered)

    seen: dict[str, object] = {}

    def fake_has_hupai_multi(cases):
        seen["three_player"] = cases[0][-1]
        return [True] * len(cases)

    monkeypatch.setattr(engine, "_pm_has_hupai_multi", fake_has_hupai_multi)

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered)

    assert reaction is not None
    assert 1 in reaction.options_by_player
    assert seen["three_player"] is True


def test_ankan_candidate_tiles_threads_three_player_wait_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3, hands=["m1111p123s123z123", "m123456789p1234", "m123456789p1234"]))
    seat = 0
    tokenizer.players[seat].is_riichi = True

    seen: list[bool] = []

    def fake_wait_mask(_counts: list[int], _meld_count: int, three_player: bool = False) -> int:
        seen.append(three_player)
        return 1 << tile_to_index("m2")

    monkeypatch.setattr(engine, "_pm_wait_mask", fake_wait_mask)

    out = tokenizer._ankan_candidate_tiles(seat, drawn_tile=tile_to_index("m1"))

    assert out == ["m1"]
    assert seen == [True, True]


def test_pm_evaluate_draw_uses_context_enabled_evaluate_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_evaluate_draw(
        _hand: tuple[int, ...],
        _melds: list[tuple[int, int]],
        _win_tile: int,
        _is_menqian: bool,
        _is_riichi: bool,
        _zhuangfeng: int,
        _lunban: int,
        _closed_kans: int,
        _check_riichi_discard: bool,
        is_haidi: bool,
        is_lingshang: bool,
    ) -> tuple[bool, bool]:
        called["flags"] = (is_haidi, is_lingshang)
        return True, False

    monkeypatch.setattr(engine, "PM_EVALUATE_DRAW_AVAILABLE", True)
    monkeypatch.setattr(engine.pm, "evaluate_draw", fake_evaluate_draw, raising=False)

    out = engine._pm_evaluate_draw(
        counts=[0] * 34,
        melds=[],
        encoded_melds=None,
        win_tile=0,
        is_menqian=True,
        is_riichi=False,
        zhuangfeng=0,
        lunban=0,
        closed_kans=0,
        check_riichi_discard=False,
        is_haidi=True,
        is_lingshang=True,
    )
    assert out == (True, False)
    assert called["flags"] == (True, True)


def test_pm_evaluate_draw_threads_three_player_to_hupai_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_has_hupai(**kwargs):
        called["three_player"] = kwargs["three_player"]
        return True

    monkeypatch.setattr(engine, "PM_EVALUATE_DRAW_AVAILABLE", False)
    monkeypatch.setattr(engine, "_pm_has_hupai", fake_has_hupai)

    out = engine._pm_evaluate_draw(
        counts=[0] * 34,
        melds=[],
        encoded_melds=None,
        win_tile=0,
        is_menqian=True,
        is_riichi=False,
        zhuangfeng=0,
        lunban=0,
        closed_kans=0,
        check_riichi_discard=False,
        three_player=True,
    )

    assert out == (True, False)
    assert called["three_player"] is True


def test_penuki_replacement_draw_is_treated_as_dead_wall_draw() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = 3
    tokenizer._on_qipai(qipai_payload(seat_count=3))
    tokenizer.players[0].concealed[tile_to_index("z4")] += 1
    tokenizer.pending_self = engine.SelfDecision(actor=0, options={"penuki"}, option_tiles={"penuki": ["z4"]})

    tokenizer._on_penuki({"l": 0, "p": "z4"})

    assert tokenizer.pending_dead_wall_draw is True
    assert tokenizer.live_draws_left == 55

    tokenizer._on_draw(
        {"l": 0, "p": "m1"},
        is_gangzimo=tokenizer.pending_dead_wall_draw,
        is_replacement_draw=tokenizer.pending_dead_wall_draw,
    )
    tokenizer.pending_dead_wall_draw = False

    assert tokenizer.live_draws_left == 55
    assert tokenizer.last_draw_was_gangzimo is True
