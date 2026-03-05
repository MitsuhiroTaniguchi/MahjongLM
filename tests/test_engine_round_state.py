from __future__ import annotations

import pytest
import pymahjong  # noqa: F401

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import ReactionDecision, SelfDecision, TenhouTokenizer, tile_to_index
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event, qipai_payload


def test_round_index_resets_per_game() -> None:
    game = minimal_game([qipai_event(), pingju_event()])
    tokenizer = TenhouTokenizer()

    first = tokenizer.tokenize_game(game)
    second = tokenizer.tokenize_game(game)

    assert first.count("round_seq_0") == 1
    assert second.count("round_seq_0") == 1
    assert "round_seq_1" not in second


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
    tokenizer._on_draw({"l": seat, "p": "m2"}, is_gangzimo=False)
    tokenizer._is_permanent_furiten(seat)
    assert calls["n"] == 2


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
            ]
        ]
    ) -> list[bool]:
        called["n"] = len(cases)
        return [True] * len(cases)

    monkeypatch.setattr(engine, "PM_MULTI_HUPAI_AVAILABLE", True)
    monkeypatch.setattr(engine.pm, "has_hupai_multi", fake_has_hupai_multi, raising=False)

    out = engine._pm_has_hupai_multi(
        [
            ([0] * 34, [], 0, False, True, False, 0, 0, True, False, False),
            ([0] * 34, [], 1, False, True, False, 0, 1, False, True, True),
        ]
    )
    assert out == [True, True]
    assert called["n"] == 2


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
