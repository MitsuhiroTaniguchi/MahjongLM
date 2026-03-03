from __future__ import annotations

import pytest

pytest.importorskip("pymahjong")

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import ReactionDecision, TenhouTokenizer, tile_to_index
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
        "_can_win",
        lambda _self, seat, _tile, is_tsumo: (not is_tsumo) and seat in {1, 2},
    )

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
