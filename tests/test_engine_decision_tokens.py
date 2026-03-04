from __future__ import annotations

import pytest

pytest.importorskip("pymahjong")

import tenhou_tokenizer.engine as engine
from tenhou_tokenizer.engine import ReactionDecision, TenhouTokenizer, tile_to_index
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event, qipai_payload


def test_permanent_furiten_blocks_ron_for_all_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    offered_tile = tile_to_index("m2")
    discarded_wait = tile_to_index("m1")

    tokenizer.players[1].furiten_tiles.add(discarded_wait)
    monkeypatch.setattr(engine, "_pm_wait_tiles", lambda *_args, **_kwargs: {discarded_wait, offered_tile})
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: True)

    reaction = tokenizer._compute_reaction_options(discarder=0, tile_idx=offered_tile)

    assert reaction is not None
    assert "ron" not in reaction.options_by_player.get(1, set())


def test_hule_with_baojia_is_classified_as_ron() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer._on_hule({"l": 2, "baojia": 0, "fenpei": [0, 0, 0, 0]})

    assert "win_ron_2_from_0" in tokenizer.tokens
    assert "win_tsumo_2" not in tokenizer.tokens


def test_kaigang_does_not_force_self_pass_before_tsumo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: {"tsumo"})

    game = minimal_game(
        [
            qipai_event(),
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
    assert tokens.index("take_self_0_tsumo") < tokens.index("win_tsumo_0")


def test_kakan_generates_reaction_decision_and_rob_kan_take(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tile = tile_to_index("m1")
    actor = 0
    p = tokenizer.players[actor]
    p.concealed[tile] = max(p.concealed[tile], 1)
    p.open_pons[tile] = 1
    p.melds = [("pon", tile)]

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

    assert "kan_kakan_0_m1" in tokenizer.tokens
    assert "opt_react_1_ron" in tokenizer.tokens
    assert "win_ron_1_from_0" in tokenizer.tokens
    assert "take_react_1_ron" in tokenizer.tokens


def test_passing_ron_by_taking_other_call_sets_temporary_furiten() -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())

    tokenizer.pending_reaction = ReactionDecision(
        discarder=0,
        discard_tile=tile_to_index("m1"),
        options_by_player={1: {"chi", "ron"}},
        chosen={1: "chi"},
    )
    tokenizer._finalize_reaction()

    assert tokenizer.players[1].temporary_furiten
    assert "take_react_1_chi" in tokenizer.tokens
    assert "pass_react_1_ron" in tokenizer.tokens


def test_riichi_ankan_requires_waits_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer._on_qipai(qipai_payload())
    tokenizer.players[0].is_riichi = True
    tile = tile_to_index("m1")
    tokenizer.players[0].concealed[tile] = 4

    def same_waits(_counts: list[int], meld_count: int) -> set[int]:
        return {tile_to_index("m1"), tile_to_index("m2")} if meld_count in {0, 1} else set()

    monkeypatch.setattr(engine, "_pm_wait_tiles", same_waits)
    opts = tokenizer._compute_self_options(actor=0, drawn_tile=tile)
    assert "ankan" in opts

    def changed_waits(_counts: list[int], meld_count: int) -> set[int]:
        if meld_count == 0:
            return {tile_to_index("m1"), tile_to_index("m2")}
        if meld_count == 1:
            return {tile_to_index("m3")}
        return set()

    monkeypatch.setattr(engine, "_pm_wait_tiles", changed_waits)
    opts = tokenizer._compute_self_options(actor=0, drawn_tile=tile)
    assert "ankan" not in opts


def test_riichi_ankan_uses_pre_draw_waits_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
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

    waits_13 = {tile_to_index("m6"), tile_to_index("p5")}
    waits_14 = waits_13 | {tile_to_index("p4")}

    def fake_waits(counts: list[int], meld_count: int) -> set[int]:
        if meld_count == 0:
            # Distinguish pre-draw(13) from post-draw(14) baseline by drawn tile count.
            return waits_13 if counts[draw_tile] == 0 else waits_14
        if meld_count == 1:
            return waits_13
        return set()

    monkeypatch.setattr(engine, "_pm_wait_tiles", fake_waits)
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)

    opts = tokenizer._compute_self_options(actor=actor, drawn_tile=draw_tile)
    assert "ankan" in opts


def test_kaigang_between_discard_and_fulou_keeps_discard_reaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TenhouTokenizer, "_compute_self_options", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(TenhouTokenizer, "_can_win", lambda *_args, **_kwargs: False)

    hands = [
        "m5567p123s123z112",
        "m123456789p1234",
        "m123456789p1234",
        "m123456789p1234",
    ]
    game = minimal_game(
        [
            qipai_event(hands=hands),
            {"zimo": {"l": 3, "p": "p1"}},
            {"dapai": {"l": 3, "p": "m6"}},
            {"kaigang": {"baopai": "z1"}},
            {"fulou": {"l": 0, "m": "m56-7"}},
            {"dapai": {"l": 0, "p": "m6"}},
            pingju_event(),
        ]
    )

    tokens = TenhouTokenizer().tokenize_game(game)

    assert "opt_react_0_chi" in tokens
    assert "take_react_0_chi" in tokens
    assert "call_chi_0_m5_m6_m7" in tokens
    assert "discard_0_m6" in tokens


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
