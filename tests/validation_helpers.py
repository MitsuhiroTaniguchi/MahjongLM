from __future__ import annotations

from typing import Sequence

from tenhou_tokenizer.engine import (
    TenhouTokenizer,
    classify_fulou,
    classify_gang,
    parse_meld_tiles,
)


def validate_token_stream(tokens: Sequence[str]) -> None:
    assert tokens.count("game_start") == 1
    assert tokens.count("game_end") == 1
    assert not any(token.startswith("event_unknown") for token in tokens)

    seen_self: set[str] = set()
    seen_react: set[str] = set()
    saw_call_in_round = False

    for token in tokens:
        if token.startswith("round_seq_"):
            seen_self.clear()
            seen_react.clear()
            saw_call_in_round = False
            continue

        if token.startswith("opt_self_"):
            key = token.replace("opt_self_", "", 1)
            if key.endswith("_kyushukyuhai"):
                assert not saw_call_in_round
            seen_self.add(key)
            continue

        if token.startswith("opt_react_"):
            seen_react.add(token.replace("opt_react_", "", 1))
            continue

        if token.startswith("take_self_"):
            assert token.replace("take_self_", "", 1) in seen_self
            continue

        if token.startswith("take_react_"):
            assert token.replace("take_react_", "", 1) in seen_react
            saw_call_in_round = True


def validate_score_rotation(game: dict) -> None:
    tokenizer = TenhouTokenizer()
    rounds = game.get("log", [])
    for idx, round_data in enumerate(rounds[:-1]):
        tokenizer._process_round(round_data)
        next_round = rounds[idx + 1]
        if not next_round or "qipai" not in next_round[0]:
            continue
        expected_scores = list(next_round[0]["qipai"]["defen"])
        internal_scores = [player.score for player in tokenizer.players]
        assert any(
            internal_scores[offset:] + internal_scores[:offset] == expected_scores
            for offset in range(4)
        )


def validate_player_state_invariants(tokenizer: TenhouTokenizer) -> None:
    for seat, player in enumerate(tokenizer.players):
        assert all(count >= 0 for count in player.concealed)
        for suit, red_count in player.red_fives.items():
            assert red_count >= 0
            five_index = {"m": 4, "p": 13, "s": 22}[suit]
            assert red_count <= player.concealed[five_index]

        meld_tile_count = 0
        for meld_type, _tile in player.melds:
            if meld_type in {"chi", "pon"}:
                meld_tile_count += 3
            elif meld_type in {"minkan", "ankan"}:
                meld_tile_count += 4
            else:
                raise AssertionError(f"unexpected meld type: {meld_type}")

        concealed_tile_count = sum(player.concealed)
        total_tiles = concealed_tile_count + meld_tile_count
        kan_count = sum(1 for meld_type, _tile in player.melds if meld_type in {"minkan", "ankan"})
        assert total_tiles in {13 + kan_count, 14 + kan_count}

        assert player.open_melds >= 0
        assert player.closed_kans >= 0
        assert len(player.melds) == player.open_melds + player.closed_kans
        assert player.meld_count == player.open_melds + player.closed_kans
        for tile, count in player.open_pons.items():
            assert count > 0
            assert player.open_pons_red.get(tile, 0) <= count + 2


def validate_round_stepwise(round_data: list[dict]) -> None:
    assert isinstance(round_data, list)
    assert round_data
    tokenizer = TenhouTokenizer()
    saw_qipai = False

    for event in round_data:
        assert isinstance(event, dict)
        assert event
        key, value = next(iter(event.items()))
        if not saw_qipai:
            assert key == "qipai"
            saw_qipai = True
        else:
            assert key != "qipai"
        before_live_draws = tokenizer.live_draws_left
        before_concealed = [sum(player.concealed) for player in tokenizer.players]

        if tokenizer.pending_reaction and not tokenizer._is_reaction_continuation(key, value):
            close_reason = "forced_rule" if key == "pingju" else "voluntary"
            tokenizer._finalize_reaction(close_reason=close_reason)
        if tokenizer.pending_self and not tokenizer._is_self_resolution(key, value):
            tokenizer._finalize_self(set())

        if key == "qipai":
            tokenizer._on_qipai(value)
        elif key == "zimo":
            tokenizer._on_draw(value, is_gangzimo=False)
        elif key == "gangzimo":
            tokenizer._on_draw(value, is_gangzimo=True)
        elif key == "dapai":
            tokenizer._on_discard(value)
        elif key == "fulou":
            tokenizer._on_fulou(value)
        elif key == "gang":
            tokenizer._on_gang(value)
        elif key == "kaigang":
            tokenizer._on_kaigang(value)
        elif key == "hule":
            tokenizer._on_hule(value)
        elif key == "pingju":
            tokenizer._on_pingju(value)
        else:
            tokenizer.tokens.append(f"event_unknown_{key}")

        after_concealed = [sum(player.concealed) for player in tokenizer.players]
        validate_player_state_invariants(tokenizer)

        if key in {"zimo", "gangzimo"}:
            actor = value["l"]
            assert after_concealed[actor] == before_concealed[actor] + 1
            assert tokenizer.live_draws_left == before_live_draws - 1
        elif key == "dapai":
            actor = value["l"]
            assert after_concealed[actor] == before_concealed[actor] - 1
        elif key == "fulou":
            actor = value["l"]
            action = classify_fulou(parse_meld_tiles(value["m"]))
            expected_delta = -2 if action in {"chi", "pon"} else -3
            assert after_concealed[actor] == before_concealed[actor] + expected_delta
        elif key == "gang":
            actor = value["l"]
            action = classify_gang(value["m"])
            if action == "ankan":
                assert after_concealed[actor] == before_concealed[actor] - 4
            else:
                assert after_concealed[actor] == before_concealed[actor] - 1
