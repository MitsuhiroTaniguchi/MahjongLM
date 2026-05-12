from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from tenhou_tokenizer.engine import (
    TenhouTokenizer,
    classify_fulou,
    classify_gang,
    parse_meld_tiles,
)


TENBO_TOKENS = {
    "TENBO_PLUS",
    "TENBO_MINUS",
    "TENBO_ZERO",
    "TENBO_100",
    "TENBO_200",
    "TENBO_300",
    "TENBO_400",
    "TENBO_500",
    "TENBO_600",
    "TENBO_700",
    "TENBO_800",
    "TENBO_900",
    "TENBO_1000",
    "TENBO_2000",
    "TENBO_3000",
    "TENBO_4000",
    "TENBO_5000",
    "TENBO_6000",
    "TENBO_7000",
    "TENBO_8000",
    "TENBO_9000",
    "TENBO_10000",
}
TILE_TOKENS = {
    *(f"m{i}" for i in range(10)),
    *(f"p{i}" for i in range(10)),
    *(f"s{i}" for i in range(10)),
    *(f"z{i}" for i in range(1, 8)),
}
RED_CHOICE_TOKENS = {"red_used", "red_not_used"}
DISCARD_MARKER_TOKENS: set[str] = set()
REACTION_PASS_SUFFIXES = ("_voluntary", "_forced_priority")


def _is_discard_token(token: str) -> bool:
    parts = token.split("_")
    return (
        len(parts) == 4
        and parts[0] == "discard"
        and parts[1].isdigit()
        and parts[2] in TILE_TOKENS
        and parts[3] in {"tedashi", "tsumogiri"}
    )


def _is_rank_token(token: str) -> bool:
    parts = token.split("_")
    return len(parts) == 3 and parts[0] == "rank" and parts[1].isdigit() and parts[2].isdigit()


def _is_final_rank_token(token: str) -> bool:
    parts = token.split("_")
    return (
        len(parts) == 4
        and parts[0] == "final"
        and parts[1] == "rank"
        and parts[2].isdigit()
        and parts[3].isdigit()
    )


def _is_rule_token(token: str) -> bool:
    return token in {
        "rule_player_3",
        "rule_player_4",
        "rule_length_tonpu",
        "rule_length_hanchan",
    }


def _is_view_token(token: str) -> bool:
    return token in {"view_complete", "view_omniscient"} or (
        token.startswith("view_imperfect_") and token.split("_")[-1].isdigit()
    )


def _is_hule_detail_token(token: str) -> bool:
    if token.startswith("hule_"):
        parts = token.split("_")
        return len(parts) == 2 and parts[1].isdigit()
    if token.startswith("opened_hand_"):
        parts = token.split("_")
        return len(parts) == 3 and parts[2].isdigit()
    if token == "ura_dora":
        return True
    if token.startswith("yaku_"):
        return True
    if token.startswith("han_") or token.startswith("fu_") or token.startswith("yakuman_"):
        parts = token.split("_")
        return len(parts) == 2 and parts[1].isdigit()
    return False


def _consume_tenbo_payload(tokens: Sequence[str], start: int) -> int:
    assert start < len(tokens)
    head = tokens[start]
    if head == "TENBO_ZERO":
        return start + 1
    assert head in {"TENBO_PLUS", "TENBO_MINUS"}
    idx = start + 1
    assert idx < len(tokens)
    saw_unit = False
    while idx < len(tokens) and tokens[idx] in TENBO_TOKENS - {"TENBO_PLUS", "TENBO_MINUS", "TENBO_ZERO"}:
        idx += 1
        saw_unit = True
    assert saw_unit
    return idx


def _consume_tile_payload(tokens: Sequence[str], start: int, *, minimum: int, exact: int | None = None) -> int:
    idx = start
    count = 0
    while idx < len(tokens) and tokens[idx] in TILE_TOKENS:
        count += 1
        idx += 1
    if exact is not None:
        assert count == exact
    else:
        assert count >= minimum
    return idx


def _consume_round_prelude(tokens: Sequence[str], start: int, *, seat_count: int) -> int:
    idx = start
    if idx < len(tokens) and tokens[idx] == "wall":
        idx = _consume_tile_payload(tokens, idx + 1, minimum=136, exact=136)
    assert idx < len(tokens) and tokens[idx].startswith("bakaze_")
    idx += 1
    assert idx < len(tokens) and tokens[idx].startswith("kyoku_")
    idx += 1
    assert idx < len(tokens) and tokens[idx] == "honba"
    idx = _consume_tenbo_payload(tokens, idx + 1)
    assert idx < len(tokens) and tokens[idx] == "riichi_sticks"
    idx = _consume_tenbo_payload(tokens, idx + 1)
    assert idx < len(tokens) and tokens[idx] == "dora"
    idx = _consume_tile_payload(tokens, idx + 1, minimum=1, exact=1)
    for seat in range(seat_count):
        assert idx < len(tokens) and tokens[idx] == f"score_{seat}"
        idx = _consume_tenbo_payload(tokens, idx + 1)
    for seat in range(seat_count):
        assert idx < len(tokens) and tokens[idx] == f"haipai_{seat}"
        idx = _consume_tile_payload(tokens, idx + 1, minimum=13, exact=13)
    return idx


class StreamPhase(str, Enum):
    ACTIONS = "actions"
    RESULT = "result"


@dataclass
class TokenStreamFSM:
    tokens: Sequence[str]
    idx: int = 0
    phase: StreamPhase = StreamPhase.ACTIONS
    seen_self: set[str] = field(default_factory=set)
    seen_react: set[str] = field(default_factory=set)
    saw_call_in_round: bool = False
    expected_score_delta_seat: int | None = None
    expected_rank_seat: int | None = None
    allow_post_tsumo_pass_self: bool = False
    started_round: bool = False
    pending_kaigang_reveals: int = 0
    awaiting_ron_score_block: bool = False
    expected_final_score_seat: int | None = None
    expected_final_rank_seat: int | None = None
    saw_final_suffix: bool = False
    seat_count: int = 4
    saw_game_start: bool = False
    saw_game_end: bool = False

    def validate(self) -> None:
        assert self.tokens.count("game_start") == 1
        assert self.tokens.count("game_end") == 1
        assert not any(token.startswith("event_unknown") for token in self.tokens)
        assert len(self.tokens) >= 3

        while self.idx < len(self.tokens):
            token = self.tokens[self.idx]
            if _is_rule_token(token):
                assert not self.saw_game_start
                assert not self.started_round
                if token == "rule_player_3":
                    self.seat_count = 3
                elif token == "rule_player_4":
                    self.seat_count = 4
                self.idx += 1
                continue
            if _is_view_token(token):
                assert not self.saw_game_start
                assert not self.started_round
                self.idx += 1
                continue
            if token == "game_start":
                assert not self.saw_game_start
                assert not self.started_round
                self.saw_game_start = True
                self.idx += 1
                continue
            if token == "round_start":
                assert self.saw_game_start
                assert not self.saw_game_end
                if self.started_round:
                    self._assert_round_closed()
                self._start_round()
                continue
            if token == "round_end":
                self._assert_round_closed()
                assert self.started_round
                self.idx += 1
                continue
            if token == "game_end":
                assert self.saw_game_start
                self._assert_round_closed()
                if self.expected_final_score_seat is not None or self.expected_final_rank_seat is not None:
                    raise AssertionError("final suffix is incomplete before game_end")
                assert self.idx > 0 and self.tokens[self.idx - 1] == "round_end"
                assert not self.saw_game_end
                self.saw_game_end = True
                self.idx += 1
                continue
            if self._consume_final_token(token):
                continue
            if self.phase is StreamPhase.RESULT or self.awaiting_ron_score_block:
                if self._consume_result_token(token):
                    continue
            if self._consume_action_token(token):
                continue
            self._fail_unexpected_token(token)

    def _start_round(self) -> None:
        self.started_round = True
        self.seen_self.clear()
        self.seen_react.clear()
        self.saw_call_in_round = False
        self.phase = StreamPhase.ACTIONS
        self.expected_score_delta_seat = None
        self.expected_rank_seat = None
        self.allow_post_tsumo_pass_self = False
        self.pending_kaigang_reveals = 0
        self.awaiting_ron_score_block = False
        self.idx = _consume_round_prelude(self.tokens, self.idx + 1, seat_count=self.seat_count)

    def _assert_round_closed(self) -> None:
        assert self.phase is StreamPhase.RESULT
        assert self.expected_score_delta_seat is None
        assert self.expected_rank_seat is None
        assert not self.allow_post_tsumo_pass_self
        assert not self.awaiting_ron_score_block

    def _enter_result_phase(self, *, allow_post_tsumo_pass_self: bool) -> None:
        self.phase = StreamPhase.RESULT
        self.expected_score_delta_seat = 0
        self.allow_post_tsumo_pass_self = allow_post_tsumo_pass_self
        self.awaiting_ron_score_block = False

    def _consume_result_token(self, token: str) -> bool:
        assert not token.startswith("opt_self_")
        assert not token.startswith("opt_react_")
        if token.startswith("take_self_"):
            raise AssertionError(f"take_self token is not allowed in result phase: {token}")
        if token.startswith("take_react_"):
            assert self.expected_score_delta_seat is None
            assert token.endswith("_ron")
            self.expected_rank_seat = None
            self.awaiting_ron_score_block = True
            self.idx += 1
            return True
        if token.startswith("pass_self_"):
            assert self.allow_post_tsumo_pass_self
            self.idx += 1
            if self.idx >= len(self.tokens) or not self.tokens[self.idx].startswith("pass_self_"):
                self.allow_post_tsumo_pass_self = False
            return True
        if token.startswith("pass_react_"):
            assert self.expected_score_delta_seat is None
            pass_key = token.replace("pass_react_", "", 1)
            matched = False
            for suffix in REACTION_PASS_SUFFIXES:
                if pass_key.endswith(suffix):
                    assert pass_key[: -len(suffix)] in self.seen_react
                    matched = True
                    break
            assert matched
            self.idx += 1
            return True
        assert not token.startswith("draw_")
        assert not _is_discard_token(token)
        assert token not in DISCARD_MARKER_TOKENS
        assert not token.startswith("chi_pos_")
        assert token not in RED_CHOICE_TOKENS
        if _is_hule_detail_token(token):
            assert self.expected_rank_seat is None
            assert self.expected_score_delta_seat in {None, 0}
            if token == "ura_dora" or token.startswith("opened_hand_"):
                self.idx = _consume_tile_payload(self.tokens, self.idx + 1, minimum=1)
            else:
                self.idx += 1
            return True
        if token.startswith("score_delta_"):
            if self.expected_score_delta_seat is None:
                assert self.awaiting_ron_score_block
                self.phase = StreamPhase.RESULT
                self.expected_score_delta_seat = 0
                self.awaiting_ron_score_block = False
            seat = int(token.split("_")[2])
            assert seat == self.expected_score_delta_seat
            self.idx = _consume_tenbo_payload(self.tokens, self.idx + 1)
            self.allow_post_tsumo_pass_self = False
            if self.expected_score_delta_seat == self.seat_count - 1:
                self.expected_score_delta_seat = None
                self.expected_rank_seat = 0
            else:
                self.expected_score_delta_seat += 1
            return True
        if _is_rank_token(token):
            assert self.expected_rank_seat is not None
            seat = int(token.split("_")[1])
            assert seat == self.expected_rank_seat
            self.idx += 1
            if self.expected_rank_seat == self.seat_count - 1:
                self.expected_rank_seat = None
            else:
                self.expected_rank_seat += 1
            return True
        return False

    def _consume_action_token(self, token: str) -> bool:
        if self.awaiting_ron_score_block:
            assert (
                token.startswith("take_react_")
                or token.startswith("pass_react_")
                or token.startswith("score_delta_")
                or token == "pingju_sanchahou"
                or _is_hule_detail_token(token)
            )
        if token.startswith("opt_self_"):
            key = token.replace("opt_self_", "", 1)
            if key.endswith("_kyushukyuhai"):
                assert not self.saw_call_in_round
            self.seen_self.add(key)
            self.idx += 1
            return True
        if token.startswith("opt_react_"):
            self.seen_react.add(token.replace("opt_react_", "", 1))
            self.idx += 1
            return True
        if token.startswith("take_self_"):
            assert token.replace("take_self_", "", 1) in self.seen_self
            parts = token.split("_")
            if parts[-1] == "tsumo":
                self._enter_result_phase(allow_post_tsumo_pass_self=True)
            if parts[-1] in {"ankan", "kakan"}:
                self.pending_kaigang_reveals += 1
            if parts[-1] in {"ankan", "kakan", "tsumo"}:
                self.idx = _consume_tile_payload(self.tokens, self.idx + 1, minimum=1, exact=1)
            else:
                self.idx += 1
            return True
        if token.startswith("pass_self_"):
            assert token.replace("pass_self_", "", 1) in self.seen_self
            self.idx += 1
            return True
        if token.startswith("take_react_"):
            assert token.replace("take_react_", "", 1) in self.seen_react
            parts = token.split("_")
            if parts[-1] == "ron":
                self.awaiting_ron_score_block = True
            if parts[-1] == "minkan":
                self.pending_kaigang_reveals += 1
            self.saw_call_in_round = True
            self.idx += 1
            if parts[-1] == "chi":
                if self.idx < len(self.tokens) and self.tokens[self.idx].startswith("chi_pos_"):
                    self.idx += 1
                if self.idx < len(self.tokens) and self.tokens[self.idx] in RED_CHOICE_TOKENS:
                    self.idx += 1
            elif parts[-1] == "pon":
                if self.idx < len(self.tokens) and self.tokens[self.idx] in RED_CHOICE_TOKENS:
                    self.idx += 1
            return True
        if token.startswith("pass_react_"):
            pass_key = token.replace("pass_react_", "", 1)
            matched = False
            for suffix in REACTION_PASS_SUFFIXES:
                if pass_key.endswith(suffix):
                    assert pass_key[: -len(suffix)] in self.seen_react
                    matched = True
                    break
            assert matched
            self.idx += 1
            return True
        if token.startswith("pingju_"):
            self._enter_result_phase(allow_post_tsumo_pass_self=False)
            self.idx += 1
            return True
        if token == "dora":
            assert self.pending_kaigang_reveals > 0
            self.idx = _consume_tile_payload(self.tokens, self.idx + 1, minimum=1, exact=1)
            self.pending_kaigang_reveals -= 1
            return True
        if _is_discard_token(token):
            self.idx += 1
            return True
        if token.startswith("draw_"):
            self.idx += 1
            return True
        return False

    def _consume_final_token(self, token: str) -> bool:
        if token.startswith("final_score_"):
            assert self.started_round
            self._assert_round_closed()
            assert self.saw_game_end
            if self.expected_final_score_seat is None and self.expected_final_rank_seat is None:
                self.expected_final_score_seat = 0
                self.saw_final_suffix = True
            assert self.expected_final_score_seat is not None
            seat = int(token.split("_")[2])
            assert seat == self.expected_final_score_seat
            self.idx = _consume_tenbo_payload(self.tokens, self.idx + 1)
            if self.expected_final_score_seat == self.seat_count - 1:
                self.expected_final_score_seat = None
                self.expected_final_rank_seat = 0
            else:
                self.expected_final_score_seat += 1
            return True
        if _is_final_rank_token(token):
            assert self.expected_final_score_seat is None
            assert self.expected_final_rank_seat is not None
            seat = int(token.split("_")[2])
            assert seat == self.expected_final_rank_seat
            self.idx += 1
            if self.expected_final_rank_seat == self.seat_count - 1:
                self.expected_final_rank_seat = None
            else:
                self.expected_final_rank_seat += 1
            return True
        return False

    def _fail_unexpected_token(self, token: str) -> None:
        assert token not in TILE_TOKENS
        assert token not in TENBO_TOKENS
        assert token not in DISCARD_MARKER_TOKENS
        assert token not in RED_CHOICE_TOKENS
        assert not token.startswith("chi_pos_")
        assert not _is_rank_token(token)
        assert not _is_final_rank_token(token)
        raise AssertionError(f"unexpected token in stream validator: {token}")


def validate_token_stream(tokens: Sequence[str]) -> None:
    TokenStreamFSM(tokens).validate()

def validate_score_rotation(game: dict) -> None:
    tokenizer = TenhouTokenizer()
    tokenizer.seat_count = tokenizer._infer_game_seat_count(game)
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
        for offset in range(len(internal_scores))
        )
    if rounds:
        tokenizer = TenhouTokenizer()
        tokenizer.seat_count = tokenizer._infer_game_seat_count(game)
        if "qijia" in game:
            tokenizer.initial_qijia = int(game["qijia"])
            tokenizer.has_initial_qijia = True
        for round_data in rounds:
            tokenizer._process_round(round_data)
        if "rank" in game:
            final_scores = tokenizer._current_game_order_scores()
            # Older Tenhou logs can expose rounded top-level owari/defen values,
            # so top-level rank is only a reliable validator when defen matches
            # the exact score reconstructed from round deltas.
            if game.get("defen") == final_scores:
                assert tokenizer._compute_final_rank_places(final_scores) == game["rank"]


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


def validate_event_token_slice(event_key: str, emitted: Sequence[str]) -> None:
    if event_key == "qipai":
        assert emitted
        assert emitted[0] == "round_start"
        seat_count = len([token for token in emitted if token.startswith("haipai_")])
        assert seat_count in {3, 4}
        assert _consume_round_prelude(emitted, 1, seat_count=seat_count) == len(emitted)
        return

    if event_key in {"zimo", "gangzimo"}:
        draw_positions = [i for i, token in enumerate(emitted) if token.startswith("draw_")]
        assert len(draw_positions) == 1
        draw_idx = draw_positions[0]
        idx = 0
        while idx < draw_idx:
            token = emitted[idx]
            if token.startswith("pass_self_") or token.startswith("pass_react_"):
                idx += 1
                continue
            if token == "dora":
                idx = _consume_tile_payload(emitted, idx + 1, minimum=1, exact=1)
                continue
            raise AssertionError(f"unexpected draw-prefix token: {token}")
        idx = draw_idx + 1
        while idx < len(emitted):
            token = emitted[idx]
            assert token.startswith("opt_self_")
            idx += 1
        return

    if event_key == "dapai":
        discard_positions = [i for i, token in enumerate(emitted) if _is_discard_token(token)]
        assert len(discard_positions) == 1
        discard_idx = discard_positions[0]
        prefix = emitted[:discard_idx]
        idx = 0
        while idx < len(prefix):
            token = prefix[idx]
            if token.startswith("take_self_"):
                idx += 1
                continue
            if token.startswith("pass_self_"):
                idx += 1
                continue
            if token.startswith("pass_react_"):
                idx += 1
                continue
            raise AssertionError(f"unexpected discard-prefix token: {token}")
        idx = discard_idx + 1
        while idx < len(emitted) and emitted[idx] == "dora":
            idx = _consume_tile_payload(emitted, idx + 1, minimum=1, exact=1)
        assert all(token.startswith("opt_react_") for token in emitted[idx:])
        return

    if event_key == "fulou":
        take_positions = [i for i, token in enumerate(emitted) if token.startswith("take_react_")]
        assert len(take_positions) == 1
        take_idx = take_positions[0]
        assert all(token.startswith("pass_react_") for token in emitted[:take_idx])
        idx = take_idx + 1
        while idx < len(emitted) and (
            emitted[idx].startswith("chi_pos_") or emitted[idx] in RED_CHOICE_TOKENS
        ):
            idx += 1
        assert all(token.startswith("pass_react_") for token in emitted[idx:])
        return

    if event_key == "gang":
        take_positions = [i for i, token in enumerate(emitted) if token.startswith("take_self_")]
        assert len(take_positions) == 1
        take_idx = take_positions[0]
        assert all(token.startswith("pass_self_") for token in emitted[:take_idx])
        idx = take_idx + 1
        assert idx < len(emitted) and emitted[idx] in TILE_TOKENS
        idx += 1
        while idx < len(emitted) and emitted[idx].startswith("pass_self_"):
            idx += 1
        assert all(token.startswith("opt_react_") for token in emitted[idx:])
        return

    if event_key == "penuki":
        take_positions = [i for i, token in enumerate(emitted) if token.startswith("take_self_")]
        assert len(take_positions) == 1
        take_idx = take_positions[0]
        assert all(token.startswith("pass_self_") or token.startswith("opt_self_") for token in emitted[:take_idx])
        assert emitted[take_idx].endswith("_penuki")
        return

    if event_key == "kaigang":
        assert len(emitted) == 0
        return

    if event_key == "hule":
        if any(token.startswith("take_react_") and token.endswith("_ron") for token in emitted):
            first_delta_idx = next(
                (i for i, token in enumerate(emitted) if token.startswith("score_delta_")),
                len(emitted),
            )
            idx = 0
            while idx < first_delta_idx:
                token = emitted[idx]
                if token.startswith("take_react_") or token.startswith("pass_react_"):
                    idx += 1
                    continue
                if token == "ura_dora" or token.startswith("opened_hand_"):
                    idx = _consume_tile_payload(emitted, idx + 1, minimum=1)
                    continue
                assert _is_hule_detail_token(token)
                idx += 1
        elif any(token.startswith("score_delta_") for token in emitted):
            first_delta_idx = next(i for i, token in enumerate(emitted) if token.startswith("score_delta_"))
            has_tsumo_take = any(
                token.startswith("take_self_") and token.endswith("_tsumo") for token in emitted[:first_delta_idx]
            )
            assert has_tsumo_take or any(token.startswith("hule_") for token in emitted[:first_delta_idx])
            idx = 0
            while idx < first_delta_idx:
                token = emitted[idx]
                if token.startswith("take_self_") or token.startswith("pass_self_") or token in TILE_TOKENS:
                    idx += 1
                    continue
                if token == "ura_dora" or token.startswith("opened_hand_"):
                    idx = _consume_tile_payload(emitted, idx + 1, minimum=1)
                    continue
                assert _is_hule_detail_token(token)
                idx += 1
        delta_positions = [i for i, token in enumerate(emitted) if token.startswith("score_delta_")]
        if delta_positions:
            seat_count = len(delta_positions)
            assert seat_count in {3, 4}
            rank_positions = [i for i, token in enumerate(emitted) if _is_rank_token(token)]
            assert len(rank_positions) in {0, seat_count}
            if rank_positions:
                assert rank_positions[0] > delta_positions[-1]
        return

    if event_key == "pingju":
        pingju_positions = [i for i, token in enumerate(emitted) if token.startswith("pingju_")]
        assert len(pingju_positions) == 1
        pingju_idx = pingju_positions[0]
        assert all(
            token.startswith("pass_self_")
            or token.startswith("pass_react_")
            or token.endswith("_kyushukyuhai") and token.startswith("take_self_")
            or token.startswith("take_react_") and token.endswith("_ron")
            for token in emitted[:pingju_idx]
        )
        delta_count = len([token for token in emitted if token.startswith("score_delta_")])
        assert delta_count in {3, 4}
        assert len([token for token in emitted if _is_rank_token(token)]) == delta_count
        first_delta_idx = next(i for i, token in enumerate(emitted) if token.startswith("score_delta_"))
        idx = pingju_idx + 1
        while idx < first_delta_idx:
            token = emitted[idx]
            assert token.startswith("opened_hand_")
            idx = _consume_tile_payload(emitted, idx + 1, minimum=1)
        return


def trace_round_token_slices(round_data: list[dict]) -> tuple[TenhouTokenizer, list[dict]]:
    assert isinstance(round_data, list)
    assert round_data
    tokenizer = TenhouTokenizer()
    saw_qipai = False
    round_ended = False
    traces: list[dict] = []

    for event_index, event in enumerate(round_data):
        assert isinstance(event, dict)
        assert event
        key, value = next(iter(event.items()))
        if not saw_qipai:
            assert key == "qipai"
            saw_qipai = True
        else:
            assert key != "qipai"
            if round_ended:
                is_multi_ron_continuation = (
                    key == "hule"
                    and isinstance(value, dict)
                    and (
                        (
                            tokenizer.pending_reaction is not None
                            and value.get("baojia") == tokenizer.pending_reaction.discarder
                        )
                        or value.get("baojia") == tokenizer.pending_multi_ron_baojia
                    )
                )
                assert is_multi_ron_continuation, "round already ended"

        before_live_draws = tokenizer.live_draws_left
        before_concealed = [sum(player.concealed) for player in tokenizer.players]
        before_token_count = len(tokenizer.tokens)

        if tokenizer.pending_reaction and not tokenizer._is_reaction_continuation(key, value):
            if key != "pingju":
                tokenizer._finalize_reaction(close_reason="voluntary")
        if tokenizer.pending_self and not tokenizer._is_self_resolution(key, value):
            tokenizer._finalize_self(set())

        if key == "qipai":
            if isinstance(value, dict):
                shoupai = value.get("shoupai")
                if isinstance(shoupai, list) and len(shoupai) in {3, 4}:
                    tokenizer.seat_count = len(shoupai)
            tokenizer._on_qipai(value)
        elif key == "zimo":
            tokenizer._on_draw(value, is_gangzimo=False)
        elif key == "gangzimo":
            tokenizer._on_draw(value, is_gangzimo=True)
        elif key == "dapai":
            tokenizer._on_discard(value)
        elif key == "fulou":
            tokenizer._on_fulou(value)
        elif key == "penuki":
            tokenizer._on_penuki(value)
        elif key == "gang":
            tokenizer._on_gang(value)
        elif key == "kaigang":
            tokenizer._on_kaigang(value)
        elif key == "hule":
            remaining_ron_winners: set[int] | None = None
            more_ron_expected = False
            if isinstance(value, dict) and value.get("baojia") is not None:
                remaining_ron_winners = set()
                lookahead_index = event_index
                expected_baojia = value.get("baojia")
                while lookahead_index < len(round_data):
                    lookahead_event = round_data[lookahead_index]
                    if not isinstance(lookahead_event, dict) or "hule" not in lookahead_event:
                        break
                    lookahead_hule = lookahead_event["hule"]
                    if not isinstance(lookahead_hule, dict) or lookahead_hule.get("baojia") != expected_baojia:
                        break
                    winner = lookahead_hule.get("l")
                    if isinstance(winner, int) and not isinstance(winner, bool):
                        remaining_ron_winners.add(winner)
                    lookahead_index += 1
                more_ron_expected = len(remaining_ron_winners) > 1
            tokenizer._on_hule(
                value,
                more_ron_expected=more_ron_expected,
                remaining_ron_winners=remaining_ron_winners,
            )
        elif key == "pingju":
            tokenizer._on_pingju(value)
        else:
            tokenizer.tokens.append(f"event_unknown_{key}")

        emitted = tokenizer.tokens[before_token_count:]
        validate_player_state_invariants(tokenizer)
        traces.append(
            {
                "event_index": event_index,
                "event_key": key,
                "event_value": value,
                "tokens": emitted,
                "before_live_draws": before_live_draws,
                "after_live_draws": tokenizer.live_draws_left,
                "before_concealed": before_concealed,
                "after_concealed": [sum(player.concealed) for player in tokenizer.players],
            }
        )
        if key in {"hule", "pingju"}:
            round_ended = True

    return tokenizer, traces


def validate_round_stepwise(round_data: list[dict]) -> None:
    tokenizer, traces = trace_round_token_slices(round_data)

    for trace in traces:
        key = trace["event_key"]
        value = trace["event_value"]
        before_live_draws = trace["before_live_draws"]
        after_live_draws = trace["after_live_draws"]
        before_concealed = trace["before_concealed"]
        after_concealed = trace["after_concealed"]

        validate_event_token_slice(key, trace["tokens"])

        if key in {"zimo", "gangzimo"}:
            actor = value["l"]
            assert after_concealed[actor] == before_concealed[actor] + 1
            assert after_live_draws == before_live_draws - 1
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
