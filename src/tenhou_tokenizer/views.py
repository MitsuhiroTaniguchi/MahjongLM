from __future__ import annotations

from dataclasses import dataclass
from .engine import TenhouTokenizer
from .viewspec import (
    TOKEN_VIEW_COMPLETE,
    VIEW_COMPLETE,
    VIEW_IMPERFECT,
    imperfect_view_token,
)


TILE_TOKENS = {
    *(f"m{i}" for i in range(10)),
    *(f"p{i}" for i in range(10)),
    *(f"s{i}" for i in range(10)),
    *(f"z{i}" for i in range(1, 8)),
}
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


@dataclass(frozen=True)
class TokenizedGameView:
    view_type: str
    viewer_seat: int | None
    tokens: list[str]


def _split_rule_prefix(tokens: list[str]) -> tuple[list[str], list[str]]:
    prefix_end = 0
    while prefix_end < len(tokens) and tokens[prefix_end].startswith("rule_"):
        prefix_end += 1
    return tokens[:prefix_end], tokens[prefix_end:]


def _hidden_haipai_token(seat: int) -> str:
    return f"hidden_haipai_{seat}"


def _hidden_draw_token(seat: int) -> str:
    return f"draw_{seat}_hidden"


def _round_kyoku_by_index(game: dict, seat_count: int) -> dict[int, int]:
    round_kyoku: dict[int, int] = {}
    log = game.get("log")
    if not isinstance(log, list):
        return round_kyoku
    for round_index, round_data in enumerate(log):
        if not isinstance(round_data, list) or not round_data:
            continue
        first = round_data[0]
        if not isinstance(first, dict):
            continue
        qipai = first.get("qipai")
        if not isinstance(qipai, dict):
            continue
        kyoku = qipai.get("jushu")
        if isinstance(kyoku, int) and not isinstance(kyoku, bool):
            round_kyoku[round_index] = kyoku % seat_count
    return round_kyoku


def _viewer_round_seat(
    viewer_player: int,
    *,
    initial_qijia: int,
    seat_count: int,
    round_kyoku: int,
) -> int:
    viewer_game_seat = (initial_qijia + viewer_player) % seat_count
    dealer_game_seat = (initial_qijia + round_kyoku) % seat_count
    return (viewer_game_seat - dealer_game_seat) % seat_count


def _consume_tenbo_payload(tokens: list[str], start: int) -> int:
    if tokens[start] == "TENBO_ZERO":
        return start + 1
    idx = start + 1
    while idx < len(tokens) and tokens[idx] in TENBO_TOKENS - {"TENBO_PLUS", "TENBO_MINUS", "TENBO_ZERO"}:
        idx += 1
    return idx


def _seat_from_self_token(token: str) -> int:
    return int(token.split("_")[2])


def _seat_from_react_token(token: str) -> int:
    return int(token.split("_")[2])


def _transform_qipai(tokens: list[str], viewer_seat: int | None) -> list[str]:
    if viewer_seat is None:
        return list(tokens)
    out: list[str] = []
    idx = 0
    out.extend(tokens[idx : idx + 4])
    idx = 4
    idx_after_honba = _consume_tenbo_payload(tokens, idx)
    out.extend(tokens[idx:idx_after_honba])
    idx = idx_after_honba
    out.append(tokens[idx])
    idx += 1
    idx_after_riichi = _consume_tenbo_payload(tokens, idx)
    out.extend(tokens[idx:idx_after_riichi])
    idx = idx_after_riichi
    out.extend(tokens[idx : idx + 2])
    idx += 2

    seat_count = len([token for token in tokens if token.startswith("haipai_")])
    for _seat in range(seat_count):
        out.append(tokens[idx])
        idx += 1
        idx_after_score = _consume_tenbo_payload(tokens, idx)
        out.extend(tokens[idx:idx_after_score])
        idx = idx_after_score

    for seat in range(seat_count):
        if viewer_seat == seat:
            out.append(tokens[idx])
            idx += 1
            out.extend(tokens[idx : idx + 13])
        else:
            out.append(_hidden_haipai_token(seat))
            idx += 1
        idx += 13

    return out


def _transform_event_tokens(tokens: list[str], viewer_seat: int | None, event_key: str) -> list[str]:
    if viewer_seat is None:
        return list(tokens)
    if event_key == "qipai":
        return _transform_qipai(tokens, viewer_seat)

    out: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("draw_"):
            actor = int(token.split("_")[1])
            if actor == viewer_seat:
                out.append(token)
            else:
                out.append(_hidden_draw_token(actor))
            idx += 1
            continue
        if token.startswith("opt_self_"):
            actor = _seat_from_self_token(token)
            keep = actor == viewer_seat
            if keep:
                out.append(token)
            idx += 1
            while idx < len(tokens) and tokens[idx] in TILE_TOKENS:
                if keep:
                    out.append(tokens[idx])
                idx += 1
            continue
        if token.startswith("pass_self_"):
            actor = _seat_from_self_token(token)
            keep = actor == viewer_seat
            if keep:
                out.append(token)
            idx += 1
            if idx < len(tokens) and tokens[idx] in TILE_TOKENS:
                if keep:
                    out.append(tokens[idx])
                idx += 1
            continue
        if token.startswith("take_self_"):
            out.append(token)
            idx += 1
            if idx < len(tokens) and tokens[idx] in TILE_TOKENS:
                out.append(tokens[idx])
                idx += 1
            continue
        if token.startswith("opt_react_"):
            actor = _seat_from_react_token(token)
            if actor == viewer_seat:
                out.append(token)
            idx += 1
            continue
        if token.startswith("pass_react_"):
            actor = _seat_from_react_token(token)
            if actor == viewer_seat:
                out.append(token)
            idx += 1
            continue
        out.append(token)
        idx += 1
    return out


def tokenize_game_views(game: dict) -> list[TokenizedGameView]:
    tokenizer = TenhouTokenizer()
    complete_tokens = tokenizer.tokenize_game(game)
    event_traces = list(tokenizer.event_traces)
    rule_prefix, body_tokens = _split_rule_prefix(complete_tokens)
    if not event_traces:
        return [
            TokenizedGameView(
                view_type=VIEW_COMPLETE,
                viewer_seat=None,
                tokens=[*rule_prefix, TOKEN_VIEW_COMPLETE, *body_tokens],
            )
        ]

    views = [
        TokenizedGameView(
            view_type=VIEW_COMPLETE,
            viewer_seat=None,
            tokens=[*rule_prefix, TOKEN_VIEW_COMPLETE, *body_tokens],
        )
    ]
    round_kyoku = _round_kyoku_by_index(game, tokenizer.seat_count)
    for viewer_player in range(tokenizer.seat_count):
        transformed: list[str] = [
            *rule_prefix,
            imperfect_view_token(viewer_player),
        ]
        cursor = len(rule_prefix)
        last_viewer_round_seat: int | None = None
        for trace in event_traces:
            trace_round_kyoku = round_kyoku.get(trace.round_index)
            if trace_round_kyoku is None:
                raise ValueError(f"missing qipai.jushu for round_index={trace.round_index}")
            viewer_round_seat = _viewer_round_seat(
                viewer_player,
                initial_qijia=tokenizer.initial_qijia,
                seat_count=tokenizer.seat_count,
                round_kyoku=trace_round_kyoku,
            )
            transformed.extend(
                _transform_event_tokens(complete_tokens[cursor : trace.start], viewer_round_seat, "gap")
            )
            transformed.extend(
                _transform_event_tokens(
                    complete_tokens[trace.start : trace.end],
                    viewer_round_seat,
                    trace.event_key,
                )
            )
            cursor = trace.end
            last_viewer_round_seat = viewer_round_seat
        transformed.extend(_transform_event_tokens(complete_tokens[cursor:], last_viewer_round_seat, "gap"))
        views.append(
            TokenizedGameView(
                view_type=VIEW_IMPERFECT,
                viewer_seat=viewer_player,
                tokens=transformed,
            )
        )
    return views
