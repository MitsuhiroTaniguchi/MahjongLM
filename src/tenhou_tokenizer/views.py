from __future__ import annotations

from dataclasses import dataclass
from .engine import TenhouTokenizer
from .viewspec import TOKEN_VIEW_COMPLETE, TOKEN_VIEW_IMPERFECT, VIEW_COMPLETE, VIEW_IMPERFECT


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
    "TENBO_500",
    "TENBO_1000",
    "TENBO_2000",
    "TENBO_3000",
    "TENBO_5000",
    "TENBO_10000",
    "TENBO_20000",
    "TENBO_30000",
}


@dataclass(frozen=True)
class TokenizedGameView:
    view_type: str
    viewer_seat: int | None
    tokens: list[str]


def _hidden_hand_token(seat: int) -> str:
    return f"hidden_hand_{seat}"


def _hidden_draw_token(seat: int) -> str:
    return f"draw_{seat}_hidden"


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
        out.append(tokens[idx])
        idx += 1
        for _ in range(13):
            if viewer_seat == seat:
                out.append(tokens[idx])
            else:
                out.append(_hidden_hand_token(seat))
            idx += 1

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
    if not event_traces:
        return [
            TokenizedGameView(
                view_type=VIEW_COMPLETE,
                viewer_seat=None,
                tokens=[complete_tokens[0], TOKEN_VIEW_COMPLETE, *complete_tokens[1:]],
            )
        ]

    views = [
        TokenizedGameView(
            view_type=VIEW_COMPLETE,
            viewer_seat=None,
            tokens=[complete_tokens[0], TOKEN_VIEW_COMPLETE, *complete_tokens[1:]],
        )
    ]
    for viewer_seat in range(tokenizer.seat_count):
        transformed: list[str] = [complete_tokens[0], TOKEN_VIEW_IMPERFECT]
        cursor = 1
        for trace in event_traces:
            transformed.extend(_transform_event_tokens(complete_tokens[cursor : trace.start], viewer_seat, "gap"))
            transformed.extend(
                _transform_event_tokens(
                    complete_tokens[trace.start : trace.end],
                    viewer_seat,
                    trace.event_key,
                )
            )
            cursor = trace.end
        transformed.extend(_transform_event_tokens(complete_tokens[cursor:], viewer_seat, "gap"))
        views.append(
            TokenizedGameView(
                view_type=VIEW_IMPERFECT,
                viewer_seat=viewer_seat,
                tokens=transformed,
            )
        )
    return views
