from __future__ import annotations

from typing import List, Optional

DEFAULT_HANDS = [
    "m123456789p1234",
    "m123456789p1234",
    "m123456789p1234",
    "m123456789p1234",
]


def qipai_payload(jushu: int = 0, hands: Optional[List[str]] = None, seat_count: int = 4) -> dict:
    default_hands = list(DEFAULT_HANDS[:seat_count])
    return {
        "zhuangfeng": 0,
        "jushu": jushu,
        "changbang": 0,
        "lizhibang": 0,
        "defen": [25000] * seat_count,
        "baopai": "m1",
        "shoupai": hands or default_hands,
    }


def qipai_event(jushu: int = 0, hands: Optional[List[str]] = None, seat_count: int = 4) -> dict:
    return {"qipai": qipai_payload(jushu=jushu, hands=hands, seat_count=seat_count)}


def pingju_event(name: str = "流局", fenpei: Optional[List[int]] = None, seat_count: int = 4) -> dict:
    return {"pingju": {"name": name, "fenpei": fenpei or ([0] * seat_count)}}


def minimal_game(round_events: List[dict]) -> dict:
    return {"log": [round_events]}
