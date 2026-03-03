from __future__ import annotations

from typing import List, Optional

DEFAULT_HANDS = [
    "m123456789p1234",
    "m123456789p1234",
    "m123456789p1234",
    "m123456789p1234",
]


def qipai_payload(jushu: int = 0, hands: Optional[List[str]] = None) -> dict:
    return {
        "zhuangfeng": 0,
        "jushu": jushu,
        "changbang": 0,
        "lizhibang": 0,
        "defen": [25000, 25000, 25000, 25000],
        "baopai": "m1",
        "shoupai": hands or list(DEFAULT_HANDS),
    }


def qipai_event(jushu: int = 0, hands: Optional[List[str]] = None) -> dict:
    return {"qipai": qipai_payload(jushu=jushu, hands=hands)}


def pingju_event(name: str = "流局", fenpei: Optional[List[int]] = None) -> dict:
    return {"pingju": {"name": name, "fenpei": fenpei or [0, 0, 0, 0]}}


def minimal_game(round_events: List[dict]) -> dict:
    return {"log": [round_events]}
