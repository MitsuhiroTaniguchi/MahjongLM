from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer import iter_tokenized_games
from tenhou_tokenizer.engine import TenhouTokenizer, TokenizeError
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event


def _write_zip(path: Path, members: dict[str, object]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            if isinstance(payload, str):
                zf.writestr(name, payload)
            else:
                zf.writestr(name, json.dumps(payload, ensure_ascii=False))


def test_iter_tokenized_games_respects_start_index_and_max_games(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(
        zip_path,
        {
            "g0.json": minimal_game([qipai_event(jushu=0), pingju_event()]),
            "g1.json": minimal_game([qipai_event(jushu=1), pingju_event()]),
            "g2.json": minimal_game([qipai_event(jushu=2), pingju_event()]),
        },
    )

    rows = list(iter_tokenized_games(str(zip_path), start_index=1, max_games=1))

    assert [name for name, _tokens in rows] == ["g1.json"]
    assert rows[0][1][0] == "game_start"
    assert rows[0][1][-1] == "game_end"


def test_iter_tokenized_games_rejects_empty_log_games(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(zip_path, {"empty.json": {"log": []}})

    with pytest.raises(TokenizeError):
        list(iter_tokenized_games(str(zip_path)))


def test_iter_tokenized_games_raises_for_invalid_json_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(
        zip_path,
        {
            "good.json": minimal_game([qipai_event(), pingju_event()]),
            "bad.json": "{not json}",
        },
    )

    iterator = iter_tokenized_games(str(zip_path), start_index=1)

    with pytest.raises(json.JSONDecodeError):
        list(iterator)


def test_tokenize_game_rejects_non_dict_game() -> None:
    with pytest.raises(TokenizeError):
        TenhouTokenizer().tokenize_game(["not", "a", "dict"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "game",
    [
        {"log": "notalist"},
        {"log": [[]]},
        {"log": [[{}]]},
        {"log": [None]},
        {"log": [["notadict"]]},
        {"log": [[{"zimo": {"l": 0, "p": "m1"}}]]},
        {
            "log": [
                [
                    qipai_event(),
                    qipai_event(jushu=1),
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000, 25000, 25000],
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": -0.1,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "x1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": "east",
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m111110p1234567"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": -1,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": ["25000"] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {"log": [[qipai_event(), {"zimo": {"l": 1.9, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 4, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"dapai": {"l": 4, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 4, "fenpei": [0, 0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 0, "baojia": 4, "fenpei": [0, 0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 0, "fenpei": [0, "1000", -500, -500]}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 0, "fenpei": [0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"pingju": {"name": "流局", "fenpei": [0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"pingju": {"name": "流局", "fenpei": [0, 0, True, 0]}}]]},
        {"log": [[qipai_event(), {"pingju": {"name": 123, "fenpei": [0, 0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"zimo": "bad"}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": 1}}]]},
        {"log": [[qipai_event(), {"dapai": "bad"}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": "bad"}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": 1}}]]},
        {"log": [[qipai_event(), {"fulou": {"l": 1, "m": 123}}]]},
        {"log": [[qipai_event(), {"gang": {"l": 0, "m": 123}}]]},
        {"log": [[qipai_event(), {"hule": "bad"}]]},
        {"log": [[qipai_event(), {"kaigang": {"baopai": "x1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1"}}, {"kaigang": {"baopai": "p1"}}]]},
        {
            "log": [[
                qipai_event(hands=["m123456789p1234", "m111p12345s123z11", "m123456789p1234", "m123456789p1234"]),
                {"zimo": {"l": 0, "p": "s1"}},
                {"dapai": {"l": 0, "p": "m1"}},
                {"kaigang": {"baopai": "p1"}},
            ]]
        },
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"zimo": {"l": 1, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 1, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1_"}}, {"zimo": {"l": 0, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1_"}}, {"zimo": {"l": 2, "p": "m1"}}]]},
        {
            "log": [[
                qipai_event(hands=["m1456789p1234s11", "m23p123456s123z11", "m123456789p1234", "m123456789p1234"]),
                {"zimo": {"l": 0, "p": "s2"}},
                {"dapai": {"l": 0, "p": "m1"}},
                {"fulou": {"l": 1, "m": "m1-23"}},
                {"dapai": {"l": 2, "p": "m1"}},
            ]]
        },
        {"log": [[qipai_event(), pingju_event(), {"zimo": {"l": 0, "p": "m1"}}]]},
        {"log": [[qipai_event(), pingju_event(), {"hule": "bad"}]]},
    ],
)
def test_iter_tokenized_games_raises_for_malformed_round_structure(tmp_path: Path, game: dict) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(zip_path, {"bad.json": game})

    with pytest.raises(TokenizeError):
        list(iter_tokenized_games(str(zip_path)))


@pytest.mark.parametrize(
    "game",
    [
        {"log": "notalist"},
        {"log": [[]]},
        {"log": [[{}]]},
        {"log": [None]},
        {"log": [["notadict"]]},
        {"log": [[{"dapai": {"l": 0, "p": "m1"}}]]},
        {
            "log": [
                [
                    qipai_event(),
                    qipai_event(jushu=1),
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000, 25000, 25000],
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": -0.1,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "x1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": "east",
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m111110p1234567"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": -1,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": [25000] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {
            "log": [
                [
                    {
                        "qipai": {
                            "zhuangfeng": 0,
                            "jushu": 0,
                            "changbang": 0,
                            "lizhibang": 0,
                            "defen": ["25000"] * 4,
                            "baopai": "m1",
                            "shoupai": ["m123456789p1234"] * 4,
                        }
                    }
                ]
            ]
        },
        {"log": [[qipai_event(), {"zimo": {"l": 1.9, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 4, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"dapai": {"l": 4, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 4, "fenpei": [0, 0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 0, "baojia": 4, "fenpei": [0, 0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 0, "fenpei": [0, "1000", -500, -500]}}]]},
        {"log": [[qipai_event(), {"hule": {"l": 0, "fenpei": [0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"pingju": {"name": "流局", "fenpei": [0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"pingju": {"name": "流局", "fenpei": [0, 0, True, 0]}}]]},
        {"log": [[qipai_event(), {"pingju": {"name": 123, "fenpei": [0, 0, 0, 0]}}]]},
        {"log": [[qipai_event(), {"zimo": "bad"}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": 1}}]]},
        {"log": [[qipai_event(), {"dapai": "bad"}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": "bad"}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": 1}}]]},
        {"log": [[qipai_event(), {"fulou": {"l": 1, "m": 123}}]]},
        {"log": [[qipai_event(), {"gang": {"l": 0, "m": 123}}]]},
        {"log": [[qipai_event(), {"hule": "bad"}]]},
        {"log": [[qipai_event(), {"kaigang": {"baopai": "x1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1"}}, {"kaigang": {"baopai": "p1"}}]]},
        {
            "log": [[
                qipai_event(hands=["m123456789p1234", "m111p12345s123z11", "m123456789p1234", "m123456789p1234"]),
                {"zimo": {"l": 0, "p": "s1"}},
                {"dapai": {"l": 0, "p": "m1"}},
                {"kaigang": {"baopai": "p1"}},
            ]]
        },
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"zimo": {"l": 1, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 1, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1_"}}, {"zimo": {"l": 0, "p": "m1"}}]]},
        {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1_"}}, {"zimo": {"l": 2, "p": "m1"}}]]},
        {
            "log": [[
                qipai_event(hands=["m1456789p1234s11", "m23p123456s123z11", "m123456789p1234", "m123456789p1234"]),
                {"zimo": {"l": 0, "p": "s2"}},
                {"dapai": {"l": 0, "p": "m1"}},
                {"fulou": {"l": 1, "m": "m1-23"}},
                {"dapai": {"l": 2, "p": "m1"}},
            ]]
        },
        {"log": [[qipai_event(), pingju_event(), {"zimo": {"l": 0, "p": "m1"}}]]},
        {"log": [[qipai_event(), pingju_event(), {"hule": "bad"}]]},
    ],
)
def test_tokenize_game_rejects_malformed_round_structure(game: dict) -> None:
    with pytest.raises(TokenizeError):
        TenhouTokenizer().tokenize_game(game)
