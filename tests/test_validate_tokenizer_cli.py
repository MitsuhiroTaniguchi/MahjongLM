from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pymahjong  # noqa: F401

from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_tokenizer.py"


def _write_zip(path: Path, members: dict[str, object]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            if isinstance(payload, str):
                zf.writestr(name, payload)
            else:
                zf.writestr(name, json.dumps(payload, ensure_ascii=False))


def _run_validator(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), check=False)


def test_validator_reports_missing_zip_without_traceback(tmp_path: Path) -> None:
    result = _run_validator("--zip-path", str(tmp_path / "missing.zip"), "--progress-every", "0")

    assert result.returncode == 2
    assert "zip not found:" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_reports_invalid_zip_without_traceback(tmp_path: Path) -> None:
    bad_zip = tmp_path / "broken.zip"
    bad_zip.write_text("not a zip", encoding="utf-8")

    result = _run_validator("--zip-path", str(bad_zip), "--progress-every", "0")

    assert result.returncode == 2
    assert "invalid zip file:" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_reports_invalid_json_member_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(zip_path, {"bad.json": "{not json}"})

    result = _run_validator("--zip-path", str(zip_path), "--progress-every", "0")

    assert result.returncode == 1
    assert "validation failed: games.zip:bad.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_reports_invariant_failure_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(
        zip_path,
        {
            "unknown_event.json": {
                "log": [
                    [
                        qipai_event(),
                        {"mystery": {"x": 1}},
                    ]
                ]
            }
        },
    )

    result = _run_validator("--zip-path", str(zip_path), "--progress-every", "0")

    assert result.returncode == 1
    assert "validation failed: games.zip:unknown_event.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_reports_malformed_round_structure_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(zip_path, {"bad.json": {"log": [[{}]]}})

    result = _run_validator("--zip-path", str(zip_path), "--progress-every", "0")

    assert result.returncode == 1
    assert "validation failed: games.zip:bad.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_reports_missing_or_duplicate_qipai_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(
        zip_path,
        {
            "missing_qipai.json": {"log": [[{"zimo": {"l": 0, "p": "m1"}}]]},
            "double_qipai.json": {"log": [[qipai_event(), qipai_event(jushu=1)]]},
        },
    )

    result = _run_validator("--zip-path", str(zip_path), "--progress-every", "0")

    assert result.returncode == 1
    assert "validation failed: games.zip:" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_reports_invalid_payload_shapes_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(
        zip_path,
        {
            "bad_qipai.json": {
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
            "bad_hand_count.json": {
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
                                "shoupai": ["m123"] * 4,
                            }
                        }
                    ]
                ]
            },
            "bad_round_meta.json": {
                "log": [
                    [
                        {
                            "qipai": {
                                "zhuangfeng": "east",
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
            "fractional_seat.json": {"log": [[qipai_event(), {"zimo": {"l": 1.9, "p": "m1"}}]]},
            "bad_actor.json": {"log": [[qipai_event(), {"zimo": {"l": 4, "p": "m1"}}]]},
            "bad_result.json": {"log": [[qipai_event(), {"hule": {"l": 0, "fenpei": [0, 0, 0]}}]]},
            "bad_turn.json": {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"zimo": {"l": 1, "p": "m1"}}]]},
            "after_end.json": {"log": [[qipai_event(), pingju_event(), {"zimo": {"l": 0, "p": "m1"}}]]},
            "bad_kaigang.json": {"log": [[qipai_event(), {"zimo": {"l": 0, "p": "m1"}}, {"dapai": {"l": 0, "p": "m1"}}, {"kaigang": {"baopai": "p1"}}]]},
            "bad_kaigang_offered_only.json": {
                "log": [[
                    qipai_event(hands=["m123456789p1234", "m111p12345s123z11", "m123456789p1234", "m123456789p1234"]),
                    {"zimo": {"l": 0, "p": "s1"}},
                    {"dapai": {"l": 0, "p": "m1"}},
                    {"kaigang": {"baopai": "p1"}},
                ]]
            },
        },
    )

    result = _run_validator("--zip-path", str(zip_path), "--progress-every", "0")

    assert result.returncode == 1
    assert "validation failed: games.zip:" in result.stderr
    assert "Traceback" not in result.stderr


def test_validator_rejects_empty_log_game(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    _write_zip(
        zip_path,
        {
            "empty.json": {"log": []},
            "normal.json": minimal_game([qipai_event(), pingju_event()]),
        },
    )

    result = _run_validator("--zip-path", str(zip_path), "--progress-every", "0")

    assert result.returncode == 1
    assert "validation failed: games.zip:empty.json" in result.stderr
    assert "Traceback" not in result.stderr
