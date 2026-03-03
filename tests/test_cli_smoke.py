from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

if importlib.util.find_spec("pymahjong") is None:
    pytest.skip("pymahjong is required for CLI tests", allow_module_level=True)

from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tokenize_tenhou.py"


def _write_zip(path: Path, members: dict[str, object]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            if isinstance(payload, str):
                zf.writestr(name, payload)
            else:
                zf.writestr(name, json.dumps(payload, ensure_ascii=False))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=False)


def test_cli_tokenizes_single_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "data2023.zip"
    out_path = tmp_path / "tokens.jsonl"

    valid_game = minimal_game([qipai_event(), pingju_event()])
    _write_zip(zip_path, {"game_valid.json": valid_game})

    result = _run_cli("--zip-path", str(zip_path), "--output", str(out_path), "--progress-every", "0")

    assert result.returncode == 0, result.stderr
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["game_id"] == "game_valid.json"
    assert record["source_zip"] == str(zip_path)


def test_cli_strict_fails_when_any_game_is_skipped(tmp_path: Path) -> None:
    zip_path = tmp_path / "data2023.zip"
    out_path = tmp_path / "tokens.jsonl"

    valid_game = minimal_game([qipai_event(), pingju_event()])
    _write_zip(
        zip_path,
        {
            "game_valid.json": valid_game,
            "game_broken.json": "{not json}",
        },
    )

    result = _run_cli(
        "--zip-path",
        str(zip_path),
        "--output",
        str(out_path),
        "--progress-every",
        "0",
        "--strict",
    )

    assert result.returncode == 1
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_cli_all_years_glob(tmp_path: Path) -> None:
    zip1 = tmp_path / "data2022.zip"
    zip2 = tmp_path / "data2023.zip"
    out_path = tmp_path / "tokens_all.jsonl"

    game1 = minimal_game([qipai_event(jushu=0), pingju_event()])
    game2 = minimal_game([qipai_event(jushu=1), pingju_event()])
    _write_zip(zip1, {"g1.json": game1})
    _write_zip(zip2, {"g2.json": game2})

    result = _run_cli(
        "--all-years",
        "--zip-glob",
        str(tmp_path / "data*.zip"),
        "--output",
        str(out_path),
        "--progress-every",
        "0",
    )

    assert result.returncode == 0, result.stderr
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert {line["source_zip"] for line in lines} == {str(zip1), str(zip2)}
