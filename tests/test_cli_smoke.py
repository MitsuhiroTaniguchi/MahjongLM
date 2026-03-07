from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import pymahjong  # noqa: F401
import scripts.tokenize_tenhou as tokenize_tenhou

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


def _run_cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), check=False)


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


def test_cli_parallel_workers_preserve_output(tmp_path: Path) -> None:
    zip_path = tmp_path / "data2023.zip"
    serial_out = tmp_path / "tokens_serial.jsonl"
    parallel_out = tmp_path / "tokens_parallel.jsonl"

    members = {
        "g0.json": minimal_game([qipai_event(jushu=0), pingju_event()]),
        "g1.json": minimal_game([qipai_event(jushu=1), pingju_event()]),
        "g2.json": minimal_game([qipai_event(jushu=2), pingju_event()]),
        "g3.json": minimal_game([qipai_event(jushu=3), pingju_event()]),
    }
    _write_zip(zip_path, members)

    serial = _run_cli(
        "--zip-path",
        str(zip_path),
        "--output",
        str(serial_out),
        "--progress-every",
        "0",
        "--workers",
        "1",
    )
    parallel = _run_cli(
        "--zip-path",
        str(zip_path),
        "--output",
        str(parallel_out),
        "--progress-every",
        "0",
        "--workers",
        "2",
        "--chunk-size",
        "2",
    )

    assert serial.returncode == 0, serial.stderr
    assert parallel.returncode == 0, parallel.stderr
    assert serial_out.read_text(encoding="utf-8") == parallel_out.read_text(encoding="utf-8")


def test_cli_main_falls_back_to_serial_when_parallel_executor_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_path = tmp_path / "data2023.zip"
    out_path = tmp_path / "tokens.jsonl"
    _write_zip(zip_path, {"g0.json": minimal_game([qipai_event(), pingju_event()])})

    monkeypatch.setattr(
        tokenize_tenhou,
        "parse_args",
        lambda: Namespace(
            zip_path=zip_path,
            all_years=False,
            zip_glob=None,
            output=out_path,
            max_games=None,
            start_index=0,
            progress_every=0,
            strict=False,
            workers=2,
            chunk_size=2,
        ),
    )

    def fail_parallel(*_args, **_kwargs):
        raise PermissionError("sandbox blocked semaphores")

    monkeypatch.setattr(tokenize_tenhou, "_tokenize_zip_parallel", fail_parallel)

    result = tokenize_tenhou.main()

    assert result == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


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
    assert "skip: data2023.zip:game_broken.json" in result.stderr
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


def test_cli_resolves_user_relative_paths_from_cwd(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    zip_path = workdir / "local.zip"
    out_path = workdir / "tokens.jsonl"

    valid_game = minimal_game([qipai_event(), pingju_event()])
    _write_zip(zip_path, {"game_valid.json": valid_game})

    result = _run_cli(
        "--zip-path",
        "local.zip",
        "--output",
        "tokens.jsonl",
        "--progress-every",
        "0",
        cwd=workdir,
    )

    assert result.returncode == 0, result.stderr
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["source_zip"] == "local.zip"


def test_cli_reports_missing_zip_without_traceback(tmp_path: Path) -> None:
    out_path = tmp_path / "tokens.jsonl"

    result = _run_cli("--zip-path", str(tmp_path / "missing.zip"), "--output", str(out_path))

    assert result.returncode == 2
    assert "zip not found:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_reports_invalid_zip_without_traceback(tmp_path: Path) -> None:
    bad_zip = tmp_path / "broken.zip"
    bad_zip.write_text("not a zip", encoding="utf-8")
    out_path = tmp_path / "tokens.jsonl"

    result = _run_cli("--zip-path", str(bad_zip), "--output", str(out_path))

    assert result.returncode == 2
    assert "invalid zip file:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_skips_malformed_round_structure_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    out_path = tmp_path / "tokens.jsonl"
    _write_zip(
        zip_path,
        {
            "good.json": minimal_game([qipai_event(), pingju_event()]),
            "bad.json": {"log": [[{}]]},
        },
    )

    result = _run_cli("--zip-path", str(zip_path), "--output", str(out_path), "--progress-every", "0")

    assert result.returncode == 0
    assert "skip: games.zip:bad.json" in result.stderr
    assert "Traceback" not in result.stderr
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_cli_skips_missing_or_duplicate_qipai_rounds_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    out_path = tmp_path / "tokens.jsonl"
    _write_zip(
        zip_path,
        {
            "good.json": minimal_game([qipai_event(), pingju_event()]),
            "missing_qipai.json": {"log": [[{"zimo": {"l": 0, "p": "m1"}}]]},
            "double_qipai.json": {"log": [[qipai_event(), qipai_event(jushu=1)]]},
        },
    )

    result = _run_cli("--zip-path", str(zip_path), "--output", str(out_path), "--progress-every", "0")

    assert result.returncode == 0
    assert "skip: games.zip:missing_qipai.json" in result.stderr
    assert "skip: games.zip:double_qipai.json" in result.stderr
    assert "Traceback" not in result.stderr
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_cli_skips_invalid_payload_shapes_without_traceback(tmp_path: Path) -> None:
    zip_path = tmp_path / "games.zip"
    out_path = tmp_path / "tokens.jsonl"
    _write_zip(
        zip_path,
        {
            "good.json": minimal_game([qipai_event(), pingju_event()]),
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
            "empty.json": {"log": []},
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

    result = _run_cli("--zip-path", str(zip_path), "--output", str(out_path), "--progress-every", "0")

    assert result.returncode == 0
    assert "skip: games.zip:bad_qipai.json" in result.stderr
    assert "skip: games.zip:bad_hand_count.json" in result.stderr
    assert "skip: games.zip:bad_round_meta.json" in result.stderr
    assert "skip: games.zip:empty.json" in result.stderr
    assert "skip: games.zip:fractional_seat.json" in result.stderr
    assert "skip: games.zip:bad_actor.json" in result.stderr
    assert "skip: games.zip:bad_result.json" in result.stderr
    assert "skip: games.zip:bad_turn.json" in result.stderr
    assert "skip: games.zip:after_end.json" in result.stderr
    assert "skip: games.zip:bad_kaigang.json" in result.stderr
    assert "Traceback" not in result.stderr
    assert "done: tokenized=1 skipped=11" in result.stdout
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_cli_writes_expected_record_shape_for_gzip_output(tmp_path: Path) -> None:
    zip_path = tmp_path / "data2023.zip"
    out_path = tmp_path / "tokens.jsonl.gz"

    valid_game = minimal_game([qipai_event(), pingju_event()])
    _write_zip(zip_path, {"game_valid.json": valid_game})

    result = _run_cli("--zip-path", str(zip_path), "--output", str(out_path), "--progress-every", "0")

    assert result.returncode == 0, result.stderr
    import gzip

    with gzip.open(out_path, "rt", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"source_zip", "game_id", "tokens"}
    assert record["source_zip"] == str(zip_path)
    assert record["game_id"] == "game_valid.json"
    assert isinstance(record["tokens"], list)
    assert record["tokens"][0] == "game_start"
    assert record["tokens"][-1] == "game_end"
