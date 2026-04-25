from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import pymahjong  # noqa: F401

from tenhou_tokenizer import TenhouTokenizer
from tests.dataset_sample import DATASET_2023, get_dataset_2023_sample_zip

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tokenize_tenhou.py"


@pytest.mark.slow
def test_cli_smoke_with_local_2023_dataset(tmp_path: Path) -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")
    sample_zip = get_dataset_2023_sample_zip()

    output = tmp_path / "tokens_2023_smoke.jsonl"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--zip-path",
        str(sample_zip),
        "--output",
        str(output),
        "--max-games",
        "20",
        "--progress-every",
        "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=False)

    assert result.returncode == 0, result.stderr
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0

    first = json.loads(lines[0])
    assert "tokens" in first
    assert first["source_zip"] == str(sample_zip)


@pytest.mark.slow
@pytest.mark.parametrize(
    "game_id",
    [
        "2023/2023091610gm-00a9-0000-bdcf1ac5.txt",
        "2023/2023092616gm-00a9-0000-cedf6b55.txt",
        "2023/2023081200gm-00a9-0000-b14c9329.txt",
        "2023/2023102817gm-00a9-0000-3abd1803.txt",
    ],
)
def test_real_dataset_regression_games_tokenize(game_id: str) -> None:
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    with zipfile.ZipFile(get_dataset_2023_sample_zip()) as zf:
        game = json.load(zf.open(game_id))

    tokens = TenhouTokenizer().tokenize_game(game)

    assert tokens[0].startswith("rule_player_")
    assert "game_start" in tokens
    assert tokens[-1] == "game_end"
