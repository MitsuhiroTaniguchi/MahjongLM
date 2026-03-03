from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tokenize_tenhou.py"
DATASET_2023 = ROOT / "data" / "raw" / "tenhou" / "data2023.zip"


@pytest.mark.slow
def test_cli_smoke_with_local_2023_dataset(tmp_path: Path) -> None:
    if importlib.util.find_spec("pymahjong") is None:
        pytest.skip("pymahjong is required")
    if not DATASET_2023.exists():
        pytest.skip(f"missing dataset: {DATASET_2023}")

    output = tmp_path / "tokens_2023_smoke.jsonl"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--zip-path",
        str(DATASET_2023),
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
    assert first["source_zip"] == str(DATASET_2023)
