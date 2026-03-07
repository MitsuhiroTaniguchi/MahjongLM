from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_2023 = ROOT / "data" / "raw" / "tenhou" / "data2023.zip"
DATASET_2023_SAMPLE = ROOT / ".pytest_cache" / "data2023_sample.zip"
DATASET_2023_SAMPLE_GAMES = 500
DATASET_2023_REGRESSION_GAME_IDS = (
    "2023/2023091610gm-00a9-0000-bdcf1ac5.txt",
    "2023/2023092616gm-00a9-0000-cedf6b55.txt",
    "2023/2023081200gm-00a9-0000-b14c9329.txt",
    "2023/2023102817gm-00a9-0000-3abd1803.txt",
)


def _selected_names(names: list[str], extra_names: Iterable[str]) -> list[str]:
    selected = list(names[:DATASET_2023_SAMPLE_GAMES])
    seen = set(selected)
    for name in extra_names:
        if name not in seen:
            selected.append(name)
            seen.add(name)
    return selected


@lru_cache(maxsize=1)
def get_dataset_2023_sample_zip() -> Path:
    if not DATASET_2023.exists():
        return DATASET_2023

    DATASET_2023_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_2023_SAMPLE.exists() and DATASET_2023_SAMPLE.stat().st_mtime >= DATASET_2023.stat().st_mtime:
        return DATASET_2023_SAMPLE

    with zipfile.ZipFile(DATASET_2023) as src, zipfile.ZipFile(
        DATASET_2023_SAMPLE,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as dst:
        for name in _selected_names(src.namelist(), DATASET_2023_REGRESSION_GAME_IDS):
            dst.writestr(name, src.read(name))
    return DATASET_2023_SAMPLE
