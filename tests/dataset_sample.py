from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_2023 = ROOT / "data" / "raw" / "tenhou" / "data2023.zip"
DATASET_2023_SAMPLE = ROOT / ".pytest_cache" / "data2023_sample.zip"
DATASET_2023_CURATED_VERSION = 2
DATASET_2023_CURATED = ROOT / ".pytest_cache" / f"data2023_curated_v{DATASET_2023_CURATED_VERSION}.zip"
DATASET_2023_SAMPLE_GAMES = 500
DATASET_2023_REGRESSION_GAME_IDS = (
    "2023/2023091610gm-00a9-0000-bdcf1ac5.txt",
    "2023/2023092616gm-00a9-0000-cedf6b55.txt",
    "2023/2023081200gm-00a9-0000-b14c9329.txt",
    "2023/2023102817gm-00a9-0000-3abd1803.txt",
)
DATASET_2023_CURATED_GAME_IDS = (
    "2023/2023021622gm-00a9-0000-75ec253d.txt",
    "2023/2023021117gm-00a9-0000-d0a6d793.txt",
    "2023/2023052310gm-00a9-0000-2faa0711.txt",
    "2023/2023070721gm-00a9-0000-d6a86063.txt",
    "2023/2023073004gm-00a9-0000-ddb6dc93.txt",
    "2023/2023080301gm-00a9-0000-d28602aa.txt",
    "2023/2023081821gm-00a9-0000-7cf4f26e.txt",
    "2023/2023091610gm-00a9-0000-bdcf1ac5.txt",
    "2023/2023091810gm-00a9-0000-1428c108.txt",
    "2023/2023112221gm-00a9-0000-6e1a680f.txt",
    *DATASET_2023_REGRESSION_GAME_IDS,
)


def _unique_preserve_order(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


DATASET_2023_CURATED_GAME_IDS = _unique_preserve_order(DATASET_2023_CURATED_GAME_IDS)


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


@lru_cache(maxsize=1)
def get_dataset_2023_curated_zip() -> Path:
    if not DATASET_2023.exists():
        return DATASET_2023

    DATASET_2023_CURATED.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_2023_CURATED.exists() and DATASET_2023_CURATED.stat().st_mtime >= DATASET_2023.stat().st_mtime:
        return DATASET_2023_CURATED

    with zipfile.ZipFile(DATASET_2023) as src, zipfile.ZipFile(
        DATASET_2023_CURATED,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as dst:
        for name in _selected_names([], DATASET_2023_CURATED_GAME_IDS):
            dst.writestr(name, src.read(name))
    return DATASET_2023_CURATED
