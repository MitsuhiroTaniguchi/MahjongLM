from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parents[1] / "data" / "raw" / "tenhou"
LOG_DIR = DATA_ROOT / "logs"
RAW_DIR = DATA_ROOT / "paifu_raw"
JSON_DIR = DATA_ROOT / "paifu_json"
TOKENIZED_DIR = DATA_ROOT / "tokenized"
HF_DATASETS_DIR = SCRIPT_DIR.parents[1] / "data" / "huggingface_datasets"
CONVERT = SCRIPT_DIR / "convert.pl"

SRC_DIR = SCRIPT_DIR.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tenhou_tokenizer import (
    TokenizedGameView,
    Vocabulary,
    load_token_ids,
    save_hf_tokenizer_assets,
    save_token_ids,
    save_year_hf_dataset,
    tokenize_game_views,
)
from tenhou_tokenizer.viewspec import view_artifact_name

TARGET_PLAYERS = {"三", "四"}
TARGET_ROUNDS = {"東", "南"}
TARGET_SPEEDS = {"－", "速"}
LINE_RE = re.compile(
    r"^\d{2}:\d{2}\s+\|\s+\d+\s+\|\s+(?P<title>[^|]+?)\s+\|\s+"
    r'<a href="https?://tenhou\.net/0/\?log=(?P<log_id>[^"]+)">牌譜</a>'
)


def iter_archive_zips() -> list[tuple[int, Path]]:
    archives: list[tuple[int, Path]] = []
    for path in sorted(LOG_DIR.glob("scraw*.zip")):
        match = re.search(r"scraw(\d{4})\.zip$", path.name)
        if match:
            archives.append((int(match.group(1)), path))
    return archives


def iter_archive_lines(zip_path: Path) -> list[str]:
    lines: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for filename in sorted(name for name in zip_ref.namelist() if "scc" in name):
            with zip_ref.open(filename) as file:
                with gzip.open(file, "rt", encoding="utf-8", errors="replace") as gz_file:
                    lines.extend(gz_file.read().splitlines())
    return lines


def parse_log_id(line: str) -> str | None:
    match = LINE_RE.search(line)
    if not match:
        return None
    title = match.group("title").strip()
    if len(title) != 6:
        return None
    if title[0] not in TARGET_PLAYERS:
        return None
    if title[1] != "鳳":
        return None
    if title[2] not in TARGET_ROUNDS:
        return None
    if title[3:5] != "喰赤":
        return None
    if title[5] not in TARGET_SPEEDS:
        return None
    return match.group("log_id")


def fetch_log_text(session: requests.Session, log_id: str) -> str:
    response = session.get(
        f"https://tenhou.net/0/log/?{log_id}",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(text)
            temp_path = Path(f.name)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def convert_raw_to_json(raw_path: Path, json_path: Path) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=json_path.parent,
            prefix=f".{json_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            subprocess.run(
                ["perl", "-T", str(CONVERT), str(raw_path)],
                check=True,
                text=True,
                stdout=f,
            )
        temp_path.replace(json_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def has_downloaded_raw(raw_path: Path) -> bool:
    return raw_path.is_file() and raw_path.stat().st_size > 0


def has_valid_json(json_path: Path) -> bool:
    if not json_path.is_file() or json_path.stat().st_size == 0:
        return False
    try:
        game = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return isinstance(game, dict) and isinstance(game.get("log"), list)


def has_valid_tokenized(tokenized_path: Path, vocab: Vocabulary) -> bool:
    if not tokenized_path.is_file() or tokenized_path.stat().st_size == 0:
        return False
    try:
        input_ids = load_token_ids(tokenized_path, expected_vocab_fingerprint=vocab.fingerprint)
    except (OSError, ValueError):
        return False
    return all(isinstance(token_id, int) and token_id >= 0 for token_id in input_ids)


def load_json_game(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def view_output_paths(tokenized_year_dir: Path, game: dict, log_id: str) -> list[tuple[TokenizedGameView, Path]]:
    views = tokenize_game_views(game)
    paths: list[tuple[TokenizedGameView, Path]] = []
    for view in views:
        paths.append(
            (
                view,
                tokenized_year_dir
                / view_artifact_name(log_id, view.view_type, view.viewer_seat),
            )
        )
    return paths


def has_all_valid_tokenized_views(
    tokenized_year_dir: Path,
    vocab: Vocabulary,
    game: dict,
    log_id: str,
) -> bool:
    return all(has_valid_tokenized(path, vocab) for _view, path in view_output_paths(tokenized_year_dir, game, log_id))


def tokenize_game_views_to_files(
    vocab: Vocabulary,
    game: dict,
    tokenized_year_dir: Path,
    log_id: str,
) -> None:
    for view, path in view_output_paths(tokenized_year_dir, game, log_id):
        save_token_ids(path, vocab.encode(view.tokens), vocab_fingerprint=vocab.fingerprint)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    archives = iter_archive_zips()
    if not archives:
        raise FileNotFoundError(f"no scraw*.zip found under {LOG_DIR}")
    if not CONVERT.is_file():
        raise FileNotFoundError(f"convert.pl not found: {CONVERT}")

    vocab = Vocabulary.load()
    save_hf_tokenizer_assets()
    with requests.Session() as session:
        for year, zip_path in archives:
            raw_year_dir = RAW_DIR / str(year)
            json_year_dir = JSON_DIR / str(year)
            tokenized_year_dir = TOKENIZED_DIR / str(year)
            hf_dataset_year_dir = HF_DATASETS_DIR / str(year)
            raw_year_dir.mkdir(parents=True, exist_ok=True)
            json_year_dir.mkdir(parents=True, exist_ok=True)
            tokenized_year_dir.mkdir(parents=True, exist_ok=True)
            year_updated = False

            for line in tqdm(iter_archive_lines(zip_path), desc=str(year)):
                log_id = parse_log_id(line)
                if not log_id:
                    continue

                raw_path = raw_year_dir / f"{log_id}.txt"
                json_path = json_year_dir / f"{log_id}.json"

                try:
                    if not has_valid_json(json_path):
                        if not has_downloaded_raw(raw_path):
                            atomic_write_text(raw_path, fetch_log_text(session, log_id))
                            time.sleep(0.1)
                        convert_raw_to_json(raw_path, json_path)
                        year_updated = True
                    game = load_json_game(json_path)
                    if has_all_valid_tokenized_views(tokenized_year_dir, vocab, game, log_id):
                        continue
                    tokenize_game_views_to_files(vocab, game, tokenized_year_dir, log_id)
                    year_updated = True
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(log_id)
                    print(exc)

            if year_updated or not hf_dataset_year_dir.exists():
                save_year_hf_dataset(
                    year=year,
                    tokenized_dir=tokenized_year_dir,
                    json_dir=json_year_dir,
                    output_dir=hf_dataset_year_dir,
                )


if __name__ == "__main__":
    main()
