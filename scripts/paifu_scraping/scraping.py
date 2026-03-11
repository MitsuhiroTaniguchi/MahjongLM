from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
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
FETCH_SLEEP_SECONDS = 0.05
PROCESS_WORKERS = 4
MAX_INFLIGHT_PROCESS_TASKS = 32
NON_TTY_PROGRESS_INTERVAL_SECONDS = 5.0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch, convert, tokenize, and build HF datasets from Tenhou archives.")
    parser.add_argument(
        "--year",
        dest="years",
        action="append",
        type=int,
        default=[],
        help="Process only this year. Repeatable.",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Process only years >= this value.",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Process only years <= this value.",
    )
    parser.add_argument(
        "--exclude-year",
        dest="excluded_years",
        action="append",
        type=int,
        default=[],
        help="Skip this year. Repeatable.",
    )
    return parser.parse_args()


def filter_archives(
    archives: list[tuple[int, Path]],
    *,
    years: set[int],
    excluded_years: set[int],
    year_min: int | None,
    year_max: int | None,
) -> list[tuple[int, Path]]:
    selected: list[tuple[int, Path]] = []
    for year, path in archives:
        if year in excluded_years:
            continue
        if years and year not in years:
            continue
        if year_min is not None and year < year_min:
            continue
        if year_max is not None and year > year_max:
            continue
        selected.append((year, path))
    return selected


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
                env={
                    **os.environ,
                    "LC_ALL": "C",
                    "LANG": "C",
                },
            )
        temp_path.replace(json_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def has_downloaded_raw(raw_path: Path) -> bool:
    return raw_path.is_file() and raw_path.stat().st_size > 0


def not_found_marker_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(raw_path.suffix + ".404")


def has_not_found_marker(raw_path: Path) -> bool:
    marker_path = not_found_marker_path(raw_path)
    return marker_path.is_file() and marker_path.stat().st_size >= 0


def write_not_found_marker(raw_path: Path) -> None:
    atomic_write_text(not_found_marker_path(raw_path), "404 Not Found\n")


def clear_not_found_marker(raw_path: Path) -> None:
    marker_path = not_found_marker_path(raw_path)
    if marker_path.exists():
        marker_path.unlink()


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


def has_any_tokenized_views(tokenized_year_dir: Path) -> bool:
    return any(tokenized_year_dir.glob("*.ids.bin"))


def _process_log_id(
    *,
    log_id: str,
    raw_path: str,
    json_path: str,
    tokenized_year_dir: str,
) -> tuple[str, bool]:
    raw_path_obj = Path(raw_path)
    json_path_obj = Path(json_path)
    tokenized_year_dir_obj = Path(tokenized_year_dir)
    vocab = Vocabulary.load()

    updated = False
    if not has_valid_json(json_path_obj):
        convert_raw_to_json(raw_path_obj, json_path_obj)
        updated = True

    game = load_json_game(json_path_obj)
    if has_all_valid_tokenized_views(tokenized_year_dir_obj, vocab, game, log_id):
        return log_id, updated

    tokenize_game_views_to_files(vocab, game, tokenized_year_dir_obj, log_id)
    return log_id, True


def _drain_completed(
    pending: dict[concurrent.futures.Future[tuple[str, bool]], str],
    *,
    block_until_one: bool,
) -> tuple[list[tuple[str, bool]], list[tuple[str, BaseException]]]:
    if not pending:
        return [], []
    done, _ = concurrent.futures.wait(
        pending.keys(),
        return_when=(
            concurrent.futures.FIRST_COMPLETED
            if block_until_one
            else concurrent.futures.ALL_COMPLETED
        ),
    )
    completed: list[tuple[str, bool]] = []
    failures: list[tuple[str, BaseException]] = []
    for future in done:
        log_id = pending.pop(future)
        try:
            completed.append(future.result())
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            failures.append((log_id, exc))
    return completed, failures


def iter_with_progress(lines: list[str], *, year: int):
    if sys.stderr.isatty():
        yield from tqdm(lines, desc=str(year), dynamic_ncols=True)
        return

    total = len(lines)
    print(f"{year}: starting {total} archive lines")
    started_at = time.monotonic()
    last_reported_at = started_at
    for index, line in enumerate(lines, start=1):
        now = time.monotonic()
        should_report = (
            index == 1
            or index == total
            or (now - last_reported_at) >= NON_TTY_PROGRESS_INTERVAL_SECONDS
        )
        if should_report:
            elapsed = max(now - started_at, 1e-9)
            rate = index / elapsed
            remaining = max(total - index, 0)
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_minutes, eta_remainder = divmod(int(eta_seconds), 60)
            percent = (index / total) * 100 if total else 100.0
            print(
                f"{year}: {index}/{total} ({percent:5.2f}%) "
                f"at {rate:,.1f} lines/s ETA {eta_minutes:02d}:{eta_remainder:02d}"
            )
            last_reported_at = now
        yield line


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    archives = filter_archives(
        iter_archive_zips(),
        years=set(args.years),
        excluded_years=set(args.excluded_years),
        year_min=args.year_min,
        year_max=args.year_max,
    )
    if not archives:
        raise FileNotFoundError(f"no scraw*.zip found under {LOG_DIR}")
    if not CONVERT.is_file():
        raise FileNotFoundError(f"convert.pl not found: {CONVERT}")

    save_hf_tokenizer_assets()
    with requests.Session() as session, concurrent.futures.ProcessPoolExecutor(max_workers=PROCESS_WORKERS) as executor:
        for year, zip_path in archives:
            raw_year_dir = RAW_DIR / str(year)
            json_year_dir = JSON_DIR / str(year)
            tokenized_year_dir = TOKENIZED_DIR / str(year)
            hf_dataset_year_dir = HF_DATASETS_DIR / str(year)
            raw_year_dir.mkdir(parents=True, exist_ok=True)
            json_year_dir.mkdir(parents=True, exist_ok=True)
            tokenized_year_dir.mkdir(parents=True, exist_ok=True)
            year_updated = False
            pending_tasks: dict[concurrent.futures.Future[tuple[str, bool]], str] = {}

            archive_lines = iter_archive_lines(zip_path)
            for line in iter_with_progress(archive_lines, year=year):
                log_id = parse_log_id(line)
                if not log_id:
                    continue

                raw_path = raw_year_dir / f"{log_id}.txt"
                json_path = json_year_dir / f"{log_id}.json"

                try:
                    if not has_valid_json(json_path):
                        if has_not_found_marker(raw_path) and not has_downloaded_raw(raw_path):
                            continue
                        if not has_downloaded_raw(raw_path):
                            try:
                                atomic_write_text(raw_path, fetch_log_text(session, log_id))
                            except requests.HTTPError as exc:
                                response = exc.response
                                if response is not None and response.status_code == 404:
                                    write_not_found_marker(raw_path)
                                    print(f"{log_id} 404 not found; marked to skip future fetches")
                                    continue
                                raise
                            clear_not_found_marker(raw_path)
                            time.sleep(FETCH_SLEEP_SECONDS)
                    future = executor.submit(
                        _process_log_id,
                        log_id=log_id,
                        raw_path=str(raw_path),
                        json_path=str(json_path),
                        tokenized_year_dir=str(tokenized_year_dir),
                    )
                    pending_tasks[future] = log_id
                    if len(pending_tasks) >= MAX_INFLIGHT_PROCESS_TASKS:
                        completed, failures = _drain_completed(pending_tasks, block_until_one=True)
                        year_updated = year_updated or any(updated for _log_id, updated in completed)
                        for failed_log_id, exc in failures:
                            print(failed_log_id)
                            print(exc)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(log_id)
                    print(exc)

            completed, failures = _drain_completed(pending_tasks, block_until_one=False)
            year_updated = year_updated or any(updated for _log_id, updated in completed)
            for failed_log_id, exc in failures:
                print(failed_log_id)
                print(exc)

            if not has_any_tokenized_views(tokenized_year_dir):
                print(f"{year}: no tokenized views; skipping HF dataset build")
                continue

            if year_updated or not hf_dataset_year_dir.exists():
                save_year_hf_dataset(
                    year=year,
                    tokenized_dir=tokenized_year_dir,
                    json_dir=json_year_dir,
                    output_dir=hf_dataset_year_dir,
                )


if __name__ == "__main__":
    main()
