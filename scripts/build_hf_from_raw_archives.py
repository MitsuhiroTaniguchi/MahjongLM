#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, Features, Sequence, Value, disable_progress_bar

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tenhou_tokenizer import Vocabulary, save_hf_tokenizer_assets, tokenize_game_views

DEFAULT_RAW_DIR = ROOT.parent / "raw_data"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "huggingface_datasets"
DEFAULT_CONVERT = ROOT / "scripts" / "paifu_scraping" / "convert.pl"

_WORKER_CONVERT: str | None = None
_WORKER_TOKEN_TO_ID: Dict[str, int] | None = None
_WORKER_PERL: str | None = None
SHUFFLE_SEED_RE = re.compile(r'<SHUFFLE\s+[^>]*seed="mt19937ar-sha512-n288-base64,([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-year Hugging Face datasets directly from yearly raw ZIP archives."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing YYYY.zip archives (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination root for per-year datasets (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache root for Dataset.from_generator (default: <output-dir>/_cache)",
    )
    parser.add_argument(
        "--convert-path",
        type=Path,
        default=DEFAULT_CONVERT,
        help=f"Path to convert.pl (default: {DEFAULT_CONVERT})",
    )
    parser.add_argument(
        "--perl-path",
        type=Path,
        default=None,
        help="Explicit path to perl executable. Auto-detected by default.",
    )
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        default=[],
        help="Process only this year. Repeatable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker process count for convert+tokenize (default: 4).",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=64,
        help="Max in-flight worker tasks while streaming archives (default: 64).",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep intermediate Arrow cache directories after completion.",
    )
    return parser.parse_args()


def resolve_perl_executable(explicit_path: Path | None = None) -> str:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    env_perl = os.environ.get("PERL")
    if env_perl:
        candidates.append(Path(env_perl))

    which_perl = shutil.which("perl")
    if which_perl:
        candidates.append(Path(which_perl))

    candidates.extend(
        [
            Path(r"C:\Program Files\Git\usr\bin\perl.exe"),
            Path(r"C:\Strawberry\perl\bin\perl.exe"),
            Path(r"C:\msys64\usr\bin\perl.exe"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return str(candidate)

    searched = "\n".join(str(path) for path in seen)
    raise FileNotFoundError(f"perl executable not found. Searched:\n{searched}")


def infer_seat_count(game: dict) -> int:
    title = game.get("title")
    if isinstance(title, str):
        if "三" in title:
            return 3
        if "四" in title:
            return 4
    for key in ("defen", "rank", "player", "point"):
        value = game.get(key)
        if isinstance(value, list) and len(value) in {3, 4}:
            return len(value)
    log = game.get("log")
    if isinstance(log, list):
        for round_data in log:
            if not isinstance(round_data, list):
                continue
            for event in round_data:
                if not isinstance(event, dict):
                    continue
                for payload_key in ("qipai", "hule", "pingju"):
                    payload = event.get(payload_key)
                    if not isinstance(payload, dict):
                        continue
                    for list_key in ("shoupai", "defen", "fenpei"):
                        values = payload.get(list_key)
                        if isinstance(values, list) and len(values) in {3, 4}:
                            return len(values)
    return 4


def _init_worker(convert_path: str, token_to_id: Dict[str, int], perl_path: str) -> None:
    global _WORKER_CONVERT, _WORKER_TOKEN_TO_ID, _WORKER_PERL
    _WORKER_CONVERT = convert_path
    _WORKER_TOKEN_TO_ID = token_to_id
    _WORKER_PERL = perl_path


def _encode(tokens: List[str]) -> List[int]:
    assert _WORKER_TOKEN_TO_ID is not None
    try:
        return [_WORKER_TOKEN_TO_ID[token] for token in tokens]
    except KeyError as exc:
        raise KeyError(f"token not in vocab: {exc}") from exc


def _extract_shuffle_seed(raw_text: str) -> str | None:
    match = SHUFFLE_SEED_RE.search(raw_text)
    return match.group(1) if match else None


def _process_one(year: int, log_id: str, raw_text: str) -> Tuple[List[dict], str | None]:
    assert _WORKER_CONVERT is not None
    assert _WORKER_PERL is not None
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    try:
        proc = subprocess.run(
            [_WORKER_PERL, "-T", _WORKER_CONVERT],
            input=raw_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
            env=env,
        )
        game = json.loads(proc.stdout)
        shuffle_seed = _extract_shuffle_seed(raw_text)
        if shuffle_seed is not None:
            game["_shuffle_seed"] = shuffle_seed
        views = tokenize_game_views(game)
        seat_count = infer_seat_count(game)
        rows: List[dict] = []
        for view in views:
            rows.append(
                {
                    "game_id": log_id,
                    "group_id": log_id,
                    "year": year,
                    "seat_count": seat_count,
                    "view_type": view.view_type,
                    "viewer_seat": -1 if view.viewer_seat is None else view.viewer_seat,
                    "length": len(view.tokens),
                    "input_ids": _encode(view.tokens),
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _iter_raw_logs(zip_path: Path) -> Iterable[Tuple[str, str]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        payloads = [name for name in zf.namelist() if not name.endswith("/")]
        if len(payloads) != 1:
            raise RuntimeError(f"{zip_path.name}: expected one payload, got {len(payloads)}")
        with zf.open(payloads[0], "r") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    member_name = Path(member.name).name
                    if not member_name.endswith(".txt"):
                        continue
                    if member_name.startswith("._"):
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    yield member_name[:-4], src.read().decode("utf-8", errors="replace")


def iter_year_rows(
    *,
    year: int,
    zip_path: str,
    workers: int,
    max_inflight: int,
    convert_path: str,
    perl_path: str,
    token_to_id: Dict[str, int],
) -> Iterable[dict]:
    started_at = time.time()
    pending: Dict[concurrent.futures.Future, int] = {}
    submitted = 0
    emitted = 0
    skipped = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(convert_path, token_to_id, perl_path),
    ) as executor:
        for log_id, raw_text in _iter_raw_logs(Path(zip_path)):
            future = executor.submit(_process_one, year, log_id, raw_text)
            pending[future] = 1
            submitted += 1

            if len(pending) < max_inflight:
                continue

            done, _ = concurrent.futures.wait(
                pending.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for completed in done:
                pending.pop(completed)
                rows, err = completed.result()
                if err is not None:
                    skipped += 1
                else:
                    for row in rows:
                        emitted += 1
                        yield row
            if submitted % 2000 == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                print(
                    f"[{year}] submitted={submitted} emitted_rows={emitted} "
                    f"skipped_games={skipped} rate={submitted/elapsed:.2f} games/s",
                    flush=True,
                )

        while pending:
            done, _ = concurrent.futures.wait(
                pending.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for completed in done:
                pending.pop(completed)
                rows, err = completed.result()
                if err is not None:
                    skipped += 1
                else:
                    for row in rows:
                        emitted += 1
                        yield row

    elapsed = time.time() - started_at
    print(
        f"[{year}] complete submitted_games={submitted} emitted_rows={emitted} "
        f"skipped_games={skipped} elapsed={elapsed:.1f}s",
        flush=True,
    )


def build_year_dataset(
    *,
    year: int,
    zip_path: Path,
    output_dir: Path,
    cache_root: Path,
    workers: int,
    max_inflight: int,
    convert_path: Path,
    perl_path: str,
    token_to_id: Dict[str, int],
    token_dtype: str,
    keep_cache: bool,
) -> None:
    year_out_dir = output_dir / str(year)
    year_cache_dir = cache_root / str(year)

    if year_out_dir.exists():
        shutil.rmtree(year_out_dir)
    if year_cache_dir.exists():
        shutil.rmtree(year_cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    features = Features(
        {
            "game_id": Value("string"),
            "group_id": Value("string"),
            "year": Value("int32"),
            "seat_count": Value("int8"),
            "view_type": Value("string"),
            "viewer_seat": Value("int8"),
            "length": Value("int32"),
            "input_ids": Sequence(Value(token_dtype)),
        }
    )

    print(f"[{year}] building dataset from {zip_path.name}", flush=True)
    try:
        dataset = Dataset.from_generator(
            iter_year_rows,
            gen_kwargs={
                "year": year,
                "zip_path": str(zip_path),
                "workers": workers,
                "max_inflight": max_inflight,
                "convert_path": str(convert_path),
                "perl_path": perl_path,
                "token_to_id": token_to_id,
            },
            features=features,
            cache_dir=str(year_cache_dir),
        )
        dataset.save_to_disk(str(year_out_dir), max_shard_size="512MB")
        print(f"[{year}] saved rows={len(dataset)} -> {year_out_dir}", flush=True)
    finally:
        if not keep_cache:
            shutil.rmtree(year_cache_dir, ignore_errors=True)


def _iter_year_archives(raw_dir: Path, only_years: set[int]) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for path in sorted(raw_dir.glob("[0-9][0-9][0-9][0-9].zip")):
        year = int(path.stem)
        if only_years and year not in only_years:
            continue
        items.append((year, path))
    return items


def main() -> int:
    args = parse_args()
    if not args.convert_path.is_file():
        raise FileNotFoundError(f"convert.pl not found: {args.convert_path}")
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(f"raw-dir not found: {args.raw_dir}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_inflight <= 0:
        raise ValueError("--max-inflight must be positive")

    perl_path = resolve_perl_executable(args.perl_path)

    disable_progress_bar()
    save_hf_tokenizer_assets()
    vocab = Vocabulary.load()
    token_to_id = vocab.token_to_id
    token_dtype = "uint16" if len(vocab.tokens) < 2**16 else "uint32"
    cache_root = args.cache_dir if args.cache_dir is not None else (args.output_dir / "_cache")

    selected_years = set(args.year)
    archives = _iter_year_archives(args.raw_dir, selected_years)
    if not archives:
        raise FileNotFoundError(f"no YYYY.zip archives found under {args.raw_dir}")

    for year, zip_path in archives:
        build_year_dataset(
            year=year,
            zip_path=zip_path,
            output_dir=args.output_dir,
            cache_root=cache_root,
            workers=args.workers,
            max_inflight=args.max_inflight,
            convert_path=args.convert_path,
            perl_path=perl_path,
            token_to_id=token_to_id,
            token_dtype=token_dtype,
            keep_cache=args.keep_cache,
        )

    if not args.keep_cache:
        shutil.rmtree(cache_root, ignore_errors=True)

    print("all requested years complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
