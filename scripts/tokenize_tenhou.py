#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import sys
from glob import glob
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_ZIP_PATH = ROOT / "tests/fixtures/tenhou/data2023_sample_v1.zip"
DEFAULT_ZIP_GLOB = str(ROOT / "tests/fixtures/tenhou/data2023*.zip")
DEFAULT_OUTPUT_PATH = ROOT / "data/processed/tenhou/tokens_2023.jsonl.gz"
AUTO_PARALLEL_MIN_GAMES = 512

_WORKER_ZIP_PATH: Optional[str] = None
_WORKER_ZF = None
_WORKER_TOKENIZER = None
_WORKER_TOKENIZE_ERROR = None


def open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _iter_chunks(items: Sequence[str], chunk_size: int) -> Iterator[List[str]]:
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _init_tokenize_worker(zip_path: str) -> None:
    import zipfile

    global _WORKER_ZIP_PATH, _WORKER_ZF, _WORKER_TOKENIZER, _WORKER_TOKENIZE_ERROR
    from tenhou_tokenizer import TenhouTokenizer, TokenizeError

    if _WORKER_ZIP_PATH == zip_path and _WORKER_ZF is not None and _WORKER_TOKENIZER is not None:
        return
    if _WORKER_ZF is not None:
        _WORKER_ZF.close()
    _WORKER_ZIP_PATH = zip_path
    _WORKER_ZF = zipfile.ZipFile(zip_path)
    _WORKER_TOKENIZER = TenhouTokenizer()
    _WORKER_TOKENIZE_ERROR = TokenizeError


def _tokenize_chunk_worker(names: Sequence[str]) -> List[Tuple[str, Optional[List[str]], Optional[str]]]:
    global _WORKER_ZF, _WORKER_TOKENIZER, _WORKER_TOKENIZE_ERROR
    assert _WORKER_ZF is not None
    assert _WORKER_TOKENIZER is not None
    assert _WORKER_TOKENIZE_ERROR is not None

    rows: List[Tuple[str, Optional[List[str]], Optional[str]]] = []
    for name in names:
        try:
            with _WORKER_ZF.open(name) as f:
                game = json.load(f)
            tokens = _WORKER_TOKENIZER.tokenize_game(game)
            rows.append((name, tokens, None))
        except KeyboardInterrupt:
            raise
        except (json.JSONDecodeError, KeyError, _WORKER_TOKENIZE_ERROR, ValueError) as exc:
            rows.append((name, None, str(exc)))
    return rows


def _tokenize_zip_serial(
    names: Sequence[str],
    tokenizer,
    zf,
) -> Iterator[Tuple[str, Optional[List[str]], Optional[str]]]:
    from tenhou_tokenizer import TokenizeError

    for name in names:
        try:
            with zf.open(name) as f:
                game = json.load(f)
            yield name, tokenizer.tokenize_game(game), None
        except KeyboardInterrupt:
            raise
        except (json.JSONDecodeError, KeyError, TokenizeError, ValueError) as exc:
            yield name, None, str(exc)


def _tokenize_zip_parallel(
    zip_path: Path,
    names: Sequence[str],
    workers: int,
    chunk_size: int,
) -> Iterator[Tuple[str, Optional[List[str]], Optional[str]]]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_tokenize_worker,
        initargs=(str(zip_path),),
    ) as executor:
        for chunk_rows in executor.map(_tokenize_chunk_worker, _iter_chunks(names, chunk_size)):
            for row in chunk_rows:
                yield row


def _tokenize_zip_parallel_with_fallback(
    zip_path: Path,
    names: Sequence[str],
    workers: int,
    chunk_size: int,
    tokenizer,
    zf,
) -> Iterator[Tuple[str, Optional[List[str]], Optional[str]]]:
    try:
        yield from _tokenize_zip_parallel(
            zip_path=zip_path,
            names=names,
            workers=workers,
            chunk_size=chunk_size,
        )
    except (OSError, PermissionError) as exc:
        print(
            f"parallel tokenization unavailable for {zip_path.name}; "
            f"falling back to serial ({exc})",
            file=sys.stderr,
        )
        yield from _tokenize_zip_serial(names, tokenizer=tokenizer, zf=zf)


def _resolve_workers(requested_workers: int, names_count: int) -> int:
    if requested_workers == 1 or names_count <= 1:
        return 1
    if requested_workers <= 0:
        if names_count < AUTO_PARALLEL_MIN_GAMES:
            return 1
        requested_workers = os.process_cpu_count() or 1
    return max(1, min(requested_workers, names_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-tokenize Tenhou JSON zip data.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help=f"Path to one Tenhou ZIP archive. Default: {DEFAULT_ZIP_PATH}",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Process all ZIPs matching --zip-glob.",
    )
    parser.add_argument(
        "--zip-glob",
        type=str,
        default=None,
        help=f"Glob pattern used when --all-years is set. Default: {DEFAULT_ZIP_GLOB}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSONL(.gz) path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in zip namelist.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N tokenized games.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any game is skipped.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes. Use 0 to auto-select; small inputs stay serial. Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Games per worker task when --workers != 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = args.zip_path if args.zip_path is not None else DEFAULT_ZIP_PATH
    zip_glob = args.zip_glob if args.zip_glob is not None else DEFAULT_ZIP_GLOB
    output_path = args.output if args.output is not None else DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import zipfile
    try:
        from tenhou_tokenizer import TenhouTokenizer, TokenizeError
    except ModuleNotFoundError as exc:
        if exc.name == "pymahjong":
            print(
                "pymahjong is required. Activate your venv and install it first: "
                "./scripts/setup_pymahjong.sh",
                file=sys.stderr,
            )
            return 2
        raise

    written = 0
    skipped = 0

    if args.all_years:
        zip_paths = sorted(Path(p) for p in glob(zip_glob))
        if not zip_paths:
            print(f"no zip matched --zip-glob: {zip_glob}", file=sys.stderr)
            return 2
    else:
        zip_paths = [zip_path]

    with open_output(output_path) as out:
        for zip_path in zip_paths:
            try:
                zf = zipfile.ZipFile(zip_path)
            except FileNotFoundError:
                print(f"zip not found: {zip_path}", file=sys.stderr)
                return 2
            except (IsADirectoryError, PermissionError):
                print(f"cannot open zip: {zip_path}", file=sys.stderr)
                return 2
            except zipfile.BadZipFile:
                print(f"invalid zip file: {zip_path}", file=sys.stderr)
                return 2

            with zf:
                names = zf.namelist()
                selected_names = names[args.start_index :]
                if args.max_games is not None and written >= args.max_games:
                    break
                workers = _resolve_workers(args.workers, len(selected_names))
                chunk_size = max(1, args.chunk_size)
                tokenizer = None
                if workers == 1:
                    tokenizer = TenhouTokenizer()
                    row_iter = _tokenize_zip_serial(selected_names, tokenizer=tokenizer, zf=zf)
                else:
                    tokenizer = TenhouTokenizer()
                    row_iter = _tokenize_zip_parallel_with_fallback(
                        zip_path=zip_path,
                        names=selected_names,
                        workers=workers,
                        chunk_size=chunk_size,
                        tokenizer=tokenizer,
                        zf=zf,
                    )

                for local_idx, (name, tokens, error_text) in enumerate(row_iter, start=args.start_index):
                    if args.max_games is not None and written >= args.max_games:
                        break
                    if error_text is not None:
                        skipped += 1
                        if skipped <= 10:
                            print(f"skip: {zip_path.name}:{name} ({error_text})", file=sys.stderr)
                        continue

                    out.write(
                        json.dumps(
                            {
                                "source_zip": str(zip_path),
                                "game_id": name,
                                "tokens": tokens,
                            },
                            ensure_ascii=False,
                        )
                    )
                    out.write("\n")
                    written += 1

                    if args.progress_every > 0 and written % args.progress_every == 0:
                        print(
                            f"tokenized={written} skipped={skipped} zip={zip_path.name} last_index={local_idx}",
                            file=sys.stderr,
                        )

            if args.max_games is not None and written >= args.max_games:
                break

    print(f"done: tokenized={written} skipped={skipped} output={output_path}")
    if args.strict and skipped > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
