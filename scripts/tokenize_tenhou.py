#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_ZIP_PATH = ROOT / "data/raw/tenhou/data2023.zip"
DEFAULT_ZIP_GLOB = str(ROOT / "data/raw/tenhou/data*.zip")
DEFAULT_OUTPUT_PATH = ROOT / "data/processed/tenhou/tokens_2023.jsonl.gz"


def open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


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

    tokenizer = TenhouTokenizer()
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
                for idx, name in enumerate(names):
                    if idx < args.start_index:
                        continue
                    if args.max_games is not None and written >= args.max_games:
                        break

                    try:
                        with zf.open(name) as f:
                            game = json.load(f)
                        tokens = tokenizer.tokenize_game(game)
                    except KeyboardInterrupt:
                        raise
                    except (json.JSONDecodeError, KeyError, TokenizeError, ValueError) as exc:
                        skipped += 1
                        if skipped <= 10:
                            print(f"skip: {zip_path.name}:{name} ({exc})", file=sys.stderr)
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
                            f"tokenized={written} skipped={skipped} zip={zip_path.name} last_index={idx}",
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
