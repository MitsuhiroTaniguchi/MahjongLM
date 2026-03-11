#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

DEFAULT_ZIP_PATH = ROOT / "tests/fixtures/tenhou/data2023_sample_v1.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tokenizer invariants over Tenhou ZIP data.")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from tenhou_tokenizer import TenhouTokenizer, TokenizeError
        from validation_helpers import (
            validate_round_stepwise,
            validate_score_rotation,
            validate_token_stream,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pymahjong":
            print(
                "pymahjong is required. Activate your venv and install it first: "
                "./scripts/setup_pymahjong.sh",
                file=sys.stderr,
            )
            return 2
        raise

    try:
        zf = zipfile.ZipFile(args.zip_path)
    except FileNotFoundError:
        print(f"zip not found: {args.zip_path}", file=sys.stderr)
        return 2
    except (IsADirectoryError, PermissionError):
        print(f"cannot open zip: {args.zip_path}", file=sys.stderr)
        return 2
    except zipfile.BadZipFile:
        print(f"invalid zip file: {args.zip_path}", file=sys.stderr)
        return 2

    checked = 0
    with zf:
        for idx, name in enumerate(zf.namelist()):
            if idx < args.start_index:
                continue
            if args.max_games is not None and checked >= args.max_games:
                break

            try:
                with zf.open(name) as f:
                    game = json.load(f)
                tokens = TenhouTokenizer().tokenize_game(game)
                validate_token_stream(tokens)
                validate_score_rotation(game)
                for round_data in game.get("log", []):
                    validate_round_stepwise(round_data)
            except KeyboardInterrupt:
                raise
            except (json.JSONDecodeError, KeyError, TokenizeError, ValueError, AssertionError) as exc:
                print(f"validation failed: {args.zip_path.name}:{name} ({exc})", file=sys.stderr)
                return 1

            checked += 1
            if args.progress_every > 0 and checked % args.progress_every == 0:
                print(f"validated={checked} last_game={name}", file=sys.stderr)

    print(f"done: validated={checked} zip={args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
