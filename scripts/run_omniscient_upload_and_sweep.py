from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DATASET_ROOT = ROOT / "data" / "huggingface_datasets_omniscient"


def run_logged(command: list[str], *, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with log_file.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("+ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
        handle.write(f"\n[exit_code] {return_code}\n")
        handle.flush()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the omniscient dataset, then train/publish the 10M and 100M models."
    )
    parser.add_argument("--repo-id", default="mitsutani/mahjonglm-dataset")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--large-upload-workers", type=int, default=2)
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)
    if not args.skip_upload:
        run_logged(
            [
                str(PYTHON),
                str(ROOT / "scripts" / "upload_hf_dataset.py"),
                "--repo-id",
                args.repo_id,
                "--source-dir",
                str(DATASET_ROOT),
                "--tokenizer-dir",
                str(ROOT / "tokenizer"),
                "--large-upload-workers",
                str(args.large_upload_workers),
            ],
            log_file=args.log_file,
        )
    run_logged(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "run_omniscient_sweep_pipeline.py"),
            "--models",
            "10m",
            "100m",
            "--publish",
            "--run-root",
            str(args.run_root),
        ],
        log_file=args.log_file,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        raise
