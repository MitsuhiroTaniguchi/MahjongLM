from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PUBLISH = ROOT / "scripts" / "publish_omniscient_models.py"


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a model once final_model appears.")
    parser.add_argument("--key", required=True, choices=["1m", "10m", "100m"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    run_root = args.run_root.resolve()
    marker = run_root / f"publish_{args.key}.done"
    log_file = run_root / f"publish_{args.key}_on_complete.log"
    append_log(log_file, f"waiting for {output_dir / 'final_model'}")

    while True:
        if marker.exists():
            append_log(log_file, f"marker exists, exiting: {marker}")
            return
        if (output_dir / "final_model").is_dir():
            command = [
                str(PYTHON),
                str(PUBLISH),
                "--run-root",
                str(run_root),
                "--model",
                f"{args.key}={output_dir}",
            ]
            append_log(log_file, "+ " + " ".join(command))
            with (run_root / f"publish_{args.key}_subprocess.log").open("a", encoding="utf-8") as handle:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            append_log(log_file, f"publish exited code={process.returncode}")
            if process.returncode == 0:
                marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
                return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
