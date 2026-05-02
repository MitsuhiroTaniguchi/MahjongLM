from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRAIN_SCRIPT = str(ROOT / "scripts" / "train_qwen3.py")


def log(message: str, log_file: Path) -> None:
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def get_process_command_lines() -> list[str]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'train_qwen3.py' } | "
        "Select-Object -ExpandProperty CommandLine | ConvertTo-Json"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    text = proc.stdout.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    return []


def extract_arg(tokens: list[str], name: str) -> str | None:
    for index, token in enumerate(tokens[:-1]):
        if token == name:
            return tokens[index + 1]
    return None


def split_logged_command(line: str) -> list[str]:
    if line.startswith("+ "):
        line = line[2:]
    return [token.strip("'\"") for token in line.strip().split() if token.strip()]


def command_output_dir(command_line: str) -> Path | None:
    match = re.search(r"--output-dir\s+('|\")?([^'\"\s]+)", command_line)
    if not match:
        return None
    return Path(match.group(2)).resolve()


def active_output_dirs() -> set[Path]:
    active: set[Path] = set()
    for command_line in get_process_command_lines():
        output_dir = command_output_dir(command_line)
        if output_dir is not None:
            active.add(output_dir)
    return active


def iter_logged_train_commands(run_root: Path) -> dict[Path, list[str]]:
    commands: dict[Path, list[str]] = {}
    for log_path in run_root.glob("*.log"):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.startswith("+ ") or "train_qwen3.py" not in line:
                continue
            tokens = split_logged_command(line)
            output_dir_text = extract_arg(tokens, "--output-dir")
            if not output_dir_text:
                continue
            commands[Path(output_dir_text).resolve()] = tokens
    return commands


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    checkpoints = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    return checkpoints[-1] if checkpoints else None


def is_complete(output_dir: Path) -> bool:
    return (output_dir / "final_model").is_dir()


def build_resume_command(tokens: list[str]) -> list[str]:
    command = list(tokens)
    if "--resume-latest-checkpoint" not in command:
        command.append("--resume-latest-checkpoint")
    return command


def start_resume(output_dir: Path, tokens: list[str], watchdog_log: Path) -> None:
    resume_log = output_dir / f"watchdog_resume_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["WANDB__SERVICE_WAIT"] = "300"
    env["WANDB_MODE"] = "online"
    command = build_resume_command(tokens)
    log(f"restarting {output_dir.name} from {latest_checkpoint(output_dir)}", watchdog_log)
    log("+ " + " ".join(command), watchdog_log)
    with resume_log.open("a", encoding="utf-8") as handle:
        handle.write("+ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Restart crashed MahjongLM training runs from latest checkpoints.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=180)
    parser.add_argument("--stale-seconds", type=int, default=180)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    watchdog_log = run_root / "watchdog.log"
    restarted: dict[Path, float] = {}
    log(f"watchdog started run_root={run_root}", watchdog_log)

    while True:
        try:
            commands = iter_logged_train_commands(run_root)
            active = active_output_dirs()
            now = time.time()
            for output_dir, tokens in sorted(commands.items(), key=lambda item: item[0].name):
                if output_dir in active or is_complete(output_dir):
                    continue
                if (output_dir / "STOP").exists():
                    continue
                checkpoint = latest_checkpoint(output_dir)
                if checkpoint is None:
                    continue
                last_restart = restarted.get(output_dir, 0.0)
                if now - last_restart < args.stale_seconds:
                    continue
                start_resume(output_dir, tokens, watchdog_log)
                restarted[output_dir] = now
        except Exception as exc:  # keep the watchdog alive even if a transient parse fails
            log(f"watchdog error: {exc!r}", watchdog_log)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
