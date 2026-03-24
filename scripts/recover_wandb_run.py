from __future__ import annotations

import argparse
import json
import random
import shutil
import string
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import wandb


def _generate_run_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _find_local_run_dir(wandb_dir: Path, run_id: str) -> Path:
    matches = sorted(wandb_dir.glob(f"run-*-{run_id}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find a local wandb run directory for run id {run_id!r} under {wandb_dir}")
    return matches[0]


def _build_wandb_sync_command(
    *,
    python_executable: Path,
    entity: str,
    project: str,
    new_run_id: str,
    backup_dir: Path,
) -> list[str]:
    wandb_exe = python_executable.with_name("wandb.exe")
    if wandb_exe.exists():
        return [
            str(wandb_exe),
            "sync",
            "--include-online",
            "--mark-synced",
            "--id",
            new_run_id,
            "--project",
            project,
            "--entity",
            entity,
            str(backup_dir),
        ]

    return [
        str(python_executable),
        "-m",
        "wandb",
        "sync",
        "--include-online",
        "--mark-synced",
        "--id",
        new_run_id,
        "--project",
        project,
        "--entity",
        entity,
        str(backup_dir),
    ]


@dataclass
class RecoveryResult:
    old_run_id: str
    old_run_name: str
    old_run_url: str
    local_run_dir: str
    backup_dir: str
    new_run_id: str
    new_run_url: str
    new_run_name: str
    new_run_state: str


def recover_run(
    *,
    entity: str,
    project: str,
    run_id: str,
    recovery_root: Path,
    wandb_dir: Path,
    python_executable: Path,
    delete_artifacts: bool,
    new_run_id: str | None,
) -> RecoveryResult:
    api = wandb.Api()
    old_run = api.run(f"{entity}/{project}/{run_id}")
    local_run_dir = _find_local_run_dir(wandb_dir, run_id)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = recovery_root / f"{local_run_dir.name}-backup-{timestamp}"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    shutil.copytree(local_run_dir, backup_dir)

    old_run_name = old_run.name
    old_run_url = old_run.url
    old_run.delete(delete_artifacts=delete_artifacts)

    replacement_run_id = new_run_id or _generate_run_id()
    sync_cmd = _build_wandb_sync_command(
        python_executable=python_executable,
        entity=entity,
        project=project,
        new_run_id=replacement_run_id,
        backup_dir=backup_dir,
    )
    subprocess.run(sync_cmd, check=True)

    new_run = api.run(f"{entity}/{project}/{replacement_run_id}")
    return RecoveryResult(
        old_run_id=run_id,
        old_run_name=old_run_name,
        old_run_url=old_run_url,
        local_run_dir=str(local_run_dir),
        backup_dir=str(backup_dir),
        new_run_id=replacement_run_id,
        new_run_url=new_run.url,
        new_run_name=new_run.name,
        new_run_state=new_run.state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover stale online W&B runs by backing up the local run directory, deleting the stale server run, and re-syncing it as a new run id."
    )
    parser.add_argument("--entity", required=True, help="W&B entity, for example a21-3jck-.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        dest="run_ids",
        help="Stale W&B run id to recover. Pass multiple times to recover multiple runs.",
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        help="Local wandb directory that contains run-* folders.",
    )
    parser.add_argument(
        "--recovery-root",
        default="wandb_recovery",
        help="Directory where local backup copies and the recovery manifest will be written.",
    )
    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="Also delete server-side artifacts associated with the stale run before recovery.",
    )
    parser.add_argument(
        "--new-run-id",
        action="append",
        dest="new_run_ids",
        default=None,
        help="Optional replacement run id. If set, pass one per --run-id in the same order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.new_run_ids and len(args.new_run_ids) != len(args.run_ids):
        raise SystemExit("--new-run-id must be passed exactly once per --run-id when provided.")

    recovery_root = Path(args.recovery_root).resolve()
    recovery_root.mkdir(parents=True, exist_ok=True)
    wandb_dir = Path(args.wandb_dir).resolve()
    python_executable = Path(sys.executable).resolve()

    results: list[RecoveryResult] = []
    for index, run_id in enumerate(args.run_ids):
        replacement_id = args.new_run_ids[index] if args.new_run_ids else None
        result = recover_run(
            entity=args.entity,
            project=args.project,
            run_id=run_id,
            recovery_root=recovery_root,
            wandb_dir=wandb_dir,
            python_executable=python_executable,
            delete_artifacts=args.delete_artifacts,
            new_run_id=replacement_id,
        )
        results.append(result)

    manifest = {
        "entity": args.entity,
        "project": args.project,
        "results": [asdict(result) for result in results],
    }
    manifest_path = recovery_root / f"recovery-manifest-{time.strftime('%Y%m%d-%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "results": manifest["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
