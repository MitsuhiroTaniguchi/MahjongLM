from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

WANDB_PROJECT = "mahjongLM_view_imperfect_dpo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end launcher for view_imperfect DPO fine-tuning. "
            "This is explicit/foreground only and does not touch existing pretraining runs."
        )
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", action="append", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "view_imperfect_dpo")
    parser.add_argument("--hf-dataset-repo", type=str, default="")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--per-rule-prompts", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--generations-jsonl", type=Path, default=None)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.work_dir / f"dpo-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts_jsonl = run_dir / "omniscient_prompts.jsonl"
    generations_jsonl = args.generations_jsonl or (run_dir / "rollouts.jsonl")
    pref_dataset_dir = run_dir / "preference_dataset"
    dpo_output_dir = run_dir / "dpo_model"

    dataset_args: list[str] = []
    for dataset_dir in args.dataset_dir:
        dataset_args.extend(["--dataset-dir", str(dataset_dir)])
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "sample_dpo_omniscient_prompts.py"),
            *dataset_args,
            "--output-jsonl",
            str(prompts_jsonl),
            "--tokenizer-dir",
            str(args.tokenizer_dir),
            "--per-rule",
            str(args.per_rule_prompts),
        ]
    )

    if not args.skip_rollouts and args.generations_jsonl is None:
        run(
            [
                str(PYTHON),
                str(ROOT / "scripts" / "generate_dpo_rollouts.py"),
                "--model-dir",
                args.model_dir,
                "--prompts-jsonl",
                str(prompts_jsonl),
                "--output-jsonl",
                str(generations_jsonl),
                "--tokenizer-dir",
                str(args.tokenizer_dir),
                "--batch-size",
                str(args.rollout_batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--trust-remote-code",
            ]
        )

    build_dataset_cmd = [
        str(PYTHON),
        str(ROOT / "scripts" / "build_dpo_preference_dataset.py"),
        "--generations-jsonl",
        str(generations_jsonl),
        "--output-dir",
        str(pref_dataset_dir),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
    ]
    if args.hf_dataset_repo:
        build_dataset_cmd.extend(["--repo-id", args.hf_dataset_repo])
    run(build_dataset_cmd)

    if not args.skip_training:
        run(
            [
                str(PYTHON),
                str(ROOT / "scripts" / "train_dpo_unsloth.py"),
                "--model-dir",
                args.model_dir,
                "--dataset-dir",
                str(pref_dataset_dir),
                "--output-dir",
                str(dpo_output_dir),
                "--tokenizer-dir",
                str(args.tokenizer_dir),
                "--load-in-4bit",
                "--require-wandb",
                "--wandb-project",
                WANDB_PROJECT,
                "--wandb-run-name",
                f"view-imperfect-dpo-{stamp}",
            ]
        )


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
