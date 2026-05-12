from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN = ROOT / "scripts" / "train_qwen3.py"
DATASET_ROOT = ROOT / "data" / "huggingface_datasets_omniscient"
WANDB_PROJECT = "mahjongLM_qwen3_plain_omniscient_sweep"
PUBLISH_ROOT = ROOT / "outputs" / "omniscient_publish"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_args: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec(
        key="1m",
        model_args=(
            "--qwen-arch",
            "custom",
            "--qwen-hidden-size",
            "128",
            "--qwen-intermediate-size",
            "384",
            "--qwen-num-hidden-layers",
            "5",
            "--qwen-num-attention-heads",
            "4",
            "--qwen-num-key-value-heads",
            "1",
            "--qwen-head-dim",
            "32",
            "--qwen-max-position-embeddings",
            "16384",
        ),
    ),
    ModelSpec(
        key="10m",
        model_args=(
            "--qwen-arch",
            "custom",
            "--qwen-hidden-size",
            "256",
            "--qwen-intermediate-size",
            "768",
            "--qwen-num-hidden-layers",
            "13",
            "--qwen-num-attention-heads",
            "4",
            "--qwen-num-key-value-heads",
            "1",
            "--qwen-head-dim",
            "64",
            "--qwen-max-position-embeddings",
            "16384",
        ),
    ),
    ModelSpec(
        key="100m",
        model_args=(
            "--qwen-arch",
            "Q100",
            "--qwen-max-position-embeddings",
            "16384",
        ),
    ),
)


def now_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def dataset_args() -> list[str]:
    args: list[str] = []
    for year in range(2011, 2025):
        args.extend(["--dataset-dir", str(DATASET_ROOT / str(year))])
    return args


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_train_command(spec: ModelSpec, output_dir: Path, run_name: str, stop_file: Path) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN),
        *dataset_args(),
        "--output-dir",
        str(output_dir),
        "--tokenizer-dir",
        str(ROOT / "tokenizer"),
        "--model-family",
        "qwen3",
        *spec.model_args,
        "--max-seq-length",
        "16384",
        "--packing-mode",
        "unpadded",
        "--attn-implementation",
        "flash_attention_2",
        "--pad-to-multiple-of",
        "16",
        "--max-tokens-per-batch",
        "65536",
        "--eval-max-tokens-per-batch",
        "65536",
        "--per-device-train-batch-size",
        "8",
        "--per-device-eval-batch-size",
        "8",
        "--gradient-accumulation-steps",
        "64",
        "--train-steps",
        "0",
        "--train-epochs",
        "0.2",
        "--eval-interval",
        "100",
        "--save-interval",
        "200",
        "--log-interval",
        "10",
        "--optimizer-name",
        "muon",
        "--use-muon-plus",
        "--lr-scheduler-type",
        "linear",
        "--learning-rate",
        "0.03",
        "--muon-aux-learning-rate",
        "0.03",
        "--warmup-steps",
        "200",
        "--label-smoothing",
        "0",
        "--train-split-eval-ratio",
        "0.001",
        "--split-seed",
        "1337",
        "--data-seed",
        "1337",
        "--seed",
        "1337",
        "--gradient-checkpointing",
        "--eval-device",
        "cuda",
        "--require-wandb",
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-run-name",
        run_name,
        "--wandb-tags",
        "plain-qwen3",
        "--wandb-tags",
        "bos-eos",
        "--wandb-tags",
        "hule-result",
        "--wandb-tags",
        "omniscient",
        "--wandb-tags",
        "ctx-16384",
        "--wandb-tags",
        spec.key,
        "--stop-file",
        str(stop_file),
    ]


def run_training(cmd: list[str], *, run_root: Path, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ps_path = run_root / "launch_train.ps1"
    exe = cmd[0]
    args = cmd[1:]
    args_block = ", ".join(ps_quote(arg) for arg in args)
    ps_path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"Set-Location -LiteralPath {ps_quote(str(ROOT))}",
                "$env:WANDB__SERVICE_WAIT = '300'",
                "$env:WANDB_MODE = 'online'",
                f"$exe = {ps_quote(exe)}",
                f"$arguments = @({args_block})",
                "& $exe @arguments",
                "exit $LASTEXITCODE",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with log_file.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps_path),
        ],
        cwd=ROOT,
        check=True,
    )


def run_model(spec: ModelSpec, run_root: Path) -> Path:
    run_name = f"q{spec.key}-omniscient-allyears-0p2ep-{now_slug()}"
    output_dir = ROOT / "outputs" / run_name
    stop_file = output_dir / "STOP"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== training {spec.key}: {run_name} ===", flush=True)
    run_training(build_train_command(spec, output_dir, run_name, stop_file), run_root=run_root, log_file=run_root / f"{spec.key}.log")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete/imperfect/omniscient 1M/10M/100M training sweep.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.key for spec in MODEL_SPECS],
        default=[spec.key for spec in MODEL_SPECS],
    )
    parser.add_argument("--run-root", type=Path, default=ROOT / "outputs" / f"omniscient_sweep_{now_slug()}")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish each model raw checkpoint and Q4_K_M GGUF immediately after it finishes.",
    )
    parser.add_argument("--publish-root", type=Path, default=PUBLISH_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    missing = [str(DATASET_ROOT / str(year)) for year in range(2011, 2025) if not (DATASET_ROOT / str(year)).exists()]
    if missing:
        raise FileNotFoundError("omniscient dataset directories are missing:\n" + "\n".join(missing))
    args.run_root.mkdir(parents=True, exist_ok=True)
    publish_model: Callable[[str, Path], None] | None = None
    if args.publish:
        from publish_omniscient_models import MODEL_SPECS as PUBLISH_SPECS
        from publish_omniscient_models import publish_existing

        args.publish_root.mkdir(parents=True, exist_ok=True)

        def _publish_model(key: str, output_dir: Path) -> None:
            publish_existing(PUBLISH_SPECS[key], output_dir, args.publish_root)

        publish_model = _publish_model
    selected = set(args.models)
    for spec in MODEL_SPECS:
        if spec.key in selected:
            output_dir = run_model(spec, args.run_root)
            if publish_model is not None:
                print(f"=== publishing {spec.key}: {output_dir} ===", flush=True)
                publish_model(spec.key, output_dir)
    print("=== complete ===", flush=True)


if __name__ == "__main__":
    main()
