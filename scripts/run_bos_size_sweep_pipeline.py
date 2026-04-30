from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN = ROOT / "scripts" / "train_qwen3.py"
LLAMA_CPP = ROOT / "external" / "llama.cpp"
CONVERT_GGUF = LLAMA_CPP / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE_WSL = "/mnt/c/Users/taniguchi/MahjongLM/external/llama.cpp/build-cpu/bin/llama-quantize"
WORDLEVEL_PATCH_SOURCE = (
    ROOT
    / "outputs"
    / "hf_upload_q100_baseline_step11000_q4km"
    / "patches"
    / "0001-wordlevel-tokenizer-support.patch"
)
WORDLEVEL_PATCH_COMMIT_SOURCE = ROOT / "outputs" / "hf_upload_q100_baseline_step11000_q4km" / "PATCHED_LLAMA_CPP_COMMIT.txt"

WANDB_PROJECT = "mahjongLM_qwen3_plain_bos_sweep"
HF_COLLECTION = "mitsutani/mahjonglm"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    raw_repo: str
    gguf_repo: str
    model_args: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec(
        key="1m",
        label="MahjongLM 1M",
        raw_repo="mitsutani/mahjonglm-1m",
        gguf_repo="mitsutani/mahjonglm-1m-q4-k-m-gguf",
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
            "8192",
        ),
    ),
    ModelSpec(
        key="10m",
        label="MahjongLM 10M",
        raw_repo="mitsutani/mahjonglm-10m",
        gguf_repo="mitsutani/mahjonglm-10m-q4-k-m-gguf",
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
            "8192",
        ),
    ),
    ModelSpec(
        key="100m",
        label="MahjongLM 100M",
        raw_repo="mitsutani/mahjonglm-100m",
        gguf_repo="mitsutani/mahjonglm-100m-q4-k-m-gguf",
        model_args=("--qwen-arch", "Q100"),
    ),
)


def now_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def dataset_args() -> list[str]:
    args: list[str] = []
    for year in range(2011, 2025):
        args.extend(["--dataset-dir", str(ROOT / "data" / "huggingface_datasets" / str(year))])
    return args


def run_checked(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
    capture_to_log: bool = True,
) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if log_file is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        if not capture_to_log:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
            return
        subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)


def append_log(log_file: Path, message: str) -> None:
    print(message, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(message + "\n")
        log.flush()


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if drive:
        tail = resolved.relative_to(resolved.anchor).as_posix()
        return f"/mnt/{drive}/{tail}"
    completed = subprocess.run(
        ["wsl.exe", "wslpath", "-a", str(resolved)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_training_via_powershell(cmd: list[str], *, run_root: Path, output_dir: Path, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")

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
        "8192",
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
        spec.key,
        "--stop-file",
        str(stop_file),
    ]


def stage_raw_model(spec: ModelSpec, output_dir: Path, stage_root: Path) -> Path:
    final_model = output_dir / "final_model"
    tokenizer_dir = output_dir / "tokenizer"
    if not final_model.exists():
        raise FileNotFoundError(f"missing final model: {final_model}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_dir}")

    stage_dir = stage_root / f"{spec.key}-raw"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    shutil.copytree(final_model, stage_dir)
    for tokenizer_file in tokenizer_dir.iterdir():
        if tokenizer_file.is_file():
            shutil.copy2(tokenizer_file, stage_dir / tokenizer_file.name)
    for metadata_name in ("training_config.json", "model_config.json"):
        metadata_path = output_dir / metadata_name
        if metadata_path.exists():
            shutil.copy2(metadata_path, stage_dir / metadata_name)
    (stage_dir / "README.md").write_text(raw_readme(spec), encoding="utf-8")
    return stage_dir


def raw_readme(spec: ModelSpec) -> str:
    return f"""---
language:
- en
- ja
tags:
- qwen3
- mahjong
- causal-lm
license: mit
---

# {spec.label}

Plain Qwen3 MahjongLM checkpoint trained with explicit `<bos>` prepended and `<eos>` appended by the training collator.

Training sweep:
- W&B project: `{WANDB_PROJECT}`
- data: all-year MahjongLM dataset, 2011-2024
- train epochs: 0.2
- train/eval batch size: 8
- gradient accumulation: 64
- optimizer: Muon+
- learning rate: 0.03
- warmup steps: 200
- label smoothing: 0

Prompt example:

```text
<bos> rule_player_4 rule_length_hanchan view_complete game_start
```
"""


def convert_and_quantize(spec: ModelSpec, raw_stage: Path, stage_root: Path, log_file: Path) -> Path:
    gguf_stage = stage_root / f"{spec.key}-gguf"
    gguf_stage.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_stage / f"{spec.key}-bos-F16.gguf"
    q4_path = gguf_stage / f"{spec.key}-bos-Q4_K_M.gguf"

    run_checked(
        [
            str(PYTHON),
            str(CONVERT_GGUF),
            str(raw_stage),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
            "--model-name",
            spec.label,
        ],
        log_file=log_file,
    )
    run_checked(
        [
            "wsl.exe",
            "bash",
            "-lc",
            f"'{LLAMA_QUANTIZE_WSL}' '{to_wsl_path(f16_path)}' '{to_wsl_path(q4_path)}' Q4_K_M",
        ],
        log_file=log_file,
    )

    if WORDLEVEL_PATCH_SOURCE.exists():
        patch_dir = gguf_stage / "patches"
        patch_dir.mkdir(exist_ok=True)
        shutil.copy2(WORDLEVEL_PATCH_SOURCE, patch_dir / WORDLEVEL_PATCH_SOURCE.name)
    if WORDLEVEL_PATCH_COMMIT_SOURCE.exists():
        shutil.copy2(WORDLEVEL_PATCH_COMMIT_SOURCE, gguf_stage / WORDLEVEL_PATCH_COMMIT_SOURCE.name)
    (gguf_stage / "README.md").write_text(gguf_readme(spec, q4_path.name), encoding="utf-8")
    if f16_path.exists():
        f16_path.unlink()
    return gguf_stage


def gguf_readme(spec: ModelSpec, gguf_name: str) -> str:
    return f"""---
language:
- en
- ja
tags:
- gguf
- qwen3
- mahjong
- llama.cpp
license: mit
---

# {spec.label} GGUF Q4_K_M

This repository contains the `Q4_K_M` GGUF export of `{spec.raw_repo}`.

GGUF file:
- `{gguf_name}`

Prompt example:

```text
<bos> rule_player_4 rule_length_hanchan view_complete game_start
```

This model uses a custom WordLevel Mahjong tokenizer. The GGUF includes tokenizer metadata, but upstream `llama.cpp` needs the small WordLevel tokenizer patch included in `patches/` before running this file.
"""


def upload_folder(folder: Path, repo_id: str, *, commit_message: str, log_file: Path | None = None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if log_file is not None:
        append_log(log_file, f"=== create repo {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False, token=token)
    if log_file is not None:
        append_log(log_file, f"=== upload folder {folder} -> {repo_id} ===")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message,
        token=token,
    )
    if log_file is not None:
        append_log(log_file, f"=== add collection item {repo_id} ===")
    api.add_collection_item(HF_COLLECTION, item_id=repo_id, item_type="model", exists_ok=True, token=token)
    if log_file is not None:
        append_log(log_file, f"=== upload complete {repo_id} ===")


def postprocess_and_upload_existing(spec: ModelSpec, output_dir: Path, run_root: Path) -> None:
    log_file = run_root / f"{spec.key}.log"
    stage_root = run_root / "hf_staging" / spec.key
    stage_root.mkdir(parents=True, exist_ok=True)

    append_log(log_file, f"=== staging raw {spec.key} from existing output {output_dir} ===")
    raw_stage = stage_raw_model(spec, output_dir, stage_root)

    append_log(log_file, f"=== converting gguf {spec.key} ===")
    gguf_stage = convert_and_quantize(spec, raw_stage, stage_root, log_file)

    append_log(log_file, f"=== uploading raw {spec.key} -> {spec.raw_repo} ===")
    upload_folder(raw_stage, spec.raw_repo, commit_message=f"Upload {spec.label} raw checkpoint", log_file=log_file)

    append_log(log_file, f"=== uploading gguf {spec.key} -> {spec.gguf_repo} ===")
    upload_folder(gguf_stage, spec.gguf_repo, commit_message=f"Upload {spec.label} Q4_K_M GGUF", log_file=log_file)


def run_model(spec: ModelSpec, run_root: Path) -> None:
    run_name = f"q{spec.key}-baseline-bos-allyears-0p2ep-{now_slug()}"
    output_dir = ROOT / "outputs" / run_name
    stop_file = output_dir / "STOP"
    log_file = run_root / f"{spec.key}.log"
    stage_root = run_root / "hf_staging" / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    print(f"=== training {spec.key}: {run_name} ===", flush=True)
    run_training_via_powershell(
        build_train_command(spec, output_dir, run_name, stop_file),
        run_root=run_root,
        output_dir=output_dir,
        log_file=log_file,
    )

    print(f"=== staging raw {spec.key} ===", flush=True)
    raw_stage = stage_raw_model(spec, output_dir, stage_root)

    print(f"=== converting gguf {spec.key} ===", flush=True)
    gguf_stage = convert_and_quantize(spec, raw_stage, stage_root, log_file)

    print(f"=== uploading raw {spec.key} -> {spec.raw_repo} ===", flush=True)
    upload_folder(raw_stage, spec.raw_repo, commit_message=f"Upload {spec.label} raw checkpoint", log_file=log_file)

    print(f"=== uploading gguf {spec.key} -> {spec.gguf_repo} ===", flush=True)
    upload_folder(gguf_stage, spec.gguf_repo, commit_message=f"Upload {spec.label} Q4_K_M GGUF", log_file=log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BOS-fixed 1M/10M/100M sweep and upload raw + Q4_K_M GGUF.")
    parser.add_argument("--models", nargs="+", choices=[spec.key for spec in MODEL_SPECS], default=[spec.key for spec in MODEL_SPECS])
    parser.add_argument("--run-root", type=Path, default=ROOT / "outputs" / f"bos_size_sweep_pipeline_{now_slug()}")
    parser.add_argument(
        "--postprocess-existing",
        action="append",
        default=[],
        metavar="KEY=OUTPUT_DIR",
        help="Stage, convert, and upload an already trained model before running selected models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    if not CONVERT_GGUF.exists():
        raise FileNotFoundError(CONVERT_GGUF)
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        print("WARNING: HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set; upload will require a cached Hugging Face login.", flush=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    specs_by_key = {spec.key: spec for spec in MODEL_SPECS}
    for item in args.postprocess_existing:
        if "=" not in item:
            raise ValueError(f"--postprocess-existing must be KEY=OUTPUT_DIR: {item}")
        key, output_dir = item.split("=", 1)
        if key not in specs_by_key:
            raise ValueError(f"unknown model key in --postprocess-existing: {key}")
        postprocess_and_upload_existing(specs_by_key[key], Path(output_dir), args.run_root)
    selected = {key for key in args.models}
    for spec in MODEL_SPECS:
        if spec.key in selected:
            run_model(spec, args.run_root)
    print("=== complete ===", flush=True)


if __name__ == "__main__":
    main()
