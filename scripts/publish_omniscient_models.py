from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
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

WANDB_PROJECT = "mahjongLM_qwen3_plain_omniscient_sweep"
DATASET_REPO = "mitsutani/mahjonglm-dataset"
HF_COLLECTION = "mitsutani/mahjonglm"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    raw_repo: str
    gguf_repo: str


MODEL_SPECS = {
    "1m": ModelSpec(
        key="1m",
        label="MahjongLM 1M",
        raw_repo="mitsutani/mahjonglm-1m",
        gguf_repo="mitsutani/mahjonglm-1m-q4-k-m-gguf",
    ),
    "10m": ModelSpec(
        key="10m",
        label="MahjongLM 10M",
        raw_repo="mitsutani/mahjonglm-10m",
        gguf_repo="mitsutani/mahjonglm-10m-q4-k-m-gguf",
    ),
    "100m": ModelSpec(
        key="100m",
        label="MahjongLM 100M",
        raw_repo="mitsutani/mahjonglm-100m",
        gguf_repo="mitsutani/mahjonglm-100m-q4-k-m-gguf",
    ),
}

QWEN3_CONFIG_KEYS = {
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "dtype",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "max_window_layers",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_parameters",
    "sliding_window",
    "tie_word_embeddings",
    "transformers_version",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
}


def run_checked(cmd: list[str], *, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    line = "+ " + " ".join(cmd)
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        handle.write(process.stdout)
        handle.flush()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{rest}"


def sanitize_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sanitized = {key: config[key] for key in QWEN3_CONFIG_KEYS if key in config}
    config_path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_readme(spec: ModelSpec, output_dir: Path) -> str:
    metrics = read_summary_metrics(output_dir)
    metric_lines = ""
    if metrics:
        metric_lines = "\n".join(f"- {name}: `{value}`" for name, value in metrics.items()) + "\n"
    if not metric_lines:
        metric_lines = "- Final metrics are recorded in the corresponding W&B run.\n"
    return f"""---
language:
- ja
- en
tags:
- qwen3
- mahjong
- causal-lm
- tenhou
- wordlevel-tokenizer
license: other
---

# {spec.label}

{spec.label} is a compact Qwen3-style causal language model trained for Mahjong game-log token generation.
It uses the MahjongLM WordLevel tokenizer and a 16,384-token context window.

## Training

- Dataset: [`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO})
- Source data span: Tenhou game logs from 2011 through 2024
- Views: `view_complete`, `view_imperfect_0` to `view_imperfect_3`, and `view_omniscient`
- Omniscient view: adds a `wall` block immediately after `round_start`, followed by the reconstructed 136-tile wall including red fives (`m0`, `p0`, `s0`)
- Architecture: plain Qwen3 causal LM, without XSA, gated attention, attention residuals, or Mamba
- Training length: 0.2 epoch
- Batch settings: train/eval batch size 8, gradient accumulation 64
- Optimizer: Muon+
- Learning rate: 0.03, linear decay, 200 warmup steps
- W&B project: `{WANDB_PROJECT}`

## Metrics

{metric_lines}
## Prompt Format

Training adds `<bos>` at the beginning and `<eos>` at the end. A prompt should therefore usually start with `<bos>`, followed by the rule tokens, one view token, then `game_start`.

Basic four-player complete-information prompt:

```text
<bos> rule_player_4 rule_length_hanchan view_complete game_start
```

Four-player imperfect-information prompt from player 0's perspective:

```text
<bos> rule_player_4 rule_length_hanchan view_imperfect_0 game_start
```

Omniscient prompt:

```text
<bos> rule_player_4 rule_length_hanchan view_omniscient game_start round_start wall
```

For `view_omniscient`, the model is expected to continue the `wall` block with 136 tile tokens before the normal round metadata. Tile tokens are the 34 tile types plus red fives: `m0`, `p0`, and `s0`.

## Token Grammar Notes

- `rule_player_3` / `rule_player_4` describes sanma or yonma.
- `rule_length_tonpu` / `rule_length_hanchan` describes game length.
- `round_start` opens a hand and `round_end` closes the hand result section.
- `game_start` and `game_end` delimit the full game log.
- Actions are represented by `opt_*`, then matching `take_*` or `pass_*` decisions in the same option order.
- Win details begin with `hule_{{seat}}`; score deltas are emitted once after all win details.
"""


def gguf_readme(spec: ModelSpec, gguf_name: str) -> str:
    return f"""---
language:
- ja
- en
tags:
- gguf
- qwen3
- mahjong
- llama.cpp
- wordlevel-tokenizer
license: other
---

# {spec.label} Q4_K_M GGUF

This repository contains the Q4_K_M GGUF export of [`{spec.raw_repo}`](https://huggingface.co/{spec.raw_repo}).

GGUF file:

- `{gguf_name}`

## Training

The source model was trained on [`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO}), using complete, imperfect, and omniscient MahjongLM views over Tenhou logs from 2011 through 2024.

## Prompt Format

```text
<bos> rule_player_4 rule_length_hanchan view_complete game_start
```

For `view_omniscient`, use:

```text
<bos> rule_player_4 rule_length_hanchan view_omniscient game_start round_start wall
```

and then provide or generate the 136 wall tile tokens.

## llama.cpp WordLevel Tokenizer Patch

MahjongLM uses a fixed WordLevel tokenizer over Mahjong log tokens rather than a byte-pair or sentencepiece tokenizer.
The GGUF file contains the vocabulary metadata, but upstream llama.cpp builds may not know how to interpret this tokenizer type.

This repository includes the minimal llama.cpp patch in `patches/`:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git apply /path/to/patches/0001-wordlevel-tokenizer-support.patch
cmake -B build
cmake --build build --config Release
```

After building the patched binary, run the model normally with `llama-cli` or compatible llama.cpp tools. The patch is tokenizer-only; the model architecture is plain Qwen3.
"""


def read_summary_metrics(output_dir: Path) -> dict[str, object]:
    metrics_path = output_dir / "training_config.json"
    if not metrics_path.exists():
        return {}
    summary_candidates = sorted((ROOT / "wandb").glob("run-*/files/wandb-summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    run_name = output_dir.name
    for summary in summary_candidates:
        config_path = summary.parent / "config.yaml"
        if config_path.exists() and run_name in config_path.read_text(encoding="utf-8", errors="ignore"):
            payload = json.loads(summary.read_text(encoding="utf-8"))
            selected: dict[str, object] = {}
            for key in ("run/parameter_count", "trainer/global_step", "eval/perplexity", "eval/loss"):
                if key in payload:
                    value = payload[key]
                    if isinstance(value, dict) and "max" in value:
                        value = value["max"]
                    selected[key] = value
            return selected
    return {}


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
    sanitize_config(stage_dir / "config.json")
    (stage_dir / "README.md").write_text(raw_readme(spec, output_dir), encoding="utf-8")
    return stage_dir


def convert_and_quantize(spec: ModelSpec, raw_stage: Path, stage_root: Path, log_file: Path) -> Path:
    gguf_stage = stage_root / f"{spec.key}-gguf"
    if gguf_stage.exists():
        shutil.rmtree(gguf_stage)
    gguf_stage.mkdir(parents=True)
    f16_path = gguf_stage / f"mahjonglm-{spec.key}-F16.gguf"
    q4_path = gguf_stage / f"mahjonglm-{spec.key}-Q4_K_M.gguf"

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


def upload_folder(folder: Path, repo_id: str, *, commit_message: str, log_file: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"=== create repo {repo_id} ===\n")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False, token=token)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"=== upload folder {folder} -> {repo_id} ===\n")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message,
        token=token,
    )
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"=== add collection item {repo_id} ===\n")
    api.add_collection_item(HF_COLLECTION, item_id=repo_id, item_type="model", exists_ok=True, token=token)


def publish_existing(spec: ModelSpec, output_dir: Path, run_root: Path) -> None:
    log_file = run_root / f"publish_{spec.key}.log"
    stage_root = run_root / "hf_staging" / spec.key
    stage_root.mkdir(parents=True, exist_ok=True)

    raw_stage = stage_raw_model(spec, output_dir, stage_root)
    gguf_stage = convert_and_quantize(spec, raw_stage, stage_root, log_file)
    upload_folder(raw_stage, spec.raw_repo, commit_message=f"Update {spec.label}", log_file=log_file)
    upload_folder(gguf_stage, spec.gguf_repo, commit_message=f"Update {spec.label} Q4_K_M GGUF", log_file=log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish omniscient MahjongLM raw and Q4_K_M GGUF models.")
    parser.add_argument("--run-root", type=Path, default=ROOT / "outputs" / "omniscient_publish")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="KEY=OUTPUT_DIR",
        help="Publish an existing model output directory. KEY is one of 1m, 10m, 100m.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if not CONVERT_GGUF.exists():
        raise FileNotFoundError(CONVERT_GGUF)
    for item in args.model:
        key, output = item.split("=", 1)
        if key not in MODEL_SPECS:
            raise ValueError(f"unknown model key: {key}")
        publish_existing(MODEL_SPECS[key], Path(output), args.run_root)


if __name__ == "__main__":
    main()
