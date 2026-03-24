# Qwen3 Q1 Unsloth Investigation

Reference:
- [Qwen3-0.6B-Base config.json](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)
- [Unsloth releases](https://github.com/unslothai/unsloth/releases)
- [PyTorch Triton `triton_key` issue](https://github.com/pytorch/pytorch/issues/127271)

## Goal

Check whether Unsloth can materially reduce memory for the adopted `Q1` model in this repository.

Current Q1 shape:
- `32` layers
- `1024` hidden size
- `3072` intermediate size
- `16` attention heads
- `8` KV heads
- `128` head dim
- `8192` context
- `504,225,792` parameters

## Important Scope

There are two different questions here:

1. Can Unsloth reduce memory for a standard Hugging Face `Qwen3ForCausalLM` training step?
2. Can Unsloth replace the current packed MahjongLM pretraining stack?

The answer to `1` is partially yes, but only for LoRA-style training.
The answer to `2` is currently no.

Why not:
- The current trainer uses `PackedGroupCollator` with a dense packed attention mask.
- Unsloth patches the standard Hugging Face causal LM path, not this packed mask path.
- The main long-context bottleneck in this repository is still the dense packed mask itself.

## Experiment Setup

Script:

```powershell
python scripts/experiment_unsloth_q1.py --checkpoint-dir outputs/qwen3-q1-init/model --seq-length 8192 --batch-size 1
```

Modes:
- `hf_bf16`: plain Hugging Face bf16 dense training
- `unsloth_full16`: Unsloth full finetuning with all Q1 parameters trainable
- `unsloth_4bit_lora`: Unsloth 4-bit base + LoRA adapters

Checkpoint used:
- local random-init Q1 checkpoint at `outputs/qwen3-q1-init/model`

## Results

### seq_length=4096, batch_size=1

| Mode | Trainable Params | Peak Allocated GiB | Step Time Sec | Notes |
|---|---:|---:|---:|---|
| HF bf16 dense | 504,225,792 | 6.11 | 1.53 | Baseline |
| Unsloth full16 | 504,225,792 | 8.17 | 4.87 | Heavier and slower than baseline |
| Unsloth 4bit + LoRA | 11,534,336 | 0.60 | 1.90 | Much smaller, but not full-model training |

### seq_length=8192, batch_size=1

| Mode | Trainable Params | Peak Allocated GiB | Step Time Sec | Notes |
|---|---:|---:|---:|---|
| HF bf16 dense | 504,225,792 | 18.33 | 5.04 | Baseline |
| Unsloth full16 | 504,225,792 | 26.45 | 12.93 | Still heavier and slower |
| Unsloth 4bit + LoRA | 11,534,336 | 0.86 | 1.37 | Dramatically smaller, but LoRA-only |

## Interpretation

### What worked

- Unsloth imports and runs on this machine:
  - Windows
  - RTX 5090
  - `torch 2.8.0+cu129`
  - `unsloth 2026.3.8`
- Q1 can be loaded through Unsloth from a local checkpoint.
- Unsloth 4-bit + LoRA really does slash VRAM usage.

### What did not work for the actual project goal

- `unsloth_full16` did not beat plain Hugging Face bf16 dense training in these tests.
- More importantly, `4bit + LoRA` is not a drop-in replacement for from-scratch dense pretraining.

That last point is the key one:
- LoRA trains only adapters on top of a frozen base.
- If the base is random-init Q1, freezing it and training only LoRA adapters is not a sensible way to create a foundation model from scratch.

So while `4bit + LoRA` is excellent for later adaptation of an already-trained Q1 checkpoint, it is not the right way to build the first Q1 checkpoint.

## Practical Decision

For the current MahjongLM project:
- Keep dense Q1 pretraining on the existing `sdpa` training stack.
- Do not switch the packed pretraining path to Unsloth.
- Keep Unsloth available for later stages:
  - continued pretraining of an already-trained checkpoint
  - SFT / instruction tuning
  - low-memory LoRA experiments

## Reproducibility Artifacts

Probe outputs:
- `outputs/unsloth-q1-probe/seq4096_bs1_hf.json`
- `outputs/unsloth-q1-probe/seq4096_bs1_full16.json`
- `outputs/unsloth-q1-probe/seq4096_bs1_4bit_lora.json`
- `outputs/unsloth-q1-probe/seq8192_bs1_hf.json`
- `outputs/unsloth-q1-probe/seq8192_bs1_full16.json`
- `outputs/unsloth-q1-probe/seq8192_bs1_4bit_lora.json`
