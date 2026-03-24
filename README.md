# MahjongLM

This repository prepares the 2021 MahjongLM tokenized dataset for GPT-2 training.

## Layout

- `data/raw/2021/` - original split ZIP archives
- `data/processed/2021/` - extracted Hugging Face dataset files
- `scripts/prepare_dataset.ps1` - repeatable extraction helper
- `scripts/train_gpt2.py` - convenience wrapper for the main GPT-2 training module
- `src/gpt2/` - upstream-style GPT-2 training stack with sampler/collator logic
- `src/tenhou_tokenizer/` - tokenizer loader used by the training stack
- `docs/quantization_and_attention.md` - notes on flash attention and 4-bit fine-tuning

## Data status

The 2021 dataset is already extracted locally and ready to load from:

`data/processed/2021`

The dataset is pre-tokenized and stored as a Hugging Face `Dataset` on disk.

## Training

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you want GPU acceleration on NVIDIA hardware, use the CUDA wheel set:

```powershell
pip install -r requirements-gpu.txt
```

Then verify with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

For wandb monitoring:

```powershell
wandb login
```

Prepare the dataset again if needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_dataset.ps1
```

Start GPT-2 training:

```powershell
python scripts/train_gpt2.py --output-dir outputs/gpt2-mahjong-2021
```

The canonical A/B/C/D comparison launcher is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_abcd_1400.ps1
```

To train the adopted Qwen3 `Q1` model:

```powershell
python scripts/train_qwen3.py --model-family qwen3 --qwen-arch Q1 --output-dir outputs/qwen3-q1
```

To compare matched 100M-class GPT-2 and Qwen3 bodies, use the built-in presets:

```powershell
python scripts/train_gpt2.py --model-family gpt2 --arch G100
python scripts/train_qwen3.py --model-family qwen3 --qwen-arch Q100
python scripts/measure_100m_model_pairs.py
```

To reproduce the Unsloth memory probe against the saved Q1 checkpoint:

```powershell
python scripts/experiment_unsloth_q1.py --checkpoint-dir outputs/qwen3-q1-init/model --seq-length 8192 --batch-size 1
```

For a short smoke test, keep the dataset tiny and disable wandb:

```powershell
python scripts/train_gpt2.py --output-dir outputs/smoke --wandb-mode disabled --max-train-groups 16 --max-eval-groups 4 --train-steps 16 --eval-interval 8
```

To monitor a run in wandb with a custom name:

```powershell
python scripts/train_gpt2.py --wandb-entity a21-3jck- --wandb-project mahjongLM_gpt2 --wandb-run-name gpt2-A-y2021-20260320-120000
```

If an online wandb run gets stuck in `running` or `stopping` after the process is already dead, recover it like this:

```powershell
python scripts/recover_wandb_run.py --entity a21-3jck- --project mahjongLM_qwen3 --run-id 0e96pgua --run-id gmwn0sf2
```

This creates a backup under `wandb_recovery/`, deletes the stale server runs, and re-uploads the local data as new finished runs.

For faster attention on GPU, use:

```powershell
python scripts/train_gpt2.py --attn-implementation sdpa
```

For the current packed MahjongLM training path, keep using `sdpa`. `flash_attention_2` is not compatible with the dense packed attention mask used by `PackedGroupCollator`.

## Sweep

The 100M-class candidate table and sweep setup are saved here:

- [docs/gpt2_hparam_candidates.md](docs/gpt2_hparam_candidates.md)
- [sweeps/gpt2_arch_sweep.yaml](sweeps/gpt2_arch_sweep.yaml)

To launch the sweep:

```powershell
wandb sweep sweeps/gpt2_arch_sweep.yaml
wandb agent a21-3jck-/mahjongLM_gpt2/<sweep_id>
```

Each sweep run is written under `outputs/gpt2-sweep/<wandb-run-name>/`.

## Notes

- The model is trained from scratch on the token IDs already present in the dataset.
- The script defaults to `tokenizer/` and loads `tokenizer.json` directly with `PreTrainedTokenizerFast`; if that path is missing, it falls back to `vocab.txt`, then to the maximum `input_ids` value in the dataset.
- The canonical training stack uses `max_seq_length=8192`, `max_tokens_per_batch=65536`, `PackedGroupCollator`, and `TokenBudgetGroupBatchSampler`.
- The wrapper script `scripts/train_gpt2.py` forwards to `src/gpt2/train.py`, so the upstream-style training path is the default again.
- `sdpa` is the default attention backend and the practical choice for the current packed pretraining path.
- The current `src/gpt2/train.py` CLI does not expose a 4-bit/LoRA fine-tuning path.
- Qwen3 work is currently standardized on the `Q1` body: `32` layers, `1024` hidden size, `3072` intermediate size, `16` attention heads, `8` KV heads, `128` head dim, and `8192` context.
- Unsloth is currently a Q1 investigation and later fine-tuning tool, not the default engine for dense packed Q1 pretraining.
- The dataset does not include a validation split, so the training script creates one deterministically.
- The train/eval split is cached under `data/cache/splits/`, so repeated runs do not rebuild it.
- Smoke runs that use `--max-train-groups` or `--max-eval-groups` are trimmed before splitting, so they avoid hidden full-dataset split/cache work.
- Windows subprocess eval is intentionally disabled in the training stack because it has been unstable with CUDA in this workspace.
- The dense packed attention mask in `PackedGroupCollator` is still the main remaining long-context throughput bottleneck.
- Candidate GPT-2 settings are saved in [docs/gpt2_hparam_candidates.md](docs/gpt2_hparam_candidates.md).
- The adopted Qwen3 configuration is saved in [docs/qwen3_0.5b_candidates.md](docs/qwen3_0.5b_candidates.md).
- Unsloth investigation notes are saved in [docs/qwen3_unsloth_q1.md](docs/qwen3_unsloth_q1.md).
