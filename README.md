# MahjongLM

This repository prepares the 2021 MahjongLM tokenized dataset for GPT-2 training.

## Layout

- `data/raw/2021/` - original split ZIP archives
- `data/processed/2021/` - extracted Hugging Face dataset files
- `scripts/prepare_dataset.ps1` - repeatable extraction helper
- `scripts/train_gpt2.py` - GPT-2 training entry point

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
python scripts/train_gpt2.py --dataset-path data/processed/2021 --output-dir outputs/gpt2-mahjong-2021
```

To keep a short local smoke test on CPU, add `--use-cpu --report-to none`.

To monitor a run in wandb with a custom name:

```powershell
python scripts/train_gpt2.py --wandb-entity a21-3jck- --wandb-project mahjongLM_gpt2 --wandb-run-name gpt2-2021-smoke
```

If you omit `--wandb-run-name`, the script auto-generates one in this pattern:

`mahjonglm-y2021-gpt2-v815-l12-h12-d768-bs1024-s50-lr5e-4-gpu-0319-0000`

That keeps the dataset year, vocab size, model size, block size, step count, learning rate, device type, and a short timestamp together in one place.

## Notes

- The model is trained from scratch on the token IDs already present in the dataset.
- The script defaults to `tokenizer/` and loads `tokenizer.json` directly with `PreTrainedTokenizerFast`; if that path is missing, it falls back to `vocab.txt`, then to the maximum `input_ids` value in the dataset.
- The dataset does not include a validation split, so the training script creates one deterministically.
