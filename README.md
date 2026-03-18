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

Prepare the dataset again if needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_dataset.ps1
```

Start GPT-2 training:

```powershell
python scripts/train_gpt2.py --dataset-path data/processed/2021 --output-dir outputs/gpt2-mahjong-2021
```

## Notes

- The model is trained from scratch on the token IDs already present in the dataset.
- The script uses a GPT-2 style causal LM with `vocab_size=65536`, which is safe for the stored `uint16` token IDs.
- The dataset does not include a validation split, so the training script creates one deterministically.

