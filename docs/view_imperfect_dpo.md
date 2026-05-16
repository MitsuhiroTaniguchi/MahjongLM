# View-Imperfect DPO Fine-Tuning

This document describes the offline fine-tuning path for turning a MahjongLM pretraining checkpoint into a player-AI-oriented `view_imperfect_*` model.

## Goal

Pretraining learns a token distribution over Tenhou logs. DPO fine-tuning instead compares same-seed model rollouts and reinforces the rollout that gives the viewer a better final rank.

## Data Flow

1. `scripts/sample_dpo_omniscient_prompts.py`
   - Samples balanced `view_omniscient` prompt prefixes from the reconstructed all-year dataset.
   - Each prompt includes `<bos>` and the first `round_start wall ...` block.
   - Use `--per-rule` so rule buckets are sampled evenly.
2. `scripts/generate_dpo_rollouts.py`
   - Generates 8 sampled continuations for each same-seed prompt.
   - Writes generation batches to JSONL.
3. `scripts/build_dpo_preference_dataset.py`
   - Discards batches whose `final_rank_*` vector is identical across all continuations.
   - For each seat, chooses at most one better-vs-worse pair.
   - Converts both continuations from `view_omniscient` to `view_imperfect_{seat}`.
   - Saves `prompt`, `chosen`, and `rejected` rows for TRL/Unsloth DPO and can push them to Hugging Face.
4. `scripts/train_dpo_unsloth.py`
   - Runs LoRA DPO with Unsloth.
   - Forces W&B online mode when `--require-wandb` is set.

The convenience launcher `scripts/run_view_imperfect_dpo_pipeline.py` wires the four steps together, but it is explicit foreground work only. It does not stop or modify active pretraining runs.

## Example

```powershell
.\.venv\Scripts\python.exe scripts\run_view_imperfect_dpo_pipeline.py `
  --model-dir outputs\mahjonglm-100m\final_model `
  --dataset-dir data\huggingface_datasets_omniscient\2011 `
  --dataset-dir data\huggingface_datasets_omniscient\2012 `
  --hf-dataset-repo mitsutani/mahjonglm-view-imperfect-dpo
```

For a safer staged run, first generate rollouts, inspect them, then build the preference dataset:

```powershell
.\.venv\Scripts\python.exe scripts\sample_dpo_omniscient_prompts.py `
  --dataset-dir data\huggingface_datasets_omniscient\2024 `
  --output-jsonl outputs\dpo_probe\prompts.jsonl `
  --per-rule 4

.\.venv\Scripts\python.exe scripts\generate_dpo_rollouts.py `
  --model-dir outputs\mahjonglm-100m\final_model `
  --prompts-jsonl outputs\dpo_probe\prompts.jsonl `
  --output-jsonl outputs\dpo_probe\rollouts.jsonl `
  --batch-size 8 `
  --trust-remote-code

.\.venv\Scripts\python.exe scripts\build_dpo_preference_dataset.py `
  --generations-jsonl outputs\dpo_probe\rollouts.jsonl `
  --output-dir outputs\dpo_probe\preference_dataset `
  --repo-id mitsutani/mahjonglm-view-imperfect-dpo
```

## Notes

- `prompt` is currently `<bos>` and `chosen`/`rejected` contain the remainder of the full `view_imperfect_*` game through `<eos>`.
- Same-rank batches are intentionally discarded because they do not identify a seat-level preference.
- The conversion removes `wall`, masks other players' initial hands and draws, and keeps public actions/results.
- DPO dependencies are intentionally separated in `requirements-dpo.txt` because Unsloth and TRL are CUDA-version sensitive.
