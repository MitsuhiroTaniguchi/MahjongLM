# Outcome-conditioned Unsloth fine-tuning

This path fine-tunes an existing MahjongLM checkpoint on the public pretraining
dataset while leaking the final result at the beginning of each imperfect-view
sequence.

For every `view_imperfect` row, the collator builds:

```text
<bos> final_rank_* final_rank_* ... original_input_ids <eos>
```

The original dataset row is otherwise unchanged. The `final_rank_*` block is
copied from the original token sequence, and is validated to contain exactly
three distinct places for sanma and four distinct places for yonma.

## Install

Use a CUDA environment compatible with Unsloth, then install the optional stack:

```bash
pip install -r requirements-dpo.txt
```

Log in before launching if the dataset or model is gated:

```bash
huggingface-cli login
wandb login
```

## Launch

```bash
PYTHONPATH=src python scripts/train_outcome_conditioned_unsloth.py \
  --model-name mitsutani/mahjonglm-10m \
  --dataset-repo mitsutani/mahjonglm-dataset \
  --output-dir outputs/mahjonglm-10m-outcome-conditioned \
  --require-wandb \
  --wandb-mode online \
  --wandb-project mahjongLM_outcome_conditioned \
  --wandb-run-name mahjonglm-10m-final-rank-prefix \
  --num-train-epochs 0.2
```

By default this downloads and trains on yearly dataset folders `2011` through
`2024`, filtering to `view_type == "imperfect"`, for `0.2` epochs. Periodic eval is capped
to 1024 games by default so `eval_steps` does not scan the full held-out split; pass
`--max-eval-groups 0` only when you explicitly want full evaluation. To test a smaller run first:

```bash
PYTHONPATH=src python scripts/train_outcome_conditioned_unsloth.py \
  --model-name mitsutani/mahjonglm-10m \
  --dataset-year 2024 \
  --max-train-groups 100 \
  --max-eval-groups 20 \
  --max-steps 20 \
  --require-wandb
```

If a conditioned sequence exceeds the model context length, the default behavior
is to fail. Use `--drop-overlong` only when you explicitly accept excluding those
games instead of changing the original token sequence.
