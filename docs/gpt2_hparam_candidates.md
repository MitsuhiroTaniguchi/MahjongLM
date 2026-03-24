# GPT-2 Candidate Settings

This table records the main 100M-class GPT-2 candidates for MahjongLM.

## Candidate Defaults

- `dataset-dir`: `data/processed/2021`
- `tokenizer-dir`: `tokenizer`
- `vocab_size`: inferred from `tokenizer/tokenizer.json` via `PreTrainedTokenizerFast`
- `max_seq_length`: `8192`
- `max_tokens_per_batch`: `65536`
- `eval_max_tokens_per_batch`: `65536`
- `learning_rate`: `3e-4`
- `warmup_steps`: `100`
- `weight_decay`: `0.1`
- `scheduler`: `cosine`
- `optimizer`: `AdamW`
- `precision`: `bf16`
- `train_steps`: choose per experiment budget
- `sampler`: `TokenBudgetGroupBatchSampler`
- `collator`: `PackedGroupCollator`
- `wandb_entity`: `a21-3jck-`
- `wandb_project`: `mahjongLM_gpt2`

These are architecture-comparison defaults, not a promise that every launcher uses these exact values.
The current comparison launcher is the source of truth for the live A/B/C/D protocol.

## Main Candidates

The parameter counts below are measured from `transformers.GPT2LMHeadModel` with `vocab_size=815` and `n_positions=8192`.

| Label | Layers | Width | Heads | Params | Notes |
|---|---:|---:|---:|---:|---|
| A | 20 | 640 | 10 | 99,648,640 | Closest to the 100M cap without exceeding it |
| B | 10 | 896 | 14 | 98,103,936 | Wide baseline near the cap |
| C | 16 | 704 | 11 | 96,600,768 | Balanced depth/width candidate |
| D | 24 | 576 | 9 | 96,791,616 | Deep, narrower variant near the cap |

## Suggested Run Name Pattern

Pass `--wandb-run-name` explicitly if you want a human-readable name.

## Recommended Order

1. A
2. B
3. C
4. D
