# GPT-2 Candidate Settings

This table records the main 100M-class GPT-2 candidates for MahjongLM.

## Fixed Training Settings

- `dataset-path`: `data/processed/2021`
- `tokenizer-path`: `tokenizer`
- `vocab_size`: inferred from `tokenizer/tokenizer.json` via `PreTrainedTokenizerFast`
- `block_size`: `1024`
- `learning_rate`: `3e-4`
- `warmup_ratio`: `0.01`
- `weight_decay`: `0.1`
- `scheduler`: `cosine`
- `optimizer`: `AdamW`
- `precision`: `bf16`
- `max_steps`: choose per experiment budget
- `report_to`: `wandb`
- `wandb_entity`: `a21-3jck-`
- `wandb_project`: `mahjongLM_gpt2`

## Main Candidates

The parameter counts below are measured from `transformers.GPT2LMHeadModel` with `vocab_size=815` and `n_positions=1024`.

| Label | Layers | Width | Heads | Params | Notes |
|---|---:|---:|---:|---:|---|
| A | 20 | 640 | 10 | 99,648,640 | Closest to the 100M cap without exceeding it |
| B | 10 | 896 | 14 | 98,103,936 | Wide baseline near the cap |
| C | 16 | 704 | 11 | 96,600,768 | Balanced depth/width candidate |
| D | 24 | 576 | 9 | 96,791,616 | Deep, narrower variant near the cap |

## Suggested Run Name Pattern

If `--wandb-run-name` is omitted, the script should generate names in this style:

`mahjonglm-y2021-gpt2-archA-v815-l20-h10-d640-bs1024-s50-lr3e-4-gpu-0319-0000`

## Recommended Order

1. A
2. B
3. C
4. D
