# GPT-2 / Qwen3 100M Comparison Baseline

This note defines a like-for-like 100M-class comparison pair for isolating whether the current instability is mainly caused by the architecture or by optimization/training settings.

Common assumptions:

- `vocab_size = 815`
- `max_position_embeddings / n_positions = 8192`
- single-GPU causal LM pretraining in the current MahjongLM stack

## Adopted 100M pair

### GPT-2 `G100`

- `n_layer = 19`
- `n_embd = 640`
- `n_head = 10`
- `n_positions = 8192`
- parameter count (measured): `99,312,640`

### Qwen3 `Q100`

- `hidden_size = 640`
- `intermediate_size = 1920`
- `num_hidden_layers = 20`
- `num_attention_heads = 10`
- `num_key_value_heads = 5`
- `head_dim = 64`
- `max_position_embeddings = 8192`
- parameter count (measured): `98,854,400`

## Why this pair

- The parameter counts are close enough that throughput and optimization behavior are easier to compare directly.
- Both use the same tokenizer and the same 8192 context limit.
- `G100` is still a conventional GPT-2 body.
- `Q100` keeps the Qwen3-style grouped-query attention and RMSNorm/SwiGLU-style body while staying in the same parameter class.

## Re-measurement

Use:

```powershell
python scripts/measure_100m_model_pairs.py
```

This prints the current measured parameter counts from the actual local codebase.
