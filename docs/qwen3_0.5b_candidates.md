# Qwen3 0.5B Candidates

Reference configuration:
- [Qwen3-0.6B-Base config.json](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)

The reference model uses:
- `hidden_size=1024`
- `intermediate_size=3072`
- `num_hidden_layers=28`
- `num_attention_heads=16`
- `num_key_value_heads=8`
- `head_dim=128`
- `hidden_act=silu`
- `rms_norm_eps=1e-6`
- `rope_theta=1000000`
- `tie_word_embeddings=true`

For MahjongLM, the tokenizer vocabulary is `815`, so the same Qwen3 body is much smaller than the original 0.6B release. To target roughly 0.5B parameters, the following Qwen3-family presets were measured with `transformers.Qwen3ForCausalLM` and the local tokenizer.

For the current project direction, `Q1` is the adopted configuration. The other rows are kept here as historical measurements only.

## Measured Presets

| Label | Layers | Hidden | MLP | Heads | KV Heads | Head Dim | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 31 | 1024 | 3072 | 16 | 8 | 128 | 488,494,848 |
| Q1 | 32 | 1024 | 3072 | 16 | 8 | 128 | 504,225,792 |
| Q2 | 28 | 1088 | 3264 | 17 | 8 | 128 | 494,228,992 |
| Q3 | 30 | 1056 | 3168 | 16 | 8 | 128 | 496,661,376 |
| Q4 | 25 | 1152 | 3456 | 18 | 9 | 128 | 498,668,032 |

## Recommendation

- Adopted model: `Q1`
  - keeps the original Qwen3 width, MLP ratio, GQA ratio, and head dim
  - reaches ~`504.2M` by increasing depth from `28` to `32`
- Historical alternatives:
  - `Q0` lands at ~`488.5M`
  - `Q2`, `Q3`, and `Q4` were exploratory comparisons
  - they are no longer the active target

## CLI

The adopted training entry point is:

```powershell
python scripts/train_gpt2.py --model-family qwen3 --qwen-arch Q1 --output-dir outputs/qwen3-q1
```

For custom Qwen3 shapes:

```powershell
python scripts/train_gpt2.py `
  --model-family qwen3 `
  --qwen-arch custom `
  --qwen-hidden-size 1024 `
  --qwen-intermediate-size 3072 `
  --qwen-num-hidden-layers 32 `
  --qwen-num-attention-heads 16 `
  --qwen-num-key-value-heads 8 `
  --qwen-head-dim 128 `
  --qwen-max-position-embeddings 8192 `
  --output-dir outputs/qwen3-custom
```

To re-measure preset parameter counts:

```powershell
python scripts/measure_qwen3_params.py
```

Unsloth investigation notes for `Q1` are saved in `docs/qwen3_unsloth_q1.md`.
