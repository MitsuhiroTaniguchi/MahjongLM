# Quantization and Attention Notes

This repository currently exposes one practical acceleration path in the packed GPT-2 trainer:

- `attn_implementation=sdpa` for the current packed GPT-2 pretraining path

## Flash attention

For scratch training, the safest fast path is:

```powershell
python scripts/train_gpt2.py --attn-implementation sdpa
```

On modern PyTorch/CUDA builds, `sdpa` can dispatch to fused flash-style kernels automatically.
The current MahjongLM packed pretraining path uses a dense packed attention mask, so `flash_attention_2` is not supported here even if the external package is installed.

## 4-bit quantization

4-bit quantization is not part of the current `src/gpt2/train.py` CLI.
The optional dependency file is kept only for future checkpoint fine-tuning experiments outside the packed pretraining path.

## Recommendation

For this repository's current GPT-2 training, use:

- `--attn-implementation sdpa` as the default
- the packed dense attention mask remains the main throughput bottleneck at long context lengths
