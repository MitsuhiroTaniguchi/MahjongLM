# Mamba-3 Hybrid 100M Candidates

Target:
- Preserve the current Qwen3/XSA baseline attention geometry:
  - `hidden_size=768`
  - `intermediate_size=2304`
  - `num_attention_heads=12`
  - `num_key_value_heads=6`
  - `head_dim=64`
- Replace non-attention layers with Mamba-3 in a `3:1` pattern (`mamba3_attention_period=4`)
- Keep the hybrid close to `100M` parameters without forcing unusually large Mamba widths

Primary sources:
- Mamba-2 / SSD: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- Mamba-3 official implementation defaults:
  - `d_state=128`
  - `expand=2`
  - `headdim=64`
  - `ngroups=1`
  - [official module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

Interpretation for this repo:
- Forcing the current `14`-layer Qwen baseline into a hybrid keeps too few parameters unless the Mamba block is made unnaturally wide.
- The more natural 100M regime is to keep `expand=2` and `headdim=64`, then increase depth.

Measured candidates (repo-local exact parameter counts):
- `QM100S64`
  - `22 layers`
  - `mamba3_expand=2`
  - `mamba3_d_state=64`
  - params: `99,074,864`
- `QM100S96`
  - `22 layers`
  - `mamba3_expand=2`
  - `mamba3_d_state=96`
  - params: `100,042,096`
- `22 layers / d_state=128`
  - params: `101,009,328`
  - viable, but heavier than needed

Recommended defaults:
- SISO starting point: `QM100S96`
  - closest to 100M
  - keeps official-style `expand=2`
  - moderate `d_state`
- safer memory fallback: `QM100S64`
  - slightly under 100M
  - same depth and geometry

Why not `14` layers?
- With `14` layers and the current `3:1` replacement pattern, `expand=2` lands far below 100M.
- Reaching 100M at `14` layers requires `expand=4`, which is harder to justify from the official defaults and is likely worse for memory at fixed `bs=40`.
