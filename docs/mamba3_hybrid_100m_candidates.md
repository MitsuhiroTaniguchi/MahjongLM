# Mamba-3 Hybrid Candidates Near 100M

Goal:
- Preserve the current strong Qwen/XSA baseline geometry:
  - `hidden_size=768`
  - `intermediate_size=2304`
  - `num_attention_heads=12`
  - `num_key_value_heads=6`
  - `head_dim=64`
- Replace layers in an exact `3:1` hybrid pattern
  - `mamba3_attention_period=4`
  - therefore use a layer count divisible by `4`

Primary references:
- Mamba-2 / SSD: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- Mamba-3 official implementation defaults:
  - `d_state=128`
  - `expand=2`
  - `headdim=64`
  - `ngroups=1`
  - [official module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

Why the earlier `22L` idea was wrong:
- The current hybrid implementation keeps every 4th layer as attention.
- With `22` layers that gives `17` Mamba + `5` attention, not an exact `3:1`.
- Exact `3:1` requires `16`, `20`, `24`, ... layers.

Measured exact-3:1 candidates (repo-local parameter counts):

## `QM95S192`
- `20` layers
- `15` Mamba + `5` attention
- `mamba3_expand=2`
- `mamba3_d_state=192`
- params: `95,070,928`
- Interpretation:
  - closest candidate that keeps the official-style `expand=2`
  - more conservative for memory than forcing an exact 100M count

## `QM100S384`
- `20` layers
- `15` Mamba + `5` attention
- `mamba3_expand=2`
- `mamba3_d_state=384`
- params: `100,191,568`
- Interpretation:
  - exact-3:1 and very close to 100M
  - but `d_state=384` is far above the official default and likely expensive in memory/state size

## `QM99E3S256`
- `16` layers
- `12` Mamba + `4` attention
- `mamba3_expand=3`
- `mamba3_d_state=256`
- params: `99,187,040`
- Interpretation:
  - also close to 100M
  - but reaches it by widening the Mamba block beyond the official `expand=2` default

Recommendation:
- Practical starting point: `QM95S192`
  - exact `3:1`
  - stays closer to the official `expand=2` design
  - safer for fixed `bs=40`
- Exact-count experiment: `QM100S384`
  - use if matching ~100M matters more than keeping Mamba internals moderate
- Alternative width-heavy experiment: `QM99E3S256`
  - use if we specifically want to test whether larger `expand` scales better than larger `d_state`

Current launcher default:
- [launch_qwen_mamba_wsl.ps1](/C:/Users/taniguchi/MahjongLM/scripts/launch_qwen_mamba_wsl.ps1)
- points to `QM95S192`
