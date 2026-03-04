# pymahjong Upstream PR Notes (Fast APIs)

Updated: 2026-03-04

## Goal

Contribute tokenizer-oriented fast APIs to `pymahjong` upstream so this repository no longer needs a local patch workflow.

## Status

- PR: https://github.com/MitsuhiroTaniguchi/pymahjong/pull/4
- State: merged on 2026-03-04
- Merge commit: `0b80bafa99c5b4b8bc1ade256c5a86b679238576`
- Batch API PR: https://github.com/MitsuhiroTaniguchi/pymahjong/pull/5
- Batch API state: open (local patch kept until merged)
- Current local extension patch: `scripts/patches/pymahjong-batchapi.patch`
  - `has_hupai_multi`
  - `evaluate_draw`

## Proposed PR scope

Target file in upstream `pymahjong`:
- `src/bindings.cpp`

Additions:
- `wait_mask(hand, meld_count) -> int`
- `has_riichi_discard(hand, meld_count) -> bool`
- `has_hupai(hand, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban) -> bool`
- `Shoupai.tingpai_mask() -> int`
- `Shoupai.tingpai_list() -> List[int]`

Compatibility:
- additive only (no breaking API changes)
- existing Python APIs stay unchanged

## Why this helps

Tokenizer repeatedly checks:
- current waits
- riichi-discard availability
- win legality (ron/tsumo)

Direct C++ bindings reduce Python object construction and repeated slow-path calls.

## Local benchmark summary (consumer repo)

Condition:
- input: Tenhou 2023 zip
- sample: first 3000 games

Observed throughput:
- old path: `19.28 games/s`
- with fast APIs + tokenizer integration: `60.13 games/s`

Note:
- this includes tokenizer-side caching/short-circuiting on top of fast APIs.
- API-only impact should still be clearly positive and measurable.

## PR title

`Add fast tokenizer-oriented Python bindings (wait_mask / has_riichi_discard / has_hupai)`

## PR description (English)

This PR adds several additive Python bindings to speed up high-frequency Mahjong state checks used by downstream tokenizers/training pipelines.

### Added bindings
- `wait_mask(hand, meld_count)`
- `has_riichi_discard(hand, meld_count)`
- `has_hupai(hand, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban)`
- `Shoupai.tingpai_mask()`
- `Shoupai.tingpai_list()`

### Motivation
Downstream tokenization runs call wait/riichi/win checks millions of times.  
These direct bindings avoid repeated Python-side object setup and provide a significantly faster path.

### Compatibility
All changes are additive and keep existing APIs intact.

### Validation
- Built and imported on macOS local environment.
- Downstream tokenizer integration passed unit tests and strict smoke tokenization on real Tenhou logs.

## Note for this repository

This repository installs from upstream merged commit and additionally applies a local patch
for batch APIs (`has_hupai_multi`, `evaluate_draw`) until next upstream PR is merged.
