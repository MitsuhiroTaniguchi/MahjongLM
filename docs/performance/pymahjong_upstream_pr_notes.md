# pymahjong Upstream PR Notes (Fast APIs)

Updated: 2026-03-05

## Goal

Contribute tokenizer-oriented fast APIs to `pymahjong` upstream and consume them directly from the repository.

## Status

- PR: https://github.com/MitsuhiroTaniguchi/pymahjong/pull/4
- State: merged on 2026-03-04
- Merge commit: `0b80bafa99c5b4b8bc1ade256c5a86b679238576`
- Batch API PR: https://github.com/MitsuhiroTaniguchi/pymahjong/pull/5
- Batch/context API PR: https://github.com/MitsuhiroTaniguchi/pymahjong/pull/6
- Batch/context API state: merged on 2026-03-05 (PR #6)
- Batch/context API merge commit: `d92973ad6a34bcd564c8a9d8da4c50627da43485`
- Setup/toolchain fix in PR #6: stop forcing `gcc/g++` in `setup.py` and honor environment/default toolchain.

## Proposed PR scope

Target file in upstream `pymahjong`:
- `src/bindings.cpp`

Changes:
- `wait_mask(hand, meld_count) -> int`
- `has_riichi_discard(hand, meld_count) -> bool`
- `has_hupai(hand, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban, is_haidi, is_lingshang, is_qianggang) -> bool`
- `has_hupai_multi(cases_with_context) -> List[bool]`
- `evaluate_draw(..., check_riichi_discard, is_haidi, is_lingshang) -> Tuple[bool, bool]`
- `Shoupai.tingpai_mask() -> int`
- `Shoupai.tingpai_list() -> List[int]`

Compatibility:
- breaking API update accepted for tokenizer performance/clarity

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

`Integrate context flags into fast tokenizer APIs (breaking update)`

## PR description (English)

This PR updates the fast Python bindings used by downstream tokenizers/training pipelines, integrating context flags directly into existing APIs.

### Added bindings
- `wait_mask(hand, meld_count)`
- `has_riichi_discard(hand, meld_count)`
- `has_hupai(hand, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban, is_haidi, is_lingshang, is_qianggang)`
- `has_hupai_multi(cases_with_context)`
- `evaluate_draw(..., check_riichi_discard, is_haidi, is_lingshang)`
- `Shoupai.tingpai_mask()`
- `Shoupai.tingpai_list()`

### Motivation
Downstream tokenization runs call wait/riichi/win checks millions of times.  
These direct bindings avoid repeated Python-side object setup and provide a significantly faster path.

### Compatibility
This PR intentionally updates signatures of existing fast APIs.

### Validation
- Built and imported on macOS local environment.
- Downstream tokenizer integration passed unit tests and strict smoke tokenization on real Tenhou logs.

## Note for this repository

This repository installs `pymahjong` directly from GitHub (`main` by default).
To validate a PR branch before merge, use `PYMAHJONG_REF=<branch-or-commit>`.
