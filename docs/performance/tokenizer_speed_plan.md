# Tokenizer Speed Plan

Updated: 2026-03-04

## Goal

Reduce Tenhou tokenization wall-clock time while keeping token semantics unchanged.

## Current baseline and recent gain

Benchmark condition:
- machine: local dev machine
- input: `data2023.zip`
- sample size: first 3000 games
- command: in-repo benchmark script (same logic as tokenizer core loop)

Results:
- Before (old pymahjong API path): `19.28 games/s` (`3000 / 155.634s`)
- After (pymahjong fast APIs + engine integration): `35.81 games/s` (`3000 / 83.775s`)
- After phase-1 cache + shape-gate + furiten short-circuit: `60.13 games/s` (`3000 / 49.889s`)
- After phase-2 batch APIs (`evaluate_draw`, `has_hupai_multi`): `62.98 games/s` (`3000 / 47.636s`)
- Total gain vs baseline: `+226.7%`

## What changed (already done)

1. Added pymahjong fast APIs (upstream merged):
- `wait_mask(hand, meld_count)`
- `has_riichi_discard(hand, meld_count)`
- `has_hupai(hand, melds, win_tile, is_tsumo, is_menqian, is_riichi, zhuangfeng, lunban)`
- `Shoupai.tingpai_mask()` / `Shoupai.tingpai_list()`

2. Engine switched to those APIs:
- replaced Python-side repeated loops and object construction where possible
- reduced redundant list copies
- reduced expensive `kakan` scan from fixed 34-loop to open-pon keys only
- removed duplicate `parse_hand_counts` in `qipai`

3. Added tokenizer-side phase-1 optimization:
- per-player `wait_mask` cache with explicit invalidation on state-changing events
- ron checks short-circuit when offered tile is outside current wait mask
- removed temporary `p.concealed` swap in reaction path (`_can_win_with_counts`)

4. Setup made reproducible:
- `scripts/setup_pymahjong.sh` installs from upstream merged commit + local batch-API patch.

5. Added pymahjong-side phase-2 APIs (local patch):
- `evaluate_draw(...)` for combined tsumo/riichi-discard evaluation in one call
- `has_hupai_multi(...)` for batched hupai checks
- tokenizer engine switched self/reaction paths to use these APIs

## Remaining bottlenecks (cProfile, 1000 games)

Top cumulative contributors still include:
- `_compute_reaction_options`
- `_compute_self_options`
- `pymahjong.has_hupai`
- `pymahjong.wait_mask`
- `pymahjong.has_riichi_discard`
- Python event loop overhead in `_on_draw` / `_on_discard`

Interpretation:
- major C++ call overhead remains from high call counts, not single-call slowness.

## Next optimization phases

### Phase 1 (completed)

Completed with measured gain from `35.81 -> 60.13 games/s`.

### Phase 2 (partially completed)

Completed:
- `has_hupai_multi(...)` and `evaluate_draw(...)` introduced and integrated

Remaining:
- optional API to return `wait_mask` and riichi-discard possibility in one pass
- upstream PR/merge for phase-2 local patch

### Phase 3 (throughput scaling)

1. Multi-process sharding for CLI
- split by zip index range
- merge JSONL outputs after processing

Expected: near-linear wall-clock reduction by worker count.

## Safety constraints

- No semantic regression in decision tokens (`opt/take/pass`) is allowed.
- Every speed change requires:
  - existing unit tests pass
  - at least one known tricky log replay passes
  - benchmark comparison with same sample window.

## Acceptance targets

Short term target:
- single-process throughput >= `45 games/s` on current machine

Mid term target:
- single-process throughput >= `55 games/s`
- plus optional multi-process mode for full-year runs
