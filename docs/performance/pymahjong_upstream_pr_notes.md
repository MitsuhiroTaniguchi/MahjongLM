# pymahjong Upstream PR Notes

更新日: 2026-03-12

このノートは、この repository が `pymahjong` に何を期待しているか、その upstream 変更をどう追ってきたかをまとめたもの。

## この repository が使う API

現在 Tenhou tokenizer が依存している主な `pymahjong` API は次のとおり。

- `wait_mask`
- `has_riichi_discard`
- `has_hupai`
- `has_hupai_multi`
- `evaluate_draw`
- `compute_self_option_mask`
- `compute_reaction_option_masks`
- `compute_rob_kan_option_masks`
- `Shoupai`
- `Xiangting`

三麻では `penuki`、`qianggang`、`haidi/lingshang` 文脈も正しく流す必要がある。

## これまでの upstream 変更

### PR #4

fast tokenizer 向け API の最初の整理。

主なポイント:

- `wait_mask`
- `has_riichi_discard`
- `has_hupai`
- `Shoupai.tingpai_mask()`
- `Shoupai.tingpai_list()`

### PR #5 / #6

batch/context API の追加。

主なポイント:

- `has_hupai_multi`
- `evaluate_draw`
- `is_haidi`
- `is_lingshang`
- `is_qianggang`

Tokenizer 側では self/reaction 両方でこれを使う。

### PR #10

package/import/build まわりの整理。

確認した点:

- 拡張モジュール名が `pymahjong._core` になっても `pymahjong/__init__.py` で public import は維持される
- この repository 側の `import pymahjong as pm` はそのまま互換
- data file 探索は editable install と wheel install の両方に整合

この repository 側では、PR #10 相当の変更に対して tokenizer の relevant test を通し、merge blocker は見当たらないことを確認済み。

## この repository 側の要求

`pymahjong` に対して重要なのは速度だけではない。

要求は次の順序で考えるべき。

1. legality 判定が実牌譜に追随すること
2. 三麻・槍槓・北抜き・河底/海底などの文脈が正しく渡ること
3. Python 側から低コストで呼べること

特に tokenizer は「選ばれなかった合法手」まで出力するので、和了判定や furiten 判定が少しでもずれると silent corruption になりやすい。

## ローカル検証方針

upstream branch / commit を試すときは:

```bash
PYMAHJONG_REF=<branch-or-commit> ./scripts/setup_pymahjong.sh
```

最低限見るべきもの:

- `tests/test_engine_round_state.py`
- `tests/test_engine_decision_tokens.py`
- `tests/test_score_tokens.py`
- tricky 実牌譜の単体 tokenize

必要なら追加で:

- 1000 戦 benchmark
- scraping downstream rerun

## 注意

- `pymahjong` の API 変更は tokenizer 側テストとセットで扱う
- import/package 変更は optional dependency や editable install まで含めて確認する
- upstream 側で merge 済みでも、この repository での互換確認なしに追従しない
