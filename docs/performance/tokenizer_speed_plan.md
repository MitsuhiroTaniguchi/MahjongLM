# Tokenizer Speed Plan

更新日: 2026-03-12

## 現状

初期の単純実装に対して、現在の tokenizer はすでに大きく高速化されている。

主な改善:

1. `pymahjong` fast API の導入
- `wait_mask`
- `has_riichi_discard`
- `has_hupai`
- `has_hupai_multi`
- `evaluate_draw`

2. engine 側の削減
- `wait_mask` cache
- 形テン外の `ron` short-circuit
- `kakan` 候補探索の縮小
- self/reaction の batch API 利用

3. scraping 側の throughput 改善
- fetch は親プロセス直列のまま維持
- `raw -> json -> tokenized` を worker 並列化
- 404 marker 導入で再試行無駄を抑制

## 測定メモ

過去の 2023 サンプル測定では、概ね次の流れで改善した。

- old baseline: 約 `19 games/s`
- fast API 導入後: 約 `35 games/s`
- cache / short-circuit 導入後: 約 `60 games/s`
- batch API 導入後: 約 `63 games/s`

これは tokenizer 単体寄りの測定で、現行 scraping 全体の wall-clock とは別物である。

## 現在のボトルネック

tokenizer 本体:

- `_compute_reaction_options`
- `_compute_self_options`
- `pymahjong` 呼び出し回数そのもの
- Python 側イベントループの分岐コスト

scraping 全体:

- Tenhou 側 HTTP 応答待ち
- parent loop の fetch/drain/backpressure

観測上、`raw/json` が既に存在して downstream だけ回る状況では数百 `it/s` が出る。一方で実 fetch を含む run は単一 session・直列 fetch なので、速度支配は network 側になる。

## いま優先すべき最適化

### 1. 正しさ優先の profiling 継続

furiten / penuki / qianggang / haidi まわりは実牌譜 failure を通じて何度も修正が入った。速度変更でもまず実データ回帰を優先する。

必須確認:

- unit test
- tricky 実牌譜 replay
- 1000 戦レベルの downstream benchmark

### 2. tokenizer 呼び出し回数の削減

まだ有望なのは、個別 API の微小高速化より「何回呼ぶか」の削減である。

候補:

- self option 判定のさらなる統合
- reaction 判定の seat-by-seat 分岐削減
- event trace 生成時の余計な list copy 削減

### 3. scraping の運用改善

BAN 回避のため、fetch を複数 session 並列にはしていない。したがって、速度改善余地は主に downstream 側より運用面にある。

候補:

- downstream-only rerun mode の明示化
- worker 初期化の共有コスト削減
- year/filter 指定を使った小さい再実行フロー整備

## やらないこと

- Tenhou への多 session 並列 fetch
- semantic regression を伴う option/pass 省略
- benchmark 数値だけを見た unsafe なショートカット

## 受け入れ条件

速度改善を入れるときは、少なくとも次を満たすこと。

- token semantics が変わらない
- 既存 test が通る
- 実牌譜 regression が再発しない
- benchmark 条件と結果を記録する
