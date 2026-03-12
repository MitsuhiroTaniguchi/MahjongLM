# MahjongLM

[English](README.md) | [日本語](README.ja.md)

天鳳 JSON 牌譜の tokenizer、multiview dataset builder、そして麻雀シーケンス学習用の小型 GPT-2 学習基盤をまとめた repository です。

## リポジトリ構成

- [src/tenhou_tokenizer/engine.py](src/tenhou_tokenizer/engine.py): イベント逐次再生ベースの tokenizer 本体
- [src/tenhou_tokenizer/views.py](src/tenhou_tokenizer/views.py): complete / imperfect multiview 変換
- [scripts/tokenize_tenhou.py](scripts/tokenize_tenhou.py): JSON zip -> JSONL token stream CLI
- [scripts/paifu_scraping/scraping.py](scripts/paifu_scraping/scraping.py): Tenhou fetch + raw/json/tokenized/HF dataset pipeline
- [src/gpt2/train.py](src/gpt2/train.py): 小型 GPT-2 学習 entrypoint
- [tokenizer/vocab.txt](tokenizer/vocab.txt): 語彙の source of truth

## Tokenizer 概要

この tokenizer は次を出力します。

- 各 decision point における選択行動と未選択合法手
  - `opt_self_*` / `take_self_*` / `pass_self_*`
  - `opt_react_*` / `take_react_*` / `pass_react_*`
- 和了詳細トークン
  - `yaku_*`, `han_*`, `fu_*`, `yakuman_*`
- multiview 出力
  - `view_complete` 1 本
  - `view_imperfect_{player}` を seat ごとに 1 本
  - `player` は `qijia` 基準

imperfect view では、viewer 以外の初期手牌を hidden player ごとに `hidden_haipai_{seat}` 1 トークンで隠します。これは hidden な `haipai_{seat}` block 全体の置き換えであり、`haipai_{seat}` の後ろに追加されるわけではありません。

和了詳細トークンは `take_self_*_tsumo` / `take_react_*_ron` の後、`score_delta_*` の前に出ます。

正規化ルール:

- `han_*` は `han_13` を上限に正規化
- `fu_*` は `25` と `20..140` の 10 刻みを使用
- `yakuman_*` は `damanguan` が立っている手だけに出す
- `yaku_dora`, `yaku_ura_dora`, `yaku_aka_dora` は成立翻数ぶん繰り返す

トークン仕様の詳細は [トークン設計.md](docs/references/トークン設計.md) を参照してください。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
./scripts/setup_pymahjong.sh
```

`setup_pymahjong.sh` はデフォルトで GitHub の `pymahjong` `main` を install します。

```bash
PYMAHJONG_REF=<branch-or-commit> ./scripts/setup_pymahjong.sh
```

## zip 1 本を tokenize

tokenizer CLI の default input は、現在は tracked な fixture zip です。

- [data2023_sample_v1.zip](tests/fixtures/tenhou/data2023_sample_v1.zip)

例:

```bash
PYTHONPATH=src .venv/bin/python scripts/tokenize_tenhou.py --max-games 20 --progress-every 0
```

主な option:

- `--zip-path`: 単一 input zip
- `--all-years --zip-glob`: 複数 zip をまとめて処理
- `--strict`: 1 局でも skip が出たら失敗
- `--workers`: 大きい入力向けの process 並列 tokenization

default:

- input zip: `tests/fixtures/tenhou/` 配下の tracked fixture sample zip
- output: `data/processed/tenhou/tokens_2023.jsonl.gz`

output path はローカル成果物として Git 管理外です。

## Scraping pipeline

全量データ生成は [scraping.py](scripts/paifu_scraping/scraping.py) が担当します。

挙動:

- 親プロセスが 1 本の `requests.Session` で Tenhou fetch
- `raw -> json -> tokenized` は worker process で並列変換
- 年単位の Hugging Face dataset を tokenized view から再構築
- `404` fetch は `.404` marker で記録し、再試行を避ける

full run:

```bash
PYTHONPATH=src .venv/bin/python scripts/paifu_scraping/scraping.py
```

## Hugging Face dataset

保存単位は 1 game ではなく 1 view 1 row です。

列:

- `game_id`
- `group_id`
- `year`
- `seat_count`
- `view_type`
- `viewer_seat`
- `length`
- `input_ids`

`group_id == game_id` で、正しい group は次を含みます。

- complete view 1 本
- imperfect view を各 seat 1 本

## GPT-2 学習

学習コードは [src/gpt2/](src/gpt2) 配下です。

現在の batching 仕様:

- sampler は `group_id` 単位で batch を組む
- 公開されている batch-size 制御は `max_tokens_per_batch`
- collator は `EOS` を付けて複数 segment を 1 row に pack する
- 同じ `group_id` は同じ packed row に入れない
- attention mask は 4D の block-diagonal causal mask で、segment 間 attention は切る

現在の default:

- context length: `8192`
- train token budget: `65536`
- eval token budget: `65536`

例:

```bash
PYTHONPATH=src .venv/bin/python -m gpt2.train \
  --dataset-dir data/huggingface_datasets/2023 \
  --output-dir runs/example \
  --train-steps 20 \
  --wandb-mode disabled
```

## テスト

fast test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q
```

補足:

- tokenizer test には `pymahjong` が必要
- GPT-2 学習 test には追加で `torch` が必要

## 参考

- [トークン設計.md](docs/references/トークン設計.md)
- [tenhou_paifu_notes_kobalab.md](docs/references/tenhou_paifu_notes_kobalab.md)
- [tokenizer_speed_plan.md](docs/performance/tokenizer_speed_plan.md)
- [pymahjong_upstream_pr_notes.md](docs/performance/pymahjong_upstream_pr_notes.md)

## ライセンス

この repository のソースコードは `MIT` です。

第三者コードの notice は [THIRD_PARTY.md](THIRD_PARTY.md) にまとめています。
Tenhou 牌譜や派生 dataset はコードライセンスの対象外です。
詳細は [TENHOU_DATA_NOTICE.md](TENHOU_DATA_NOTICE.md) を参照してください。
