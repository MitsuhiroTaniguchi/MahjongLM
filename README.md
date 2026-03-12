# MahjongLM

[English](README.md) | [日本語](README.ja.md)

Tenhou JSON log tokenizer, multiview dataset builder, and small GPT-2 training stack for Mahjong sequence modeling.

## Repository layout

- [src/tenhou_tokenizer/engine.py](src/tenhou_tokenizer/engine.py): core event-by-event tokenizer
- [src/tenhou_tokenizer/views.py](src/tenhou_tokenizer/views.py): complete/imperfect multiview conversion
- [scripts/tokenize_tenhou.py](scripts/tokenize_tenhou.py): JSON zip -> JSONL token stream CLI
- [scripts/paifu_scraping/scraping.py](scripts/paifu_scraping/scraping.py): Tenhou fetch + raw/json/tokenized/HF dataset pipeline
- [src/gpt2/train.py](src/gpt2/train.py): tiny GPT-2 training entrypoint
- [tokenizer/vocab.txt](tokenizer/vocab.txt): vocabulary source of truth

## Tokenizer summary

The tokenizer emits:

- chosen actions and latent legal alternatives at each decision point
  - `opt_self_*` / `take_self_*` / `pass_self_*`
  - `opt_react_*` / `take_react_*` / `pass_react_*`
- winning-hand detail tokens
  - `yaku_*`, `han_*`, `fu_*`, `yakuman_*`
- multiview outputs
  - one `view_complete`
  - one `view_imperfect_{player}` per seat, where `player` is `qijia`-relative

Imperfect views hide non-viewer initial hands with a single `hidden_haipai_{seat}` token per hidden player. That token replaces the whole hidden `haipai_{seat}` block; it is not emitted after `haipai_{seat}`.

Winning-hand detail tokens appear after `take_self_*_tsumo` / `take_react_*_ron` and before `score_delta_*`.

Normalization rules:

- `han_*` is capped at `han_13`
- `fu_*` is emitted for `25` and `20..140` in steps of `10`
- `yakuman_*` is emitted only for hands where `damanguan` is present
- `yaku_dora`, `yaku_ura_dora`, `yaku_aka_dora` repeat by realized han count

See [トークン設計.md](docs/references/トークン設計.md) for the token-level specification.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
./scripts/setup_pymahjong.sh
```

`setup_pymahjong.sh` installs `pymahjong` from GitHub `main` by default.

```bash
PYMAHJONG_REF=<branch-or-commit> ./scripts/setup_pymahjong.sh
```

## Tokenize one zip

The tokenizer CLI now defaults to the tracked fixture zip:

- [data2023_sample_v1.zip](tests/fixtures/tenhou/data2023_sample_v1.zip)

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/tokenize_tenhou.py --max-games 20 --progress-every 0
```

Useful options:

- `--zip-path`: one input zip
- `--all-years --zip-glob`: process multiple zips
- `--strict`: fail if any game is skipped
- `--workers`: process-level parallel tokenization for larger inputs

Defaults:

- input zip: tracked fixture sample zip under `tests/fixtures/tenhou/`
- output: `data/processed/tenhou/tokens_2023.jsonl.gz`

The output path is intentionally local and ignored by Git.

## Scraping pipeline

Full data generation is handled by [scraping.py](scripts/paifu_scraping/scraping.py).

Behavior:

- parent process performs Tenhou fetches with one `requests.Session`
- downstream `raw -> json -> tokenized` conversion runs in worker processes
- per-year Hugging Face datasets are rebuilt from tokenized views
- `404` fetches are memoized with `.404` marker files to avoid repeated requests

The default full run is:

```bash
PYTHONPATH=src .venv/bin/python scripts/paifu_scraping/scraping.py
```

## Hugging Face datasets

Each saved row is one view, not one full game.

Columns:

- `game_id`
- `group_id`
- `year`
- `seat_count`
- `view_type`
- `viewer_seat`
- `length`
- `input_ids`

`group_id == game_id`. A valid group contains:

- one complete view
- one imperfect view per seat

## GPT-2 training

Training code lives under [src/gpt2/](src/gpt2).

Current batching semantics:

- sampler batches by `group_id`
- public batch-size control is `max_tokens_per_batch`
- collator appends `EOS`, packs multiple segments into one row, and keeps the same `group_id` out of the same packed row
- packed attention masks are 4D block-diagonal causal masks, so segments do not attend across boundaries

Current defaults:

- context length: `8192`
- train token budget: `65536`
- eval token budget: `65536`

Example:

```bash
PYTHONPATH=src .venv/bin/python -m gpt2.train \
  --dataset-dir data/huggingface_datasets/2023 \
  --output-dir runs/example \
  --train-steps 20 \
  --wandb-mode disabled
```

## Tests

Fast tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q
```

Notes:

- tokenizer tests require `pymahjong`
- training GPT-2 requires `torch`

## References

- [トークン設計.md](docs/references/トークン設計.md)
- [tenhou_paifu_notes_kobalab.md](docs/references/tenhou_paifu_notes_kobalab.md)
- [tokenizer_speed_plan.md](docs/performance/tokenizer_speed_plan.md)
- [pymahjong_upstream_pr_notes.md](docs/performance/pymahjong_upstream_pr_notes.md)

## License

This repository's source code is licensed under `MIT`.

Third-party code notices are collected in [THIRD_PARTY.md](THIRD_PARTY.md).
Tenhou logs and derived datasets are not covered by the code license; see
[TENHOU_DATA_NOTICE.md](TENHOU_DATA_NOTICE.md).
