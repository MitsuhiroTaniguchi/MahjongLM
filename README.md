# Mahjong LLM

This repository now includes a streaming Tenhou tokenizer that adds explicit option/pass tokens for latent decisions.

## Tokenizer

- Engine: `src/tenhou_tokenizer/engine.py`
- CLI: `scripts/tokenize_tenhou.py`

### What is added

At each decision point, the tokenizer emits:

- `opt_self_*` / `pass_self_*` for draw-time choices (riichi, ankan, kakan, tsumo)
- `opt_react_*` / `pass_react_*` for discard reactions (chi, pon, minkan, ron)

This addresses the key dataset issue where logs only contain chosen actions and omit unchosen-but-legal options.

### Run (small test)

```bash
./scripts/setup_pymahjong.sh
source .venv/bin/activate
python scripts/tokenize_tenhou.py --max-games 200 --progress-every 50
```

`setup_pymahjong.sh` installs upstream `pymahjong` and applies a local additive patch for
extra speed APIs (`has_hupai_multi`, `evaluate_draw`) before build/install.
That patch corresponds to upstream PR #5.
Upstream PR notes: `docs/performance/pymahjong_upstream_pr_notes.md`.

Default input is `data/raw/tenhou/data2023.zip` and default output is `data/processed/tenhou/tokens_2023.jsonl.gz`.

For all years in `data/raw/tenhou/`:

```bash
python scripts/tokenize_tenhou.py --all-years --zip-glob "data/raw/tenhou/data*.zip" --output data/processed/tenhou/tokens_all_years.jsonl.gz
```

Fail fast when any game is skipped:

```bash
python scripts/tokenize_tenhou.py --zip-path data/raw/tenhou/data2023.zip --strict
```

## Tests

```bash
source .venv/bin/activate
pip install -U pytest
pytest -m "not slow" -q
```

Optional dataset smoke test:

```bash
pytest -m slow -q
```

## Important note

This tokenizer is `pymahjong`-first: shanten and hupai checks are delegated to its C++ implementation.

## References

- [kobalab-based Tenhou paifu notes](docs/references/tenhou_paifu_notes_kobalab.md)
