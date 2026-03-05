# MahjongLM

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

`setup_pymahjong.sh` installs `pymahjong` directly from GitHub.
Default ref is `main`, and you can override with:

```bash
PYMAHJONG_REF=<branch-or-commit> ./scripts/setup_pymahjong.sh
```

Upstream PR notes: `docs/performance/pymahjong_upstream_pr_notes.md`.

Default input is `data/raw/tenhou/data2023.zip` and default output is `data/processed/tenhou/tokens_2023.jsonl.gz`.
Defaults are resolved from the repository root.
User-provided relative paths for `--zip-path`, `--zip-glob`, and `--output` are resolved from the current shell working directory.

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

`pymahjong` is required for tokenizer and CLI tests.

Optional dataset smoke test:

```bash
pytest -m slow -q
```

## Important note

This tokenizer is `pymahjong`-first: shanten and hupai checks are delegated to its C++ implementation.

Generated token outputs under `data/processed/` are local artifacts and are not tracked in Git.

## References

- [kobalab-based Tenhou paifu notes](docs/references/tenhou_paifu_notes_kobalab.md)
