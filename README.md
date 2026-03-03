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
python3 -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/MitsuhiroTaniguchi/pymahjong.git"
python scripts/tokenize_tenhou.py --max-games 200 --progress-every 50
```

Default input is `data/raw/tenhou/data2023.zip` and default output is `data/processed/tenhou/tokens_2023.jsonl.gz`.

## Important note

This tokenizer is `pymahjong`-first: shanten and hupai checks are delegated to its C++ implementation.

## References

- [kobalab-based Tenhou paifu notes](docs/references/tenhou_paifu_notes_kobalab.md)
