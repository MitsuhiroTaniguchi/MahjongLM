from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tenhou_tokenizer import (
    MahjongDataCollatorForCausalLM,
    TenhouTokenizer,
    Vocabulary,
    build_hf_tokenizer,
    load_hf_tokenizer,
    save_hf_tokenizer_assets,
    save_token_ids,
    save_year_hf_dataset,
)
from tests.fixtures.synthetic_logs import minimal_game, pingju_event, qipai_event


ROOT = Path(__file__).resolve().parents[1]
CONVERT = ROOT / "scripts" / "paifu_scraping" / "convert.pl"
SANMA_RAW = ROOT / "tests" / "fixtures" / "tenhou" / "2014091101gm-00b9-0000-5ca6b487.txt"


def _convert_sanma_sample() -> dict:
    proc = subprocess.run(
        ["perl", "-T", str(CONVERT), str(SANMA_RAW)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_hf_tokenizer_roundtrip_from_saved_assets(tmp_path: Path) -> None:
    save_hf_tokenizer_assets(tmp_path)

    tokenizer = load_hf_tokenizer(tmp_path)
    vocab = Vocabulary.load()
    tokens = ["game_start", "rule_player_3", "round_start", "game_end"]

    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.convert_tokens_to_ids(tokens) == vocab.encode(tokens)


def test_hf_tokenizer_build_matches_vocab_ids() -> None:
    tokenizer = build_hf_tokenizer()
    vocab = Vocabulary.load()

    tokens = ["game_start", "rule_player_4", "rule_length_hanchan", "game_end"]
    assert tokenizer.convert_tokens_to_ids(tokens) == vocab.encode(tokens)


def test_save_year_hf_dataset_writes_input_ids_dataset(tmp_path: Path) -> None:
    year = 2026
    tokenized_dir = tmp_path / "tokenized" / str(year)
    json_dir = tmp_path / "json" / str(year)
    tokenized_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)

    tokenizer = TenhouTokenizer()
    vocab = Vocabulary.load()
    game = minimal_game([qipai_event(), pingju_event()])
    tokens = tokenizer.tokenize_game(game)
    save_token_ids(tokenized_dir / "g0.ids.bin", vocab.encode(tokens), vocab_fingerprint=vocab.fingerprint)
    (json_dir / "g0.json").write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "hf" / str(year)
    save_year_hf_dataset(year=year, tokenized_dir=tokenized_dir, json_dir=json_dir, output_dir=output_dir)

    from datasets import load_from_disk

    dataset = load_from_disk(str(output_dir))
    row = dataset[0]
    assert row["game_id"] == "g0"
    assert row["year"] == year
    assert row["seat_count"] == 4
    assert row["length"] == len(tokens)
    assert row["input_ids"] == vocab.encode(tokens)


def test_collator_builds_attention_mask_and_labels() -> None:
    tokenizer = build_hf_tokenizer()
    collator = MahjongDataCollatorForCausalLM(tokenizer=tokenizer, pad_to_multiple_of=4, return_tensors="np")

    batch = collator(
        [
            {"input_ids": [4, 6, 10]},
            {"input_ids": [4, 5, 8, 10, 9]},
        ]
    )

    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ]
    assert batch["labels"].tolist()[0][:3] == [4, 6, 10]
    assert all(value == -100 for value in batch["labels"].tolist()[0][3:])


def test_hf_tokenizer_assets_cover_converted_sanma_game() -> None:
    tokenizer = build_hf_tokenizer()
    tokens = TenhouTokenizer().tokenize_game(_convert_sanma_sample())

    ids = tokenizer.convert_tokens_to_ids(tokens)

    assert all(token_id is not None for token_id in ids)
    assert tokenizer.convert_ids_to_tokens(ids[:8]) == tokens[:8]


def test_hf_tokenizer_import_does_not_require_pymahjong(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        """
import importlib.abc
import sys

class BlockPymahjong(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymahjong":
            raise ModuleNotFoundError("blocked for test")
        return None

sys.meta_path.insert(0, BlockPymahjong())

from tenhou_tokenizer import MahjongTokenizerFast

tok = MahjongTokenizerFast.from_pretrained("tokenizer")
print(type(tok).__name__)
print(tok.convert_tokens_to_ids(["game_start", "game_end"]))
""".strip(),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )

    assert "MahjongTokenizerFast" in proc.stdout
    assert "[4, 9]" in proc.stdout
