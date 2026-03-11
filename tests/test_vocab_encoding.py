from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tenhou_tokenizer import TenhouTokenizer, Vocabulary, load_token_ids, save_token_ids, tokenize_game_views
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


def test_vocabulary_encode_decode_roundtrip() -> None:
    vocab = Vocabulary.load()
    tokens = ["game_start", "view_imperfect_0", "rule_player_4", "round_start", "game_end"]

    ids = vocab.encode(tokens)

    assert vocab.decode(ids) == tokens


def test_vocabulary_rejects_unknown_token() -> None:
    vocab = Vocabulary.load()

    try:
        vocab.encode(["definitely_unknown_token"])
    except KeyError as exc:
        assert "definitely_unknown_token" in str(exc)
    else:
        raise AssertionError("expected unknown token lookup to fail")


def test_save_and_load_token_ids_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sample.ids.bin"
    ids = [0, 1, 2, 255, 256, 1024]
    vocab = Vocabulary.load()

    save_token_ids(path, ids, vocab_fingerprint=vocab.fingerprint)

    assert load_token_ids(path, expected_vocab_fingerprint=vocab.fingerprint) == ids


def test_load_token_ids_rejects_stale_vocab_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "sample.ids.bin"
    vocab = Vocabulary.load()

    save_token_ids(path, [1, 2, 3], vocab_fingerprint=vocab.fingerprint)

    try:
        load_token_ids(path, expected_vocab_fingerprint=b"\x00" * 16)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("expected vocab fingerprint mismatch to fail")


def test_vocab_covers_minimal_four_player_game_tokens() -> None:
    vocab = Vocabulary.load()
    tokens = TenhouTokenizer().tokenize_game(minimal_game([qipai_event(), pingju_event()]))

    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_vocab_covers_converted_sanma_game_tokens() -> None:
    vocab = Vocabulary.load()
    tokens = TenhouTokenizer().tokenize_game(_convert_sanma_sample())

    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_vocab_covers_all_multiview_tokens() -> None:
    vocab = Vocabulary.load()
    views = tokenize_game_views(minimal_game([qipai_event(), pingju_event()]))

    for view in views:
        assert vocab.decode(vocab.encode(view.tokens)) == view.tokens
