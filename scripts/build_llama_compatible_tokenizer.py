#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

try:
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
except ImportError as exc:  # pragma: no cover - dependency is env-specific
    raise SystemExit(
        "sentencepiece is required to build tokenizer.model. "
        "Install it in the current Python environment."
    ) from exc


SPECIAL_TOKEN_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


def load_vocab(tokenizer_dir: Path) -> list[str]:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    with tokenizer_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    model = payload.get("model", {})
    pre_tokenizer = payload.get("pre_tokenizer") or {}
    if model.get("type") != "WordLevel" or pre_tokenizer.get("type") != "WhitespaceSplit":
        raise ValueError("Only WordLevel + WhitespaceSplit tokenizers are supported")

    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json is missing model.vocab")

    ordered = [None] * len(vocab)
    for token, token_id in vocab.items():
        if not isinstance(token_id, int) or token_id < 0 or token_id >= len(ordered):
            raise ValueError(f"invalid token id for {token!r}: {token_id!r}")
        if ordered[token_id] is not None:
            raise ValueError(f"duplicate token id detected: {token_id}")
        ordered[token_id] = token

    if any(token is None for token in ordered):
        raise ValueError("token ids are not contiguous")

    for token, token_id in SPECIAL_TOKEN_IDS.items():
        if ordered[token_id] != token:
            raise ValueError(f"expected {token!r} at id {token_id}, found {ordered[token_id]!r}")

    return ordered


def build_model(vocab: list[str]) -> sp_pb2.ModelProto:
    helper_model = build_helper_model()
    model = sp_pb2.ModelProto()
    model.trainer_spec.model_type = sp_pb2.TrainerSpec.UNIGRAM
    model.trainer_spec.vocab_size = len(vocab)
    model.trainer_spec.unk_id = SPECIAL_TOKEN_IDS["<unk>"]
    model.trainer_spec.bos_id = SPECIAL_TOKEN_IDS["<bos>"]
    model.trainer_spec.eos_id = SPECIAL_TOKEN_IDS["<eos>"]
    model.trainer_spec.pad_id = SPECIAL_TOKEN_IDS["<pad>"]
    model.trainer_spec.unk_piece = "<unk>"
    model.trainer_spec.bos_piece = "\u2581<bos>"
    model.trainer_spec.eos_piece = "\u2581<eos>"
    model.trainer_spec.pad_piece = "<pad>"
    model.normalizer_spec.CopyFrom(helper_model.normalizer_spec)

    for token_id, token in enumerate(vocab):
        piece = model.pieces.add()
        piece.score = 0.0
        if token_id == SPECIAL_TOKEN_IDS["<pad>"]:
            piece.piece = "<pad>"
            piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
        elif token_id == SPECIAL_TOKEN_IDS["<unk>"]:
            piece.piece = "<unk>"
            piece.type = sp_pb2.ModelProto.SentencePiece.UNKNOWN
        elif token_id == SPECIAL_TOKEN_IDS["<bos>"]:
            piece.piece = "\u2581<bos>"
            piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
        elif token_id == SPECIAL_TOKEN_IDS["<eos>"]:
            piece.piece = "\u2581<eos>"
            piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
        else:
            piece.piece = "\u2581" + token
            piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL

    return model


def build_helper_model() -> sp_pb2.ModelProto:
    # SentencePiece stores whitespace handling in a compiled charsmap. We train a
    # tiny helper model just to reuse that normalizer while keeping our own ids.
    with tempfile.TemporaryDirectory(prefix="mahjonglm_llama_tok_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        corpus = tmpdir / "corpus.txt"
        corpus.write_text(
            "\n".join(
                [
                    "<bos> rule_player_4 rule_length_hanchan view_complete game_start round_start",
                    "<bos>\nrule_player_4\trule_length_hanchan   view_complete game_start",
                ]
            ),
            encoding="utf-8",
        )
        prefix = tmpdir / "helper"
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(prefix),
            model_type="unigram",
            vocab_size=32,
            unk_id=SPECIAL_TOKEN_IDS["<unk>"],
            bos_id=SPECIAL_TOKEN_IDS["<bos>"],
            eos_id=SPECIAL_TOKEN_IDS["<eos>"],
            pad_id=SPECIAL_TOKEN_IDS["<pad>"],
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            pad_piece="<pad>",
            hard_vocab_limit=False,
            remove_extra_whitespaces=True,
        )

        helper = sp_pb2.ModelProto()
        helper.ParseFromString((prefix.with_suffix(".model")).read_bytes())
        return helper


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a llama.cpp-compatible tokenizer.model")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("tokenizer"),
        help="Directory containing tokenizer.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output tokenizer.model path (defaults to <tokenizer-dir>/tokenizer.model)",
    )
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir.resolve()
    output = args.output.resolve() if args.output is not None else tokenizer_dir / "tokenizer.model"

    vocab = load_vocab(tokenizer_dir)
    model = build_model(vocab)
    output.write_bytes(model.SerializeToString())
    print(f"wrote {output}")
    print(f"vocab_size={len(vocab)}")


if __name__ == "__main__":
    main()
