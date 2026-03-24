from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedTokenizerFast


def load_hf_tokenizer(tokenizer_dir: str | Path) -> PreTrainedTokenizerFast:
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    if tokenizer_file.exists():
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
