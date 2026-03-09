from __future__ import annotations

from importlib import import_module
from typing import Any

from .vocab import Vocabulary, load_token_ids, save_token_ids

__all__ = [
    "TenhouTokenizer",
    "TokenizeError",
    "Vocabulary",
    "iter_tokenized_games",
    "load_token_ids",
    "save_token_ids",
]

try:
    from .huggingface import (
        DEFAULT_HF_DATASETS_DIR,
        DEFAULT_TOKENIZER_DIR,
        MahjongDataCollatorForCausalLM,
        MahjongTokenizerFast,
        build_hf_tokenizer,
        load_hf_tokenizer,
        save_hf_tokenizer_assets,
        save_year_hf_dataset,
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(
        [
            "DEFAULT_HF_DATASETS_DIR",
            "DEFAULT_TOKENIZER_DIR",
            "MahjongDataCollatorForCausalLM",
            "MahjongTokenizerFast",
            "build_hf_tokenizer",
            "load_hf_tokenizer",
            "save_hf_tokenizer_assets",
            "save_year_hf_dataset",
        ]
    )


def __getattr__(name: str) -> Any:
    if name in {"TenhouTokenizer", "TokenizeError", "iter_tokenized_games"}:
        engine = import_module(".engine", __name__)
        value = getattr(engine, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
