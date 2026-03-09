from __future__ import annotations

from importlib import import_module
from typing import Any

from .vocab import Vocabulary, load_token_ids, save_token_ids

__all__ = [
    "EventTrace",
    "TenhouTokenizer",
    "TokenizeError",
    "TokenizedGameView",
    "Vocabulary",
    "VIEW_COMPLETE",
    "VIEW_IMPERFECT",
    "TOKEN_VIEW_COMPLETE",
    "TOKEN_VIEW_IMPERFECT",
    "iter_tokenized_games",
    "load_token_ids",
    "save_token_ids",
    "tokenize_game_views",
]

from .viewspec import TOKEN_VIEW_COMPLETE, TOKEN_VIEW_IMPERFECT, VIEW_COMPLETE, VIEW_IMPERFECT

try:
    from .huggingface import (
        DEFAULT_HF_DATASETS_DIR,
        DEFAULT_TOKENIZER_DIR,
        MahjongDataCollatorForCausalLM,
        MahjongGroupBatchSampler,
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
            "MahjongGroupBatchSampler",
            "MahjongTokenizerFast",
            "build_hf_tokenizer",
            "load_hf_tokenizer",
            "save_hf_tokenizer_assets",
            "save_year_hf_dataset",
        ]
    )


def __getattr__(name: str) -> Any:
    if name in {"TenhouTokenizer", "TokenizeError", "iter_tokenized_games", "EventTrace"}:
        engine = import_module(".engine", __name__)
        value = getattr(engine, name)
        globals()[name] = value
        return value
    if name in {"TokenizedGameView", "tokenize_game_views"}:
        views = import_module(".views", __name__)
        value = getattr(views, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
