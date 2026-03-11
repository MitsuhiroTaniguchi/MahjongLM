from __future__ import annotations

from importlib.util import find_spec
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
    "TOKEN_VIEW_IMPERFECT_PREFIX",
    "imperfect_view_token",
    "iter_tokenized_games",
    "load_token_ids",
    "save_token_ids",
    "tokenize_game_views",
]

from .viewspec import (
    TOKEN_VIEW_COMPLETE,
    TOKEN_VIEW_IMPERFECT_PREFIX,
    VIEW_COMPLETE,
    VIEW_IMPERFECT,
    imperfect_view_token,
)
_HUGGINGFACE_EXPORTS = {
    "DEFAULT_HF_DATASETS_DIR",
    "DEFAULT_TOKENIZER_DIR",
    "MahjongDataCollatorForCausalLM",
    "MahjongGroupBatchSampler",
    "MahjongTokenizerFast",
    "build_hf_tokenizer",
    "load_hf_tokenizer",
    "save_hf_tokenizer_assets",
    "save_year_hf_dataset",
}
_HAS_HUGGINGFACE_DEPS = find_spec("datasets") is not None and find_spec("transformers") is not None
if _HAS_HUGGINGFACE_DEPS:
    __all__.extend(sorted(_HUGGINGFACE_EXPORTS))


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
    if name in _HUGGINGFACE_EXPORTS:
        if not _HAS_HUGGINGFACE_DEPS:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        huggingface = import_module(".huggingface", __name__)
        value = getattr(huggingface, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
