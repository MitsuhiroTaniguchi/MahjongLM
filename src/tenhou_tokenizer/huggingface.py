from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import PreTrainedTokenizerFast

from .viewspec import parse_view_artifact_name
from .vocab import DEFAULT_VOCAB_PATH, Vocabulary, load_token_ids


DEFAULT_TOKENIZER_DIR = DEFAULT_VOCAB_PATH.parent
DEFAULT_HF_DATASETS_DIR = DEFAULT_VOCAB_PATH.parents[1] / "data" / "huggingface_datasets"


class MahjongTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = {"tokenizer_file": "tokenizer.json", "vocab_file": "vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *init_inputs, **kwargs):
        tokenizer_dir = Path(pretrained_model_name_or_path)
        tokenizer_file = tokenizer_dir / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"tokenizer.json not found under {tokenizer_dir}")

        special_tokens_path = tokenizer_dir / "special_tokens_map.json"
        special_tokens = (
            json.loads(special_tokens_path.read_text(encoding="utf-8"))
            if special_tokens_path.is_file()
            else {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        )
        return cls(
            tokenizer_file=str(tokenizer_file),
            clean_up_tokenization_spaces=False,
            model_max_length=int(1e30),
            **special_tokens,
            **kwargs,
        )


def build_hf_tokenizer(vocab: Vocabulary | None = None) -> MahjongTokenizerFast:
    vocabulary = vocab if vocab is not None else Vocabulary.load()
    backend = Tokenizer(WordLevel(vocabulary.token_to_id, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return MahjongTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_length=int(1e30),
        clean_up_tokenization_spaces=False,
    )


def save_hf_tokenizer_assets(tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    build_hf_tokenizer().save_pretrained(str(tokenizer_dir))
    (tokenizer_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "tokenizer_class": "MahjongTokenizerFast",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "tokenizer_file": "tokenizer.json",
                "model_max_length": int(1e30),
                "clean_up_tokenization_spaces": False,
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_hf_tokenizer(tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR) -> MahjongTokenizerFast:
    return MahjongTokenizerFast.from_pretrained(str(tokenizer_dir))


def _iter_year_rows(year: int, tokenized_dir: Path, json_dir: Path, vocab_fingerprint: bytes) -> Iterable[dict]:
    for tokenized_path in sorted(tokenized_dir.glob("*.ids.bin")):
        try:
            view_spec = parse_view_artifact_name(tokenized_path.name)
        except ValueError:
            # Ignore legacy single-view caches such as "{game_id}.ids.bin".
            # The scraper regenerates multiview artifacts alongside them, and
            # dataset export should remain forward-compatible without manual cleanup.
            continue
        game_id = view_spec.game_id
        json_path = json_dir / f"{game_id}.json"
        game = json.loads(json_path.read_text(encoding="utf-8"))
        input_ids = load_token_ids(tokenized_path, expected_vocab_fingerprint=vocab_fingerprint)
        seat_count = 0
        player = game.get("player")
        if isinstance(player, list):
            seat_count = len(player)
        elif isinstance(game.get("defen"), list):
            seat_count = len(game["defen"])
        elif game.get("log") and isinstance(game["log"], list):
            first_round = game["log"][0]
            if isinstance(first_round, list) and first_round:
                first_event = first_round[0]
                if isinstance(first_event, dict):
                    qipai = first_event.get("qipai")
                    if isinstance(qipai, dict):
                        shoupai = qipai.get("shoupai")
                        defen = qipai.get("defen")
                        if isinstance(shoupai, list):
                            seat_count = len(shoupai)
                        elif isinstance(defen, list):
                            seat_count = len(defen)
        if seat_count not in {3, 4}:
            raise ValueError(f"unable to infer seat_count for {game_id}")
        yield {
            "game_id": game_id,
            "group_id": game_id,
            "year": year,
            "seat_count": seat_count,
            "view_type": view_spec.view_type,
            "viewer_seat": -1 if view_spec.viewer_seat is None else view_spec.viewer_seat,
            "length": len(input_ids),
            "input_ids": input_ids,
        }


def save_year_hf_dataset(
    year: int,
    tokenized_dir: Path,
    json_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    vocab = Vocabulary.load()
    token_dtype = "uint16" if len(vocab.tokens) < 2**16 else "uint32"
    features = Features(
        {
            "game_id": Value("string"),
            "group_id": Value("string"),
            "year": Value("int32"),
            "seat_count": Value("int8"),
            "view_type": Value("string"),
            "viewer_seat": Value("int8"),
            "length": Value("int32"),
            "input_ids": Sequence(Value(token_dtype)),
        }
    )
    dataset = Dataset.from_generator(
        _iter_year_rows,
        gen_kwargs={
            "year": year,
            "tokenized_dir": tokenized_dir,
            "json_dir": json_dir,
            "vocab_fingerprint": vocab.fingerprint,
        },
        features=features,
    )
    dataset.save_to_disk(str(output_dir), max_shard_size="512MB")
    return output_dir


@dataclass
class MahjongDataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = "np"

    def __call__(self, features: list[dict]) -> dict[str, np.ndarray]:
        if not features:
            raise ValueError("features must not be empty")
        group_ids = [feature.get("group_id") for feature in features if "group_id" in feature]
        if group_ids and len(set(group_ids)) != 1:
            raise ValueError("all features in a batch must share the same group_id")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            input_ids = list(feature["input_ids"])
            pad_length = max_length - len(input_ids)
            batch_input_ids.append(input_ids + [pad_token_id] * pad_length)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_length)
            batch_labels.append(input_ids + [self.label_pad_token_id] * pad_length)

        arrays: dict[str, np.ndarray | list[str]] = {
            "input_ids": np.asarray(batch_input_ids, dtype=np.int64),
            "attention_mask": np.asarray(batch_attention_mask, dtype=np.int64),
            "labels": np.asarray(batch_labels, dtype=np.int64),
        }
        if "group_id" in features[0]:
            arrays["group_id"] = [feature["group_id"] for feature in features]
        if "view_type" in features[0]:
            arrays["view_type"] = [feature["view_type"] for feature in features]
        if "viewer_seat" in features[0]:
            arrays["viewer_seat"] = np.asarray([feature["viewer_seat"] for feature in features], dtype=np.int64)
        if self.return_tensors == "np":
            return arrays
        if self.return_tensors == "pt":
            import torch

            out = {}
            for key, value in arrays.items():
                if isinstance(value, np.ndarray):
                    out[key] = torch.from_numpy(value)
                else:
                    out[key] = value
            return out
        raise ValueError(f"unsupported return_tensors: {self.return_tensors}")


@dataclass
class MahjongGroupBatchSampler:
    group_ids: Sequence[str]
    groups_per_batch: int = 1
    shuffle: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be positive")

    def __iter__(self):
        grouped_indices: dict[str, list[int]] = {}
        group_order: list[str] = []
        for idx, group_id in enumerate(self.group_ids):
            if group_id not in grouped_indices:
                grouped_indices[group_id] = []
                group_order.append(group_id)
            grouped_indices[group_id].append(idx)

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(group_order)

        for batch_start in range(0, len(group_order), self.groups_per_batch):
            batch_groups = group_order[batch_start : batch_start + self.groups_per_batch]
            batch_indices: list[int] = []
            for group_id in batch_groups:
                batch_indices.extend(grouped_indices[group_id])
            yield batch_indices

    def __len__(self) -> int:
        unique_groups = len(dict.fromkeys(self.group_ids))
        return math.ceil(unique_groups / self.groups_per_batch)
