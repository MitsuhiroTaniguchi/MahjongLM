from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

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


def _iter_year_rows(year: int, tokenized_dir: Path, json_dir: Path) -> Iterable[dict]:
    for tokenized_path in sorted(tokenized_dir.glob("*.ids.bin")):
        game_id = tokenized_path.name.removesuffix(".ids.bin")
        json_path = json_dir / f"{game_id}.json"
        game = json.loads(json_path.read_text(encoding="utf-8"))
        input_ids = load_token_ids(tokenized_path)
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
            "year": year,
            "seat_count": seat_count,
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
    token_dtype = "uint16" if len(Vocabulary.load().tokens) < 2**16 else "uint32"
    features = Features(
        {
            "game_id": Value("string"),
            "year": Value("int32"),
            "seat_count": Value("int8"),
            "length": Value("int32"),
            "input_ids": Sequence(Value(token_dtype)),
        }
    )
    dataset = Dataset.from_generator(
        _iter_year_rows,
        gen_kwargs={"year": year, "tokenized_dir": tokenized_dir, "json_dir": json_dir},
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

        arrays = {
            "input_ids": np.asarray(batch_input_ids, dtype=np.int64),
            "attention_mask": np.asarray(batch_attention_mask, dtype=np.int64),
            "labels": np.asarray(batch_labels, dtype=np.int64),
        }
        if self.return_tensors == "np":
            return arrays
        if self.return_tensors == "pt":
            import torch

            return {key: torch.from_numpy(value) for key, value in arrays.items()}
        raise ValueError(f"unsupported return_tensors: {self.return_tensors}")
