from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import Dataset, DatasetDict

from gpt2.preference_data import (
    GeneratedGame,
    build_preference_pairs,
    read_generation_batches,
)
from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MahjongLM DPO preference data from batched same-seed generations. "
            "Rows follow TRL/Unsloth's prompt/chosen/rejected text format."
        )
    )
    parser.add_argument("--generations-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "huggingface_dpo_preferences")
    parser.add_argument("--repo-id", type=str, default="")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.bos_token != "<bos>" or tokenizer.eos_token != "<eos>":
        raise ValueError("Mahjong tokenizer must expose <bos>/<eos> for DPO text rows")

    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []
    batch_count = 0
    discarded_same_rank = 0
    for batch in read_generation_batches(args.generations_jsonl):
        batch_count += 1
        pairs = build_preference_pairs(batch, rng=rng)
        if not pairs:
            discarded_same_rank += 1
            continue
        records.extend(pair.as_record() for pair in pairs)
        if args.max_train_rows and len(records) >= args.max_train_rows:
            records = records[: args.max_train_rows]
            break

    if not records:
        raise ValueError("no preference rows were produced from the supplied generations")

    rng.shuffle(records)
    eval_size = max(1, int(len(records) * args.eval_ratio)) if args.eval_ratio > 0 and len(records) > 1 else 0
    eval_records = records[:eval_size]
    train_records = records[eval_size:]
    dataset_dict = DatasetDict({"train": Dataset.from_list(train_records)})
    if eval_records:
        dataset_dict["validation"] = Dataset.from_list(eval_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(args.output_dir))
    card = build_dataset_card(
        row_count=len(records),
        train_count=len(train_records),
        eval_count=len(eval_records),
        batch_count=batch_count,
        discarded_same_rank=discarded_same_rank,
    )
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps({"rows": len(records), "train": len(train_records), "validation": len(eval_records)}, indent=2))

    if args.repo_id and not args.skip_upload:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset_dict.push_to_hub(args.repo_id, private=args.private, token=token)
        # Dataset cards are not always picked up by push_to_hub from save_to_disk artifacts.
        try:
            from huggingface_hub import HfApi

            HfApi(token=token).upload_file(
                path_or_fileobj=str(args.output_dir / "README.md"),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        except Exception as exc:  # pragma: no cover - upload diagnostics only
            print(f"warning: failed to upload README.md separately: {exc}", file=sys.stderr)


def build_dataset_card(
    *,
    row_count: int,
    train_count: int,
    eval_count: int,
    batch_count: int,
    discarded_same_rank: int,
) -> str:
    return f"""---
language:
- ja
task_categories:
- text-generation
- reinforcement-learning
pretty_name: MahjongLM DPO Preferences
---

# MahjongLM DPO Preferences

This dataset contains preference pairs for fine-tuning MahjongLM as an imperfect-information player AI.

Each row is generated from multiple same-seed `view_omniscient` rollouts. If all rollouts produce the same final ranks, the batch is discarded. Otherwise, for each seat, one better-ranked rollout and one worse-ranked rollout are converted to `view_imperfect_*` and stored as a DPO pair.

## Columns

- `prompt`: The shared prompt. For full-game preference DPO this is `<bos>`.
- `chosen`: The better-ranked `view_imperfect_*` token sequence after `<bos>`, ending with `<eos>`.
- `rejected`: The worse-ranked `view_imperfect_*` token sequence after `<bos>`, ending with `<eos>`.
- `viewer_seat`: The imperfect-information player seat used for the pair.
- `chosen_rank`, `rejected_rank`: Final placement labels where smaller is better.
- `seed_id`, `rule_key`, `chosen_generation_index`, `rejected_generation_index`: Provenance fields.

## Build Summary

- preference rows: {row_count}
- train rows: {train_count}
- validation rows: {eval_count}
- generation batches scanned: {batch_count}
- same-rank batches discarded: {discarded_same_rank}
"""


if __name__ == "__main__":
    main()
