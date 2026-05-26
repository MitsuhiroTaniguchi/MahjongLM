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

from datasets import DatasetDict, load_dataset

from gpt2.preference_data import read_generation_batches
from gpt2.rl_data import build_group_rl_records
from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build same-seed group-advantage RL data from MahjongLM rollout JSONL."
    )
    parser.add_argument("--generations-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "mahjonglm_rl_rollouts")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--repo-id", type=str, default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--placement-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = args.output_dir / "_jsonl_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = staging_dir / "train.jsonl"
    eval_jsonl = staging_dir / "validation.jsonl"
    row_count = 0
    train_count = 0
    eval_count = 0
    batch_count = 0
    discarded_no_valid_group = 0
    discarded_no_decisions = 0
    with train_jsonl.open("w", encoding="utf-8") as train_sink, eval_jsonl.open("w", encoding="utf-8") as eval_sink:
        for batch in read_generation_batches(args.generations_jsonl, tokenizer=tokenizer):
            batch_count += 1
            records = build_group_rl_records(
                batch,
                placement_weight=args.placement_weight,
                score_weight=args.score_weight,
            )
            if not records:
                discarded_no_valid_group += 1
                continue
            for record in records:
                row = record.as_tokenized_record(tokenizer)
                if not sum(row["loss_mask"]):
                    discarded_no_decisions += 1
                    continue
                sink = eval_sink if rng.random() < args.eval_ratio else train_sink
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
                if sink is eval_sink:
                    eval_count += 1
                else:
                    train_count += 1
                if args.max_train_rows and row_count >= args.max_train_rows:
                    break
            if args.max_train_rows and row_count >= args.max_train_rows:
                break
    if row_count == 0:
        raise ValueError("no RL rows were produced from the supplied generations")

    data_files = {"train": str(train_jsonl)}
    if eval_count:
        data_files["validation"] = str(eval_jsonl)
    dataset = DatasetDict(load_dataset("json", data_files=data_files))

    dataset.save_to_disk(str(args.output_dir))
    card = build_dataset_card(
        row_count=row_count,
        train_count=train_count,
        eval_count=eval_count,
        batch_count=batch_count,
        discarded_no_valid_group=discarded_no_valid_group,
        discarded_no_decisions=discarded_no_decisions,
        placement_weight=args.placement_weight,
        score_weight=args.score_weight,
    )
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": row_count,
                "train": train_count,
                "validation": eval_count,
                "discarded_no_valid_group": discarded_no_valid_group,
                "discarded_no_decisions": discarded_no_decisions,
            },
            indent=2,
        )
    )

    if args.repo_id and not args.skip_upload:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset.push_to_hub(args.repo_id, private=args.private, token=token)


def build_dataset_card(
    *,
    row_count: int,
    train_count: int,
    eval_count: int,
    batch_count: int,
    discarded_no_valid_group: int,
    discarded_no_decisions: int,
    placement_weight: float,
    score_weight: float,
) -> str:
    return f"""---
language:
- ja
task_categories:
- text-generation
- reinforcement-learning
pretty_name: MahjongLM Group-Advantage RL Rollouts
---

# MahjongLM Group-Advantage RL Rollouts

This dataset stores same-seed multi-rollout policy-gradient training rows. Unlike DPO, rollout groups are not collapsed into chosen/rejected pairs. Each row is one rollout viewed from one seat, with a group-normalized advantage computed among rollouts sharing the same prompt and wall seed.

## Columns

- `prompt_input_ids`: IDs from `<bos>` through `game_start`, including rule tokens and `view_imperfect_*`.
- `completion_input_ids`: Imperfect-information game continuation IDs through `<eos>`.
- `loss_mask`: Token-level policy-gradient mask for the viewer's discard and take/pass decision tokens only.
- `reward`: placement reward plus score reward.
- `advantage`: same-seat group-normalized advantage.
- `group_reward_mean`, `group_reward_std`: baseline statistics used to compute the advantage.
- `viewer_seat`, `rank`, `final_score`, `seed_id`, `rule_key`, `generation_index`: provenance.

## Reward

- placement weight: {placement_weight}
- score weight: {score_weight}

## Build Summary

- rows: {row_count}
- train rows: {train_count}
- validation rows: {eval_count}
- rollout groups scanned: {batch_count}
- invalid/no-rank groups discarded: {discarded_no_valid_group}
- no-decision rows discarded: {discarded_no_decisions}
"""


if __name__ == "__main__":
    main()
