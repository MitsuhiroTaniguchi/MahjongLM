from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import concatenate_datasets, load_from_disk

from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample balanced view_omniscient wall-prefix prompts for DPO rollout generation."
    )
    parser.add_argument("--dataset-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--per-rule", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    datasets = [load_from_disk(str(path)) for path in args.dataset_dir]
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    rng = random.Random(args.seed)

    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        row = dataset[idx]
        if row.get("view_type") != "omniscient":
            continue
        tokens = tokenizer.convert_ids_to_tokens(row["input_ids"])
        prompt = extract_first_wall_prompt(tokens)
        rule_key = " ".join(token for token in tokens if token.startswith("rule_"))
        if not rule_key:
            continue
        if len(buckets[rule_key]) >= args.per_rule:
            continue
        buckets[rule_key].append(
            {
                "game_id": row.get("game_id", ""),
                "seed_id": row.get("game_id", ""),
                "rule_key": rule_key,
                "seat_count": row.get("seat_count", 0),
                "prompt_tokens": prompt,
            }
        )
        if buckets and all(len(records) >= args.per_rule for records in buckets.values()):
            # Keep scanning light once all discovered rule buckets are full.
            pass

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for rule_key in sorted(buckets):
            for record in buckets[rule_key][: args.per_rule]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({rule_key: len(records) for rule_key, records in sorted(buckets.items())}, indent=2))


def extract_first_wall_prompt(tokens: list[str]) -> list[str]:
    out = list(tokens)
    if not out or out[0] != "<bos>":
        out.insert(0, "<bos>")
    try:
        wall_idx = out.index("wall")
    except ValueError as exc:
        raise ValueError("omniscient row does not contain a wall token") from exc
    wall_len = 108 if "rule_player_3" in out[:wall_idx] else 136
    end = wall_idx + 1 + wall_len
    if len(out) < end:
        raise ValueError("omniscient wall block is truncated")
    return out[:end]


if __name__ == "__main__":
    main()
