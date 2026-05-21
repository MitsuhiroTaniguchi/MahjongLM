from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import wandb


def load_jsonl_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in payload:
                records.append(payload)
    return records


def merge_records(
    sources: list[tuple[Path, int | None, int | None]],
    drop_keys: set[str],
    cumulative_keys: set[str],
    include_keys: set[str],
) -> OrderedDict[int, dict[str, Any]]:
    merged: OrderedDict[int, dict[str, Any]] = OrderedDict()
    cumulative_offsets = {key: 0.0 for key in cumulative_keys}
    for path, start_step, end_step in sources:
        source_last_values: dict[str, float] = {}
        for payload in load_jsonl_metrics(path):
            step = int(payload["step"])
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            record = merged.setdefault(step, {"trainer/global_step": step})
            for key, value in payload.items():
                if key == "step" or key == "trainer/global_step" or key in drop_keys:
                    continue
                if include_keys and key not in include_keys:
                    continue
                if key in cumulative_offsets and isinstance(value, int | float):
                    value = value + cumulative_offsets[key]
                    source_last_values[key] = float(value)
                record[key] = value
        for key, value in source_last_values.items():
            cumulative_offsets[key] = value
    return OrderedDict((step, record) for step, record in sorted(merged.items()) if len(record) > 1)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_source(value: str) -> tuple[Path, int | None, int | None]:
    # Format: path[:start:end]. Empty bounds are accepted.
    parts = value.rsplit(":", 2)
    if len(parts) == 3 and (parts[1].isdigit() or parts[1] == "") and (parts[2].isdigit() or parts[2] == ""):
        start = int(parts[1]) if parts[1] else None
        end = int(parts[2]) if parts[2] else None
        return Path(parts[0]), start, end
    return Path(value), None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-upload a consolidated W&B history from local JSON metric logs.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="Metric log path, optionally path:start:end")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--drop-key", action="append", default=[])
    parser.add_argument(
        "--include-key",
        action="append",
        default=[],
        help="Only upload these metric keys, plus trainer/global_step. By default all non-dropped keys are uploaded.",
    )
    parser.add_argument(
        "--cumulative-key",
        action="append",
        default=[],
        help="Metric key whose per-source elapsed value should be stitched by cumulatively offsetting later sources.",
    )
    parser.add_argument(
        "--wandb-step-offset",
        type=int,
        default=None,
        help="Use monotonically increasing internal W&B steps starting after this offset while keeping trainer/global_step unchanged.",
    )
    parser.add_argument(
        "--system-sample-seconds",
        type=float,
        default=0.0,
        help="Keep the run open after metric upload so W&B can record at least one system metrics sample.",
    )
    args = parser.parse_args()

    sources = [parse_source(value) for value in args.source]
    merged = merge_records(sources, set(args.drop_key), set(args.cumulative_key), set(args.include_key))
    if not merged:
        raise RuntimeError("no metric records found")

    training_config = load_json(args.output_dir / "training_config.json")
    model_config = load_json(args.output_dir / "model_config.json")
    config = {
        "training_config": training_config,
        "model_config": model_config,
        "source_output_dir": str(args.output_dir),
        "merged_history_sources": [
            {"path": str(path), "start_step": start, "end_step": end} for path, start, end in sources
        ],
    }

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.name,
        config=config,
        tags=args.tag,
        notes=args.notes,
        resume="allow",
    )
    assert run is not None
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("*", step_metric="trainer/global_step")

    for index, (step, record) in enumerate(merged.items(), start=1):
        wandb_step = args.wandb_step_offset + index if args.wandb_step_offset is not None else step
        wandb.log(record, step=wandb_step)

    final_record = next(reversed(merged.values()))
    for key, value in final_record.items():
        if key != "trainer/global_step":
            run.summary[key] = value
    run.summary["trainer/global_step"] = max(merged)
    run.summary["merged/source_record_count"] = sum(len(load_jsonl_metrics(path)) for path, _, _ in sources)
    run.summary["merged/step_count"] = len(merged)
    run.summary["merged/min_step"] = min(merged)
    run.summary["merged/max_step"] = max(merged)
    if args.system_sample_seconds > 0:
        time.sleep(args.system_sample_seconds)
    wandb.finish()

    print(f"uploaded {len(merged)} steps to {args.entity + '/' if args.entity else ''}{args.project}/{args.run_id}")


if __name__ == "__main__":
    main()
