from __future__ import annotations

import argparse
import math
import os
from itertools import chain
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-2 style model on MahjongLM.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/2021"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpt2-mahjong-2021"))
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--wandb-project", type=str, default="MahjongLM")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-positions", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def build_block_dataset(dataset, block_size: int):
    keep_columns = ["input_ids"]
    dataset = dataset.select_columns(keep_columns)

    def group_texts(examples):
        concatenated = list(chain.from_iterable(examples["input_ids"]))
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}

        blocks = [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        return {
            "input_ids": blocks,
            "labels": [block[:] for block in blocks],
        }

    return dataset.map(
        group_texts,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Packing token sequences",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    block_size = min(args.block_size, args.n_positions)
    use_cpu = args.use_cpu or not torch.cuda.is_available()

    dataset = load_from_disk(str(args.dataset_path))
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)

    if args.max_train_samples > 0:
        split["train"] = split["train"].select(range(min(args.max_train_samples, len(split["train"]))))
    if args.max_eval_samples > 0:
        split["test"] = split["test"].select(range(min(args.max_eval_samples, len(split["test"]))))

    train_dataset = build_block_dataset(split["train"], block_size)
    eval_dataset = build_block_dataset(split["test"], block_size)

    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    config = GPT2Config(
        vocab_size=65536,
        n_positions=block_size,
        n_ctx=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        bos_token_id=0,
        eos_token_id=0,
    )

    model = GPT2LMHeadModel(config)
    model.config.use_cache = False

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        fp16=args.fp16,
        report_to=None if args.report_to == "none" else args.report_to,
        remove_unused_columns=False,
        use_cpu=use_cpu,
        seed=args.seed,
    )

    if use_cpu:
        print("Using CPU training.")
    else:
        print(f"Using GPU training on: {torch.cuda.get_device_name(0)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(args.output_dir)

    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        eval_metrics["perplexity"] = (
            math.exp(eval_metrics["eval_loss"])
            if eval_metrics["eval_loss"] < 20
            else float("inf")
        )
    eval_metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
