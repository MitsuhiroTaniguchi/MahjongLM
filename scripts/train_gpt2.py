from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from itertools import chain
from pathlib import Path

from datasets import load_from_disk
import pyarrow.compute as pc
import torch
import wandb
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    PreTrainedTokenizerFast,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-2 style model on MahjongLM.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/2021"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpt2-mahjong-2021"))
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--wandb-entity", type=str, default="a21-3jck-")
    parser.add_argument("--wandb-project", type=str, default="mahjongLM_gpt2")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer"))
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


def infer_dataset_tag(dataset_path: Path) -> str:
    parts = dataset_path.parts
    for part in reversed(parts):
        if part.isdigit():
            return f"y{part}"
    return dataset_path.name or "dataset"


def format_learning_rate(learning_rate: float) -> str:
    mantissa, exponent = f"{learning_rate:.0e}".split("e")
    return f"lr{mantissa}e{int(exponent)}"


def infer_vocab_size(dataset, tokenizer_path: Path | None) -> int:
    if tokenizer_path is not None and tokenizer_path.exists():
        tokenizer_file = tokenizer_path / "tokenizer.json"
        if tokenizer_file.exists():
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_file),
                unk_token="<unk>",
                pad_token="<pad>",
                bos_token="<bos>",
                eos_token="<eos>",
            )
            return len(tokenizer)

        vocab_file = tokenizer_path / "vocab.txt"
        if vocab_file.exists():
            with vocab_file.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        return len(tokenizer)

    max_token_id = pc.max(pc.list_flatten(dataset.data.column("input_ids"))).as_py()
    return int(max_token_id) + 1


def build_run_name(args: argparse.Namespace, block_size: int, vocab_size: int) -> str:
    dataset_tag = infer_dataset_tag(args.dataset_path)
    model_tag = f"gpt2-v{vocab_size}-l{args.n_layer}-h{args.n_head}-d{args.n_embd}"
    train_tag = f"bs{block_size}-s{args.max_steps}-{format_learning_rate(args.learning_rate)}"
    suffix = "cpu" if (args.use_cpu or not torch.cuda.is_available()) else "gpu"
    stamp = datetime.now().strftime("%m%d-%H%M")
    return f"mahjonglm-{dataset_tag}-{model_tag}-{train_tag}-{suffix}-{stamp}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    block_size = min(args.block_size, args.n_positions)
    use_cpu = args.use_cpu or not torch.cuda.is_available()

    dataset = load_from_disk(str(args.dataset_path))
    vocab_size = infer_vocab_size(dataset, args.tokenizer_path)
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
        vocab_size=vocab_size,
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
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb_run_name = args.wandb_run_name or build_run_name(args, block_size, vocab_size)
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            mode=args.wandb_mode,
            config={
                "learning_rate": args.learning_rate,
                "architecture": "GPT-2",
                "dataset": str(args.dataset_path),
                "vocab_size": vocab_size,
                "epochs": args.num_train_epochs,
                "block_size": block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "max_steps": args.max_steps,
                "max_train_samples": args.max_train_samples,
                "max_eval_samples": args.max_eval_samples,
                "wandb_run_name": wandb_run_name,
            },
        )
    else:
        wandb_run = None

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

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
