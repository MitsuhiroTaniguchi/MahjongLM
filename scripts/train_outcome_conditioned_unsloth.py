from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unsloth  # noqa: F401  # Import before transformers/peft so Unsloth can patch fast paths.
from datasets import concatenate_datasets, load_from_disk

from gpt2.outcome_conditioning import build_outcome_conditioned_input_ids
from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune MahjongLM with Unsloth on view_imperfect rows whose final_rank block is "
            "prepended after <bos> at collate time."
        )
    )
    parser.add_argument("--model-name", type=str, default="mitsutani/mahjonglm-10m")
    parser.add_argument("--dataset-repo", type=str, default="mitsutani/mahjonglm-dataset")
    parser.add_argument("--dataset-dir", action="append", type=Path, default=[])
    parser.add_argument("--dataset-year", action="append", type=int, default=[])
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mahjonglm-10m-outcome-conditioned")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--drop-overlong", action="store_true")
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument(
        "--max-eval-groups",
        type=int,
        default=1024,
        help="Cap eval games by default so periodic eval does not scan the full held-out split. Use 0 for full eval.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=0,
        help="Optional hard cap on eval rows after group limiting. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-source-rows-per-dir",
        type=int,
        default=0,
        help="Debug/smoke option. Select only the first N raw rows per dataset dir before filtering view_imperfect.",
    )
    parser.add_argument("--train-split-eval-ratio", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=0.2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["auto"],
        help="LoRA target module suffixes. Use the default 'auto' to infer Qwen/Llama/GPT-style names.",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--no-tf32", action="store_false", dest="tf32", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="mahjongLM_outcome_conditioned")
    parser.add_argument("--wandb-entity", type=str, default="a21-3jck-")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--hub-token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_wandb and args.wandb_mode != "online":
        raise RuntimeError("--require-wandb requires --wandb-mode online")
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    _patch_transformers_cache_constant()
    try:
        import torch
        from transformers import Trainer, TrainingArguments
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError(
            "Unsloth training dependencies are not importable. Install requirements-dpo.txt "
            "or equivalent CUDA-matched unsloth/transformers/trl/peft/bitsandbytes packages."
        ) from exc

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_dirs = args.dataset_dir or _download_year_dirs(args)
    dataset = _load_view_imperfect_dataset(dataset_dirs, max_source_rows_per_dir=args.max_source_rows_per_dir)
    group_column = "group_id" if "group_id" in dataset.column_names else "game_id"
    train_dataset, eval_dataset = split_dataset_by_group(
        dataset,
        group_column=group_column,
        eval_ratio=args.train_split_eval_ratio,
        seed=args.split_seed,
    )
    train_dataset = limit_dataset_groups(train_dataset, args.max_train_groups, group_column=group_column)
    eval_dataset = limit_dataset_groups(eval_dataset, args.max_eval_groups, group_column=group_column)
    if args.max_eval_rows > 0 and len(eval_dataset) > args.max_eval_rows:
        eval_dataset = eval_dataset.select(range(args.max_eval_rows))

    collator = OutcomeConditionedCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        pad_to_multiple_of=8,
        drop_overlong=args.drop_overlong,
    )
    if args.drop_overlong:
        train_dataset = _drop_overlong(train_dataset, collator)
        eval_dataset = _drop_overlong(eval_dataset, collator)
        if len(train_dataset) == 0:
            raise ValueError("--drop-overlong removed all train rows; increase --max-seq-length")
        if len(eval_dataset) == 0:
            raise ValueError("--drop-overlong removed all eval rows; increase --max-seq-length or lower eval ratio")

    model, _loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if args.bf16 else None,
        load_in_4bit=args.load_in_4bit if args.finetune_mode == "lora" else False,
        load_in_8bit=False,
        load_in_16bit=args.finetune_mode == "full",
        full_finetuning=args.finetune_mode == "full",
        trust_remote_code=True,
        token=args.hub_token,
    )
    if args.finetune_mode == "lora":
        target_modules = _resolve_lora_target_modules(model, args.target_modules)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
            random_state=args.seed,
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    training_args = _make_training_args(
        TrainingArguments,
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if len(eval_dataset) else "no",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        tf32=args.tf32,
        optim="adamw_8bit" if args.load_in_4bit and args.finetune_mode == "lora" else "adamw_torch",
        report_to=[] if args.wandb_mode == "disabled" else ["wandb"],
        run_name=args.wandb_run_name or None,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )
    trainer = _make_trainer(
        Trainer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.log(
        {
            "data/train_rows": len(train_dataset),
            "data/eval_rows": len(eval_dataset),
        }
    )
    trainer.train()
    if len(eval_dataset):
        trainer.evaluate(metric_key_prefix="eval")
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))


class OutcomeConditionedCollator:
    def __init__(
        self,
        *,
        tokenizer: MahjongTokenizerFast,
        max_seq_length: int,
        pad_to_multiple_of: int = 8,
        drop_overlong: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.drop_overlong = drop_overlong

    def build_input_ids(self, feature: dict) -> list[int]:
        input_ids = build_outcome_conditioned_input_ids(
            feature["input_ids"],
            tokenizer=self.tokenizer,
            seat_count=int(feature["seat_count"]),
        )
        if len(input_ids) > self.max_seq_length:
            message = (
                f"outcome-conditioned sequence for {feature.get('game_id')} seat "
                f"{feature.get('viewer_seat')} is {len(input_ids)} tokens > {self.max_seq_length}"
            )
            if self.drop_overlong:
                raise OverflowError(message)
            raise ValueError(message + "; increase --max-seq-length or use --drop-overlong")
        return input_ids

    def __call__(self, features: list[dict]) -> dict[str, object]:
        import torch

        rows = [self.build_input_ids(feature) for feature in features]
        max_length = max(len(row) for row in rows)
        if self.pad_to_multiple_of > 1 and max_length % self.pad_to_multiple_of:
            max_length += self.pad_to_multiple_of - (max_length % self.pad_to_multiple_of)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")

        input_ids = []
        attention_mask = []
        labels = []
        for row in rows:
            pad_len = max_length - len(row)
            input_ids.append(row + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)
            labels.append(row + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _download_year_dirs(args: argparse.Namespace) -> list[Path]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required when --dataset-dir is not supplied") from exc
    years = args.dataset_year or list(range(2011, 2025))
    allow_patterns = [f"{year}/*" for year in years]
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir is not None else None,
            token=args.hub_token,
        )
    )
    return [snapshot_dir / str(year) for year in years]


def _load_view_imperfect_dataset(dataset_dirs: list[Path], *, max_source_rows_per_dir: int = 0):
    datasets = []
    for dataset_dir in dataset_dirs:
        dataset = load_from_disk(str(dataset_dir))
        if max_source_rows_per_dir > 0:
            dataset = dataset.select(range(min(len(dataset), max_source_rows_per_dir)))
        if "view_type" not in dataset.column_names:
            raise ValueError(f"{dataset_dir} does not contain view_type")
        datasets.append(dataset.filter(lambda view_type: view_type == "imperfect", input_columns=["view_type"]))
    if not datasets:
        raise ValueError("no datasets loaded")
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if len(dataset) == 0:
        raise ValueError("view_imperfect dataset is empty")
    if "group_id" not in dataset.column_names and "game_id" not in dataset.column_names:
        raise ValueError("dataset must contain either group_id or game_id")
    return dataset


def limit_dataset_groups(dataset, max_groups: int, *, group_column: str):
    if max_groups <= 0:
        return dataset
    chosen: set[str] = set()
    for group_id in dataset[group_column]:
        chosen.add(group_id)
        if len(chosen) >= max_groups:
            break
    return dataset.filter(lambda group_id: group_id in chosen, input_columns=[group_column])


def split_dataset_by_group(dataset, *, group_column: str, eval_ratio: float, seed: int):
    import math

    if not (0.0 <= eval_ratio < 1.0):
        raise ValueError("eval_ratio must be in [0, 1)")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if eval_ratio == 0.0:
        return dataset, dataset.select([])
    threshold = max(1, min(999_999, math.floor(eval_ratio * 1_000_000)))

    def is_eval_group(group_id: str) -> bool:
        payload = f"{seed}:{group_id}".encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % 1_000_000
        return value < threshold

    train_dataset = dataset.filter(lambda group_id: not is_eval_group(group_id), input_columns=[group_column])
    eval_dataset = dataset.filter(lambda group_id: is_eval_group(group_id), input_columns=[group_column])
    if len(train_dataset) == 0:
        raise ValueError("train split is empty; lower eval_ratio or provide more data")
    if len(eval_dataset) == 0:
        raise ValueError("eval split is empty; raise eval_ratio")
    return train_dataset, eval_dataset


def _drop_overlong(dataset, collator: OutcomeConditionedCollator):
    keep_indices = []
    for index, row in enumerate(dataset):
        try:
            collator.build_input_ids(row)
        except OverflowError:
            continue
        keep_indices.append(index)
    return dataset.select(keep_indices)


def _resolve_lora_target_modules(model, target_modules: list[str]) -> list[str]:
    if target_modules != ["auto"]:
        return target_modules
    module_names = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
    qwen_like = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    gpt_like = ["c_attn", "c_proj", "c_fc"]
    resolved = [name for name in qwen_like if name in module_names]
    if resolved:
        return resolved
    resolved = [name for name in gpt_like if name in module_names]
    if resolved:
        return resolved
    raise ValueError("could not infer LoRA target modules; pass --target-modules explicitly")


def _make_training_args(TrainingArguments, **kwargs):
    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" not in signature.parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def _make_trainer(Trainer, **kwargs):
    tokenizer = kwargs.pop("tokenizer")
    signature = inspect.signature(Trainer)
    if "processing_class" in signature.parameters:
        return Trainer(processing_class=tokenizer, **kwargs)
    if "tokenizer" in signature.parameters:
        return Trainer(tokenizer=tokenizer, **kwargs)
    return Trainer(**kwargs)


def _patch_transformers_cache_constant() -> None:
    try:
        import transformers.utils.hub as hub
    except Exception:
        return
    if not hasattr(hub, "TRANSFORMERS_CACHE"):
        hub.TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.environ.get("HF_HOME", ""))


if __name__ == "__main__":
    main()
