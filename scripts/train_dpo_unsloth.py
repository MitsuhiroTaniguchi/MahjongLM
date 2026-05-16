from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import load_from_disk

from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MahjongLM preference data with Unsloth DPO.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--max-prompt-length", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--wandb-project", type=str, default="mahjongLM_dpo_view_imperfect")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="a21-3jck-")
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.require_wandb and os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("DPO training requires WANDB_MODE=online")

    try:
        from unsloth import FastLanguageModel
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Unsloth DPO dependencies are not importable. Install unsloth, trl, peft, bitsandbytes, "
            "and any transitive TRL extras required by the local version before launching training."
        ) from exc

    dataset = load_from_disk(str(args.dataset_dir))
    train_dataset = dataset["train"] if hasattr(dataset, "keys") else dataset
    eval_dataset = dataset["validation"] if hasattr(dataset, "keys") and "validation" in dataset else None

    model, _loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    tokenizer = MahjongTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        bf16=True,
        tf32=True,
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to=["wandb"],
        run_name=args.wandb_run_name or None,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = _make_dpo_trainer(
        DPOTrainer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))


def _make_dpo_trainer(DPOTrainer, **kwargs):
    tokenizer = kwargs.pop("tokenizer")
    try:
        return DPOTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **kwargs)


if __name__ == "__main__":
    main()
