from __future__ import annotations

import argparse
import inspect
import os
import sys
import textwrap
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
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"],
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
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

    _patch_transformers_cache_constant()
    try:
        from unsloth import FastLanguageModel
        from trl import DPOConfig, DPOTrainer
        from trl.trainer.dpo_trainer import DataCollatorForPreference, pad
    except Exception as exc:
        raise RuntimeError(
            "Unsloth DPO dependencies are not importable. Install unsloth, trl, peft, bitsandbytes, "
            "and any transitive TRL extras required by the local version before launching training."
        ) from exc
    _patch_dpo_trainer_for_tokenized_datasets(DPOTrainer)
    _patch_dpo_trainer_for_loss_masks(DPOTrainer)
    _patch_dpo_trainer_tie_half_accuracy(DPOTrainer)
    _patch_dpo_trainer_eval_log_sections(DPOTrainer)

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
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        do_eval=eval_dataset is not None,
        eval_on_start=eval_dataset is not None,
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
        data_collator=MaskedPreferenceCollator(
            DataCollatorForPreference=DataCollatorForPreference,
            pad=pad,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    trainer.train()
    if eval_dataset is not None:
        trainer.evaluate(metric_key_prefix="eval")
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))


def _make_dpo_trainer(DPOTrainer, **kwargs):
    tokenizer = kwargs.pop("tokenizer")
    try:
        return DPOTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **kwargs)


class MaskedPreferenceCollator:
    def __init__(self, *, DataCollatorForPreference, pad, pad_token_id: int) -> None:
        self.base_collator = DataCollatorForPreference(pad_token_id=pad_token_id)
        self.pad = pad

    def __call__(self, examples):
        import torch

        output = self.base_collator(examples)
        if "chosen_loss_mask" not in examples[0] or "rejected_loss_mask" not in examples[0]:
            return output
        chosen_loss_mask = [torch.tensor(example["chosen_loss_mask"], dtype=torch.long) for example in examples]
        rejected_loss_mask = [torch.tensor(example["rejected_loss_mask"], dtype=torch.long) for example in examples]
        output["chosen_loss_mask"] = self.pad(chosen_loss_mask, padding_value=0)
        output["rejected_loss_mask"] = self.pad(rejected_loss_mask, padding_value=0)
        return output


def _patch_transformers_cache_constant() -> None:
    try:
        import transformers.utils.hub as hub
    except Exception:
        return
    if not hasattr(hub, "TRANSFORMERS_CACHE"):
        hub.TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.environ.get("HF_HOME", ""))


def _patch_dpo_trainer_for_tokenized_datasets(DPOTrainer) -> None:
    original_prepare_dataset = DPOTrainer._prepare_dataset

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        columns = set(getattr(dataset, "column_names", []) or [])
        if {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}.issubset(columns):
            return dataset
        return original_prepare_dataset(self, dataset, processing_class, args, dataset_name)

    DPOTrainer._prepare_dataset = _prepare_dataset


def _patch_dpo_trainer_for_loss_masks(DPOTrainer) -> None:
    original_concatenated_inputs = DPOTrainer.concatenated_inputs

    def concatenated_inputs(batch, padding_value):
        import torch
        from trl.trainer.dpo_trainer import pad_to_length

        output = original_concatenated_inputs(batch, padding_value)
        if "chosen_loss_mask" not in batch or "rejected_loss_mask" not in batch:
            return output
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_loss_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_loss_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_loss_mask"], max_completion_length, pad_value=0),
            ),
        )
        return output

    DPOTrainer.concatenated_inputs = staticmethod(concatenated_inputs)

    source = textwrap.dedent(inspect.getsource(DPOTrainer.concatenated_forward))
    if 'completion_attention_mask = concatenated_batch["completion_attention_mask"]' not in source:
        raise RuntimeError("TRL DPOTrainer.concatenated_forward changed; loss-mask patch needs review")
    source = source.replace(
        'completion_attention_mask = concatenated_batch["completion_attention_mask"]',
        (
            'completion_attention_mask = concatenated_batch["completion_attention_mask"]\n'
            '    completion_loss_mask = concatenated_batch.get("completion_loss_mask", completion_attention_mask)'
        ),
        1,
    )
    source = source.replace(
        "loss_mask = completion_attention_mask.bool()",
        "loss_mask = completion_loss_mask.bool()",
        1,
    )
    source = source.replace(
        "(torch.zeros_like(prompt_attention_mask), completion_attention_mask)",
        "(torch.zeros_like(prompt_attention_mask), completion_loss_mask)",
        1,
    )
    namespace = dict(DPOTrainer.concatenated_forward.__globals__)
    exec(source, namespace)
    DPOTrainer.concatenated_forward = namespace["concatenated_forward"]


def _patch_dpo_trainer_tie_half_accuracy(DPOTrainer) -> None:
    source = textwrap.dedent(inspect.getsource(DPOTrainer.get_batch_loss_metrics))
    needle = "reward_accuracies = (chosen_rewards > rejected_rewards).float()"
    if needle not in source:
        raise RuntimeError("TRL DPOTrainer.get_batch_loss_metrics changed; accuracy patch needs review")
    source = source.replace(
        needle,
        (
            "reward_accuracies = (chosen_rewards > rejected_rewards).float()\n"
            "    reward_tie_half_accuracies = reward_accuracies + 0.5 * (chosen_rewards == rejected_rewards).float()"
        ),
        1,
    )
    needle = 'metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()'
    if needle not in source:
        raise RuntimeError("TRL DPOTrainer.get_batch_loss_metrics metrics block changed; accuracy patch needs review")
    source = source.replace(
        needle,
        (
            needle + "\n"
            '    metrics[f"{prefix}rewards/tie_half_accuracies"] = ('
            "self.accelerator.gather_for_metrics(reward_tie_half_accuracies).mean().item())"
        ),
        1,
    )
    namespace = dict(DPOTrainer.get_batch_loss_metrics.__globals__)
    exec(source, namespace)
    DPOTrainer.get_batch_loss_metrics = namespace["get_batch_loss_metrics"]


def _patch_dpo_trainer_eval_log_sections(DPOTrainer) -> None:
    def log(self, logs, start_time=None):
        import torch

        train_eval = "train" if "loss" in logs else "eval"
        rewritten_logs = {}
        for key, metrics in self._stored_metrics[train_eval].items():
            value = torch.tensor(metrics).mean().item()
            if train_eval == "eval" and key.startswith("eval_"):
                rewritten_logs["eval/" + key.removeprefix("eval_")] = value
            else:
                rewritten_logs[key] = value
        del self._stored_metrics[train_eval]
        if train_eval == "eval":
            for key, value in logs.items():
                if key.startswith("eval_"):
                    rewritten_logs["eval/" + key.removeprefix("eval_")] = value
                elif key.startswith("eval/"):
                    rewritten_logs[key] = value
                else:
                    rewritten_logs[key] = value
            logs = rewritten_logs
        else:
            logs.update(rewritten_logs)
        return super(DPOTrainer, self).log(logs, start_time)

    DPOTrainer.log = log


if __name__ == "__main__":
    main()
