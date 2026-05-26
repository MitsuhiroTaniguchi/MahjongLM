from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from tenhou_tokenizer.huggingface import MahjongTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MahjongLM with same-seed group-advantage RL.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ],
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--advantage-clip", type=float, default=3.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--wandb-project", type=str, default="mahjongLM_group_advantage_rl")
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
        raise RuntimeError("RL training requires WANDB_MODE=online")

    _patch_transformers_cache_constant()
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError("Unsloth is required for RL fine-tuning.") from exc

    dataset = load_from_disk(str(args.dataset_dir))
    train_dataset = dataset["train"] if hasattr(dataset, "keys") else dataset
    eval_dataset = dataset["validation"] if hasattr(dataset, "keys") and "validation" in dataset else None
    if eval_dataset is not None and args.max_eval_samples > 0 and len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

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
    if hasattr(model, "config"):
        model.config.use_cache = False

    training_args = TrainingArguments(
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
    trainer = GroupAdvantageRLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RLCollator(pad_token_id=tokenizer.pad_token_id, max_seq_length=args.max_seq_length),
        kl_coef=args.kl_coef,
        advantage_clip=args.advantage_clip,
    )
    trainer.train()
    if eval_dataset is not None:
        trainer.evaluate(metric_key_prefix="eval")
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))


class RLCollator:
    def __init__(self, *, pad_token_id: int, max_seq_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[Any]]:
        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        loss_masks: list[torch.Tensor] = []
        for example in examples:
            prompt_ids = list(example["prompt_input_ids"])
            completion_ids = list(example["completion_input_ids"])
            completion_loss_mask = list(example["loss_mask"])
            max_completion = max(0, self.max_seq_length - len(prompt_ids))
            completion_ids = completion_ids[:max_completion]
            completion_loss_mask = completion_loss_mask[:max_completion]
            ids = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + completion_loss_mask
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_masks.append(torch.ones(len(ids), dtype=torch.long))
            loss_masks.append(torch.tensor(mask, dtype=torch.long))
        return {
            "input_ids": pad(input_ids, padding_value=self.pad_token_id),
            "attention_mask": pad(attention_masks, padding_value=0),
            "loss_mask": pad(loss_masks, padding_value=0),
            "advantages": torch.tensor([float(example["advantage"]) for example in examples], dtype=torch.float32),
            "rewards": torch.tensor([float(example["reward"]) for example in examples], dtype=torch.float32),
            "ranks": torch.tensor([int(example["rank"]) for example in examples], dtype=torch.float32),
            "viewer_seats": torch.tensor([int(example["viewer_seat"]) for example in examples], dtype=torch.long),
        }


def pad(values: list[torch.Tensor], *, padding_value: int) -> torch.Tensor:
    max_len = max(value.shape[0] for value in values)
    output = values[0].new_full((len(values), max_len), padding_value)
    for row, value in enumerate(values):
        output[row, : value.shape[0]] = value
    return output


class GroupAdvantageRLTrainer(Trainer):
    def __init__(self, *args, kl_coef: float, advantage_clip: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef
        self.advantage_clip = advantage_clip
        self._pending_train_rows: list[dict[str, float]] = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        advantages = inputs.pop("advantages").to(model.device)
        rewards = inputs.pop("rewards").to(model.device)
        ranks = inputs.pop("ranks").to(model.device)
        inputs.pop("viewer_seats", None)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        loss_mask = inputs["loss_mask"].to(model.device).bool()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        token_logps, token_mask = masked_token_logps(outputs.logits, input_ids, loss_mask)
        policy_logp = masked_mean(token_logps, token_mask)

        with torch.no_grad():
            with disabled_adapters(model):
                ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            ref_token_logps, _ = masked_token_logps(ref_outputs.logits, input_ids, loss_mask)

        clipped_advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)
        policy_loss = -(clipped_advantages * policy_logp).mean()
        log_ratio = ref_token_logps - token_logps
        kl_tokens = torch.exp(log_ratio) - log_ratio - 1.0
        kl = masked_mean(kl_tokens, token_mask).mean()
        loss = policy_loss + self.kl_coef * kl

        if model.training:
            self._pending_train_rows.extend(
                rl_metric_rows(
                    policy_logp=policy_logp,
                    ref_token_logps=ref_token_logps,
                    token_logps=token_logps,
                    token_mask=token_mask,
                    advantages=advantages,
                    rewards=rewards,
                    ranks=ranks,
                    advantage_clip=self.advantage_clip,
                )
            )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, dict(inputs), return_outputs=False)
        return loss.detach(), None, None

    def log(self, logs, start_time=None):
        if "loss" in logs and self._pending_train_rows:
            logs.update(aggregate_rl_rows(self._pending_train_rows))
            self._pending_train_rows.clear()
        return super().log(logs, start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        detailed = self.compute_detailed_eval_metrics(eval_dataset=eval_dataset)
        if detailed:
            prefixed = {f"{metric_key_prefix}/{key}": value for key, value in detailed.items()}
            log_wandb_direct(prefixed, step=self.state.global_step)
            print(prefixed, flush=True)
            metrics.update(prefixed)
        return metrics

    def compute_detailed_eval_metrics(self, eval_dataset=None) -> dict[str, float]:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {}
        dataloader = self.get_eval_dataloader(dataset)
        model = self.model
        was_training = model.training
        model.eval()
        collected: list[dict[str, float]] = []
        try:
            for inputs in dataloader:
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    collected.extend(self._batch_eval_rows(model, dict(inputs)))
        finally:
            if was_training:
                model.train()
        return aggregate_rl_rows(collected)

    def _batch_eval_rows(self, model, inputs) -> list[dict[str, float]]:
        advantages = inputs.pop("advantages").to(model.device)
        rewards = inputs.pop("rewards").to(model.device)
        ranks = inputs.pop("ranks").to(model.device)
        inputs.pop("viewer_seats", None)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        loss_mask = inputs["loss_mask"].to(model.device).bool()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        token_logps, token_mask = masked_token_logps(outputs.logits, input_ids, loss_mask)
        policy_logp = masked_mean(token_logps, token_mask)
        with disabled_adapters(model):
            ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        ref_token_logps, _ = masked_token_logps(ref_outputs.logits, input_ids, loss_mask)
        return rl_metric_rows(
            policy_logp=policy_logp,
            ref_token_logps=ref_token_logps,
            token_logps=token_logps,
            token_mask=token_mask,
            advantages=advantages,
            rewards=rewards,
            ranks=ranks,
            advantage_clip=self.advantage_clip,
        )


def masked_token_logps(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]
    logps = F.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    logps = logps.masked_fill(~shift_mask, 0.0)
    return logps, shift_mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1).to(values.dtype)
    return (values * mask.to(values.dtype)).sum(dim=1) / denom


def rl_metric_rows(
    *,
    policy_logp: torch.Tensor,
    ref_token_logps: torch.Tensor,
    token_logps: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    ranks: torch.Tensor,
    advantage_clip: float,
) -> list[dict[str, float]]:
    detached_logp = policy_logp.detach().float()
    detached_adv = advantages.detach().float()
    ref_logp = masked_mean(ref_token_logps.detach().float(), token_mask)
    log_ratio = ref_token_logps.detach().float() - token_logps.detach().float()
    kl_tokens = torch.exp(log_ratio) - log_ratio - 1.0
    sample_kl = masked_mean(kl_tokens, token_mask)
    token_counts = token_mask.sum(dim=1).detach().float()
    clipped_adv = detached_adv.clamp(-advantage_clip, advantage_clip)
    rows: list[dict[str, float]] = []
    for idx in range(detached_logp.shape[0]):
        rows.append(
            {
                "policy_logp": detached_logp[idx].item(),
                "ref_logp": ref_logp[idx].item(),
                "logp_delta": (detached_logp[idx] - ref_logp[idx]).item(),
                "advantage": detached_adv[idx].item(),
                "advantage_abs": detached_adv[idx].abs().item(),
                "reward": rewards.detach().float()[idx].item(),
                "rank": ranks.detach().float()[idx].item(),
                "decision_tokens": token_counts[idx].item(),
                "kl": sample_kl[idx].item(),
                "policy_loss": (-(clipped_adv[idx] * detached_logp[idx])).item(),
                "advantage_weighted_logp": (detached_adv[idx] * detached_logp[idx]).item(),
            }
        )
    return rows


def aggregate_rl_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    def values(key: str) -> torch.Tensor:
        return torch.tensor([row[key] for row in rows], dtype=torch.float32)

    policy_logp = values("policy_logp")
    ref_logp = values("ref_logp")
    advantage = values("advantage")
    reward = values("reward")
    rank = values("rank")
    token_counts = values("decision_tokens")
    pos = advantage > 0
    neg = advantage < 0
    zero = advantage == 0
    logp_pos = policy_logp[pos].mean() if pos.any() else torch.tensor(float("nan"))
    logp_neg = policy_logp[neg].mean() if neg.any() else torch.tensor(float("nan"))
    centered_adv = advantage - advantage.mean()
    centered_logp = policy_logp - policy_logp.mean()
    corr_denom = centered_adv.square().mean().sqrt() * centered_logp.square().mean().sqrt()
    corr = (centered_adv * centered_logp).mean() / corr_denom.clamp_min(1e-8)
    return {
        "policy_loss": values("policy_loss").mean().item(),
        "kl": values("kl").mean().item(),
        "reward_mean": reward.mean().item(),
        "reward_std": reward.std(unbiased=False).item(),
        "advantage_mean": advantage.mean().item(),
        "advantage_std": advantage.std(unbiased=False).item(),
        "advantage_abs_mean": values("advantage_abs").mean().item(),
        "advantage_pos_fraction": pos.float().mean().item(),
        "advantage_neg_fraction": neg.float().mean().item(),
        "advantage_zero_fraction": zero.float().mean().item(),
        "rank_mean": rank.mean().item(),
        "decision_tokens_mean": token_counts.mean().item(),
        "decision_tokens_min": token_counts.min().item(),
        "decision_tokens_max": token_counts.max().item(),
        "policy_logp_mean": policy_logp.mean().item(),
        "ref_logp_mean": ref_logp.mean().item(),
        "logp_delta_mean": (policy_logp - ref_logp).mean().item(),
        "logp_pos_adv": logp_pos.item(),
        "logp_neg_adv": logp_neg.item(),
        "logp_pos_minus_neg": (logp_pos - logp_neg).item(),
        "advantage_logp_corr": corr.item(),
        "advantage_weighted_logp": values("advantage_weighted_logp").mean().item(),
    }


def disabled_adapters(model):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    if hasattr(model, "disable_adapters"):
        return model.disable_adapters()
    return nullcontext()


def log_wandb_direct(metrics: dict[str, float], *, step: int) -> None:
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    wandb.log(metrics)


def _patch_transformers_cache_constant() -> None:
    try:
        import transformers.utils.hub as hub
    except Exception:
        return
    if not hasattr(hub, "TRANSFORMERS_CACHE"):
        hub.TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.environ.get("HF_HOME", ""))


if __name__ == "__main__":
    main()
