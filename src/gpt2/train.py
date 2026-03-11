from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import TinyGPT2Config, TrainingConfig
from .data import PackedBatch, PackedGroupCollator, build_group_batch_sampler, load_grouped_dataset, split_grouped_dataset, validate_grouped_dataset
from .model import build_tiny_gpt2_model, count_parameters
from tenhou_tokenizer import load_hf_tokenizer


def _require_training_deps():
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch-backed transformers training dependencies are required. Install torch and retry."
        ) from exc
    return torch, DataLoader, get_cosine_schedule_with_warmup


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = _require_training_deps()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(torch, config: TrainingConfig, device) -> object | None:
    if device.type == "cuda" and config.use_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def _init_wandb(config: TrainingConfig, model_config: TinyGPT2Config, run_metadata: dict) -> object | None:
    if config.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        if config.require_wandb:
            raise
        return None
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        tags=list(config.wandb_tags),
        config={
            "training": {
                **json.loads(config.to_json()),
            },
            "model": asdict(model_config),
            **run_metadata,
        },
    )
    return wandb


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_cosine_scheduler_with_floor(optimizer, *, warmup_steps: int, total_steps: int, min_lr_ratio: float, torch):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-12, float(current_step + 1) / float(max(1, warmup_steps)))
        progress_denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current_step - warmup_steps) / progress_denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step_{step:07d}"


def _trim_checkpoints(output_dir: Path, keep_last: int) -> None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return
    checkpoints = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    if len(checkpoints) <= keep_last:
        return
    for stale in checkpoints[:-keep_last]:
        shutil.rmtree(stale)


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    config: TrainingConfig,
    model_config: TinyGPT2Config,
) -> None:
    torch, _, _ = _require_training_deps()
    ckpt_dir = _checkpoint_dir(output_dir, step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir / "model"))
    _save_json(ckpt_dir / "training_config.json", json.loads(config.to_json()))
    _save_json(ckpt_dir / "model_config.json", asdict(model_config))
    if config.save_optimizer_state:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "rng_state": torch.random.get_rng_state(),
            },
            ckpt_dir / "trainer_state.pt",
        )
    _trim_checkpoints(output_dir, config.keep_last_checkpoints)


def _evaluate(model, dataloader, device, autocast_dtype, torch) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    packed_tokens = 0
    padded_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            assert isinstance(batch, PackedBatch)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    labels=batch.labels.to(device),
                )
            losses.append(float(outputs.loss.detach().cpu()))
            packed_tokens += batch.stats.packed_tokens
            padded_tokens += batch.stats.padded_tokens
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "eval/loss": mean_loss,
        "eval/perplexity": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
        "eval/packing_efficiency": packed_tokens / padded_tokens if padded_tokens else 0.0,
    }


def _prepare_train_eval_datasets(training_config: TrainingConfig) -> tuple[object, object | None]:
    train_dataset = load_grouped_dataset(training_config.dataset_dirs)
    validate_grouped_dataset(train_dataset)
    if training_config.eval_dataset_dirs:
        eval_dataset = load_grouped_dataset(training_config.eval_dataset_dirs)
        validate_grouped_dataset(eval_dataset)
        return train_dataset, eval_dataset
    if training_config.train_split_eval_ratio == 0.0:
        return train_dataset, None
    train_dataset, eval_dataset = split_grouped_dataset(
        train_dataset,
        eval_ratio=training_config.train_split_eval_ratio,
        seed=training_config.split_seed,
    )
    validate_grouped_dataset(train_dataset)
    validate_grouped_dataset(eval_dataset)
    return train_dataset, eval_dataset


def train(
    *,
    model_config: TinyGPT2Config,
    training_config: TrainingConfig,
) -> dict[str, float]:
    torch, DataLoader, _unused_scheduler = _require_training_deps()
    model_config.validate()
    training_config.validate()
    if training_config.max_seq_length > model_config.n_positions:
        raise ValueError(
            f"max_seq_length={training_config.max_seq_length} exceeds model context n_positions={model_config.n_positions}"
        )

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("src/gpt2/train.py currently supports single-process training only")

    _set_global_seed(training_config.seed)
    if training_config.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = load_hf_tokenizer(training_config.tokenizer_dir)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None or tokenizer.bos_token_id is None:
        raise ValueError("tokenizer must define bos/eos/pad token ids")

    train_dataset, eval_dataset = _prepare_train_eval_datasets(training_config)

    device = _select_device(torch)
    autocast_dtype = _resolve_dtype(torch, training_config, device)
    model = build_tiny_gpt2_model(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        config=model_config,
    )
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if training_config.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = _build_cosine_scheduler_with_floor(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.train_steps,
        min_lr_ratio=training_config.min_learning_rate_ratio,
        torch=torch,
    )

    collator = PackedGroupCollator(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=training_config.max_seq_length,
        pad_to_multiple_of=training_config.pad_to_multiple_of,
        return_tensors="pt",
    )

    run_metadata = {
        "train_group_count": len(dict.fromkeys(train_dataset["group_id"])),
        "eval_group_count": 0 if eval_dataset is None else len(dict.fromkeys(eval_dataset["group_id"])),
        "parameter_count": count_parameters(model),
        "device": str(device),
    }
    wandb = _init_wandb(training_config, model_config, run_metadata)

    output_dir = training_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "training_config.json", json.loads(training_config.to_json()))
    _save_json(output_dir / "model_config.json", asdict(model_config))

    global_step = 0
    accumulation_step = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    latest_metrics: dict[str, float] = {}
    epoch = 0

    if training_config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    while global_step < training_config.train_steps:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=build_group_batch_sampler(
                train_dataset,
                max_tokens_per_batch=training_config.max_tokens_per_batch,
                shuffle=True,
                seed=training_config.seed + epoch,
            ),
            collate_fn=collator,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=training_config.pin_memory and device.type == "cuda",
        )
        for batch in train_loader:
            if global_step >= training_config.train_steps:
                break
            assert isinstance(batch, PackedBatch)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    labels=batch.labels.to(device),
                )
                loss = outputs.loss / training_config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step + 1}: {float(loss.detach().cpu())}")
            loss.backward()
            accumulation_step += 1

            if accumulation_step % training_config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                latest_metrics = {
                    "train/loss": float(outputs.loss.detach().cpu()),
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "train/grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
                    "train/packing_efficiency": batch.stats.packing_efficiency,
                    "train/packed_rows": float(batch.stats.packed_row_count),
                    "train/segments": float(batch.stats.segment_count),
                    "train/tokens_per_step": float(batch.stats.packed_tokens),
                    "train/elapsed_sec": time.time() - start_time,
                }
                if wandb is not None:
                    wandb.log(latest_metrics, step=global_step)
                if global_step % training_config.log_interval == 0:
                    print(json.dumps({"step": global_step, **latest_metrics}, ensure_ascii=False))
                if eval_dataset is not None and global_step % training_config.eval_interval == 0:
                    eval_loader = DataLoader(
                        eval_dataset,
                        batch_sampler=build_group_batch_sampler(
                            eval_dataset,
                            max_tokens_per_batch=training_config.eval_max_tokens_per_batch,
                            shuffle=False,
                            seed=training_config.seed,
                        ),
                        collate_fn=collator,
                        num_workers=0,
                        pin_memory=training_config.pin_memory and device.type == "cuda",
                    )
                    eval_metrics = _evaluate(model, eval_loader, device, autocast_dtype, torch)
                    latest_metrics.update(eval_metrics)
                    if wandb is not None:
                        wandb.log(eval_metrics, step=global_step)
                    print(json.dumps({"step": global_step, **eval_metrics}, ensure_ascii=False))
                if global_step % training_config.save_interval == 0:
                    _save_checkpoint(
                        output_dir=output_dir,
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=training_config,
                        model_config=model_config,
                    )
        epoch += 1

    final_dir = output_dir / "final_model"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    if wandb is not None:
        wandb.finish()
    return latest_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GPT-2 model on grouped Tenhou multiview datasets.")
    parser.add_argument("--dataset-dir", action="append", required=True, help="Hugging Face dataset directory.")
    parser.add_argument("--eval-dataset-dir", action="append", default=[], help="Optional held-out dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"), help="Tokenizer directory.")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-inner", type=int, default=256)
    parser.add_argument("--n-positions", type=int, default=8192)
    parser.add_argument("--max-tokens-per-batch", type=int, default=65536)
    parser.add_argument("--eval-max-tokens-per-batch", type=int, default=65536)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", default="mahjonglm-gpt2")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = TinyGPT2Config(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_inner=args.n_inner,
        n_positions=args.n_positions,
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        dataset_dirs=tuple(Path(path) for path in args.dataset_dir),
        eval_dataset_dirs=tuple(Path(path) for path in args.eval_dataset_dir),
        tokenizer_dir=args.tokenizer_dir,
        train_steps=args.train_steps,
        max_tokens_per_batch=args.max_tokens_per_batch,
        eval_max_tokens_per_batch=args.eval_max_tokens_per_batch,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    metrics = train(model_config=model_config, training_config=training_config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
