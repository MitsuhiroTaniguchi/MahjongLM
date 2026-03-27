from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .config import TinyGPT2Config, TinyQwen3Config, TrainingConfig
from .data import (
    PackedBatch,
    PackedGroupStats,
    PackedGroupCollator,
    UnpackedBatch,
    UnpackedCollator,
    build_split_cache_paths,
    count_unique_groups,
    build_fixed_batch_size_sampler,
    build_group_batch_sampler,
    load_cached_split,
    limit_groups,
    load_grouped_dataset,
    save_split_cache,
    split_grouped_dataset,
    validate_grouped_dataset,
)
from .muon import SingleDeviceMuonWithAuxAdam
from .model import build_tiny_gpt2_model, build_tiny_qwen3_model, count_parameters, load_saved_causal_lm
from tenhou_tokenizer import load_hf_tokenizer


ARCH_PRESETS: dict[str, TinyGPT2Config] = {
    "A": TinyGPT2Config(n_layer=20, n_head=10, n_embd=640, n_inner=None),
    "B": TinyGPT2Config(n_layer=10, n_head=14, n_embd=896, n_inner=None),
    "C": TinyGPT2Config(n_layer=16, n_head=11, n_embd=704, n_inner=None),
    "D": TinyGPT2Config(n_layer=24, n_head=9, n_embd=576, n_inner=None),
    "G100": TinyGPT2Config(n_layer=19, n_head=10, n_embd=640, n_inner=None, n_positions=8192),
}

QWEN3_ARCH_PRESETS: dict[str, TinyQwen3Config] = {
    "Q100": TinyQwen3Config(
        hidden_size=640,
        intermediate_size=1920,
        num_hidden_layers=20,
        num_attention_heads=10,
        num_key_value_heads=5,
        head_dim=64,
        max_position_embeddings=8192,
    ),
    "Q0": TinyQwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=31,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
    ),
    "Q1": TinyQwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
    ),
    "Q2": TinyQwen3Config(
        hidden_size=1088,
        intermediate_size=3264,
        num_hidden_layers=28,
        num_attention_heads=17,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
    ),
    "Q3": TinyQwen3Config(
        hidden_size=1056,
        intermediate_size=3168,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
    ),
    "Q4": TinyQwen3Config(
        hidden_size=1152,
        intermediate_size=3456,
        num_hidden_layers=25,
        num_attention_heads=18,
        num_key_value_heads=9,
        head_dim=128,
        max_position_embeddings=8192,
    ),
    "QM100S64": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=22,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=64,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "QM100S96": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=22,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=96,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "QM95S192": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=192,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "QM100S384": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=384,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "QM99E3S256": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=256,
        mamba3_expand=3,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q20E2S16": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=16,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q20E2S32": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=32,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q12E2S64M": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_with_mlp_block=True,
        mamba3_attention_period=4,
        mamba3_d_state=64,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q18E2S64G": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_gated_attention=True,
        use_mamba3_hybrid=True,
        mamba3_attention_period=2,
        mamba3_d_state=64,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q20E2S64": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=64,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q20E2S96": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=96,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q20E2S128": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=128,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
    "Q20E2S160": TinyQwen3Config(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=64,
        max_position_embeddings=8192,
        use_mamba3_hybrid=True,
        mamba3_attention_period=4,
        mamba3_d_state=160,
        mamba3_expand=2,
        mamba3_headdim=64,
        mamba3_ngroups=1,
    ),
}


def _require_training_deps():
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "torch-backed transformers training dependencies are required. Install torch and retry."
        ) from exc
    return torch, DataLoader, get_cosine_schedule_with_warmup


def _set_global_seed(seed: int) -> None:
    torch, _, _ = _require_training_deps()
    random.seed(seed)
    np.random.seed(seed)
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
    if device.type == "cuda":
        return torch.float16
    return None


def _init_wandb(
    config: TrainingConfig,
    model_config: TinyGPT2Config | TinyQwen3Config,
    run_metadata: dict,
    *,
    resume_run_id: str | None = None,
) -> object | None:
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
        id=resume_run_id,
        resume="must" if resume_run_id else None,
        tags=list(config.wandb_tags),
        config={
            "training": json.loads(config.to_json()),
            "model": asdict(model_config),
            **run_metadata,
        },
    )
    wandb.define_metric("trainer/global_step", summary="max")
    for metric_name in (
        "train/loss",
        "train/lr",
        "train/lr_aux",
        "train/grad_norm",
        "train/packing_efficiency",
        "train/packed_rows",
        "train/segments",
        "train/tokens_per_step",
        "train/elapsed_sec",
        "eval/loss",
        "eval/perplexity",
        "eval/packing_efficiency",
    ):
        wandb.define_metric(metric_name, step_metric="trainer/global_step")
    return wandb


def _wandb_log_if_available(wandb_run, payload: dict, *, step: int | None = None) -> None:
    if wandb_run is None:
        return
    record = dict(payload)
    if step is not None:
        record["trainer/global_step"] = step
        wandb_run.log(record, step=step)
        return
    wandb_run.log(record)


def _wandb_summary_update_if_available(wandb_run, payload: dict) -> None:
    if wandb_run is None:
        return
    for key, value in payload.items():
        wandb_run.summary[key] = value


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


def _build_linear_scheduler(optimizer, *, warmup_steps: int, total_steps: int, torch):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-12, float(current_step + 1) / float(max(1, warmup_steps)))
        progress_denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current_step - warmup_steps) / progress_denominator))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _collect_optimizer_lrs(*, optimizer, scheduler, training_config: TrainingConfig) -> dict[str, float]:
    lrs = [float(value) for value in scheduler.get_last_lr()]
    if training_config.optimizer_name != "muon":
        return {"train/lr": lrs[0]}
    muon_lr = None
    aux_lrs: list[float] = []
    for lr_value, group in zip(lrs, optimizer.param_groups):
        if group.get("use_muon", False):
            muon_lr = lr_value
        else:
            aux_lrs.append(lr_value)
    payload = {
        "train/lr": float(muon_lr if muon_lr is not None else lrs[0]),
    }
    if aux_lrs:
        payload["train/lr_aux"] = float(aux_lrs[0])
    return payload


def _partition_params_for_muon(model, torch):
    muon_params: list = []
    adam_decay_params: list = []
    adam_no_decay_params: list = []
    assigned = set()
    try:
        from transformers.pytorch_utils import Conv1D
    except ModuleNotFoundError:  # pragma: no cover - transformers is already a runtime dependency
        Conv1D = ()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.requires_grad and id(module.weight) not in assigned:
                adam_decay_params.append(module.weight)
                assigned.add(id(module.weight))
            continue
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            if module.weight.requires_grad and id(module.weight) not in assigned:
                if module_name.endswith("lm_head"):
                    adam_decay_params.append(module.weight)
                else:
                    muon_params.append(module.weight)
                assigned.add(id(module.weight))
            if module.bias is not None and module.bias.requires_grad and id(module.bias) not in assigned:
                adam_no_decay_params.append(module.bias)
                assigned.add(id(module.bias))
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in assigned:
                continue
            if getattr(parameter, "_force_weight_decay", False):
                adam_decay_params.append(parameter)
            elif parameter.ndim < 2 or "norm" in module_name.lower() or "norm" in parameter_name.lower() or parameter_name.endswith("bias"):
                adam_no_decay_params.append(parameter)
            else:
                adam_decay_params.append(parameter)
            assigned.add(id(parameter))
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in assigned:
            if getattr(parameter, "_force_weight_decay", False):
                adam_decay_params.append(parameter)
            elif parameter.ndim < 2:
                adam_no_decay_params.append(parameter)
            else:
                adam_decay_params.append(parameter)
            assigned.add(id(parameter))
    return muon_params, adam_decay_params, adam_no_decay_params


def _build_optimizer(*, model, training_config: TrainingConfig, device, torch):
    if training_config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_epsilon,
            weight_decay=training_config.weight_decay,
            fused=(device.type == "cuda"),
        )
    muon_params, adam_decay_params, adam_no_decay_params = _partition_params_for_muon(model, torch)
    param_groups = []
    if adam_decay_params:
        param_groups.append(
            {
                "params": adam_decay_params,
                "lr": training_config.muon_aux_learning_rate,
                "betas": (training_config.adam_beta1, training_config.adam_beta2),
                "eps": training_config.adam_epsilon,
                "weight_decay": training_config.muon_aux_weight_decay,
                "use_muon": False,
            }
        )
    if adam_no_decay_params:
        param_groups.append(
            {
                "params": adam_no_decay_params,
                "lr": training_config.muon_aux_learning_rate,
                "betas": (training_config.adam_beta1, training_config.adam_beta2),
                "eps": training_config.adam_epsilon,
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "lr": training_config.learning_rate,
                "momentum": training_config.muon_momentum,
                "weight_decay": training_config.weight_decay,
                "nesterov": training_config.muon_nesterov,
                "ns_steps": training_config.muon_ns_steps,
                "use_muon_plus": training_config.use_muon_plus,
                "norm_eps": training_config.muonplus_norm_eps,
                "use_muon": True,
            }
        )
    if not param_groups:
        raise ValueError("no trainable parameters found for optimizer construction")
    return SingleDeviceMuonWithAuxAdam(param_groups)


def _compute_causal_lm_loss(*, logits, labels, label_smoothing: float, torch):
    if label_smoothing <= 0.0:
        return None
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
        label_smoothing=label_smoothing,
    )


def _count_optimizer_steps_for_epoch(*, train_dataset, training_config: TrainingConfig, epoch_index: int) -> int:
    sampler = _build_batch_sampler(
        train_dataset,
        training_config=training_config,
        max_tokens_per_batch=training_config.max_tokens_per_batch,
        shuffle=True,
        seed=training_config.seed + epoch_index,
    )
    return math.ceil(len(sampler) / training_config.gradient_accumulation_steps)


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step_{step:07d}"


def _latest_checkpoint_dir(output_dir: Path) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    checkpoints = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    if not checkpoints:
        return None
    return checkpoints[-1]


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
    model_config: TinyGPT2Config | TinyQwen3Config,
    runtime_state: dict | None = None,
    wandb_run_id: str | None = None,
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
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "runtime_state": runtime_state or {},
                "wandb_run_id": wandb_run_id,
            },
            ckpt_dir / "trainer_state.pt",
        )
    _trim_checkpoints(output_dir, config.keep_last_checkpoints)


def _load_trainer_state(checkpoint_dir: Path, torch) -> dict:
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.pt"
    if not trainer_state_path.is_file():
        raise FileNotFoundError(f"trainer_state.pt not found under {checkpoint_dir}")
    return torch.load(trainer_state_path, map_location="cpu")


def _restore_rng_state(*, trainer_state: dict, torch, device) -> None:
    rng_state = trainer_state.get("rng_state")
    if rng_state is not None:
        torch.random.set_rng_state(rng_state)
    python_random_state = trainer_state.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)
    numpy_random_state = trainer_state.get("numpy_random_state")
    if numpy_random_state is not None:
        np.random.set_state(numpy_random_state)
    cuda_rng_state = trainer_state.get("cuda_rng_state")
    if cuda_rng_state is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state)


def _dataset_seed_index(*, epoch_index: int, dataset_index: int, dataset_count: int) -> int:
    return epoch_index * max(1, dataset_count) + dataset_index


def _count_batches_for_dataset_epoch(*, train_dataset, training_config: TrainingConfig, epoch_index: int) -> int:
    sampler = _build_batch_sampler(
        train_dataset,
        training_config=training_config,
        max_tokens_per_batch=training_config.max_tokens_per_batch,
        shuffle=True,
        seed=training_config.seed + epoch_index,
    )
    return len(sampler)


def _infer_resume_runtime_state(
    *,
    trainer_state: dict,
    train_dataset_plan,
    training_config: TrainingConfig,
) -> dict:
    runtime_state = dict(trainer_state.get("runtime_state") or {})
    if runtime_state.get("dataset_index") is not None:
        return runtime_state

    global_step = int(trainer_state.get("step", 0))
    dataset_count = max(1, len(train_dataset_plan))
    remaining_steps = global_step
    epoch_index = 0

    while True:
        for dataset_index, (_dataset_dir, train_dataset) in enumerate(train_dataset_plan):
            seed_index = _dataset_seed_index(
                epoch_index=epoch_index,
                dataset_index=dataset_index,
                dataset_count=dataset_count,
            )
            batch_count = _count_batches_for_dataset_epoch(
                train_dataset=train_dataset,
                training_config=training_config,
                epoch_index=seed_index,
            )
            optimizer_steps = math.ceil(batch_count / training_config.gradient_accumulation_steps)
            if remaining_steps >= optimizer_steps:
                remaining_steps -= optimizer_steps
                continue
            batches_seen_in_dataset = min(
                batch_count,
                remaining_steps * training_config.gradient_accumulation_steps,
            )
            return {
                "epoch": epoch_index,
                "dataset_index": dataset_index,
                "batches_seen_in_dataset": batches_seen_in_dataset,
                "accumulation_step": 0,
                "accum_segment_count": 0,
                "accum_packed_row_count": 0,
                "accum_packed_tokens": 0,
                "accum_padded_tokens": 0,
                "latest_train_loss_value": 0.0,
                "best_early_stopping_metric": None,
                "non_improving_evals": 0,
            }
        epoch_index += 1
        if training_config.train_steps == 0 and epoch_index >= training_config.train_epochs:
            return {
                "epoch": max(0, training_config.train_epochs - 1),
                "dataset_index": max(0, dataset_count - 1),
                "batches_seen_in_dataset": 0,
                "accumulation_step": 0,
                "accum_segment_count": 0,
                "accum_packed_row_count": 0,
                "accum_packed_tokens": 0,
                "accum_padded_tokens": 0,
                "latest_train_loss_value": 0.0,
                "best_early_stopping_metric": None,
                "non_improving_evals": 0,
            }


def _evaluate(model, dataloader, device, autocast_dtype, torch) -> dict[str, float]:
    return _evaluate_over_dataloaders(model, [dataloader], device, autocast_dtype, torch)


def _evaluate_over_dataloaders(model, dataloaders, device, autocast_dtype, torch) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    packed_tokens = 0
    padded_tokens = 0
    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:
                assert isinstance(batch, (PackedBatch, UnpackedBatch))
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    outputs = model(
                        input_ids=batch.input_ids.to(device),
                        attention_mask=batch.attention_mask.to(device),
                        labels=batch.labels.to(device),
                    )
                # Eval metrics should stay comparable across runs, so always use raw causal LM loss here.
                loss = outputs.loss
                losses.append(float(loss.detach().cpu()))
                packed_tokens += batch.stats.packed_tokens
                padded_tokens += batch.stats.padded_tokens
                del outputs, loss, batch
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "eval/loss": mean_loss,
        "eval/perplexity": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
        "eval/packing_efficiency": packed_tokens / padded_tokens if padded_tokens else 0.0,
    }


def _build_eval_loader(*, eval_dataset, training_config: TrainingConfig, collator, device, seed: int, DataLoader):
    return DataLoader(
        eval_dataset,
        batch_sampler=_build_batch_sampler(
            eval_dataset,
            training_config=training_config,
            max_tokens_per_batch=training_config.eval_max_tokens_per_batch,
            shuffle=False,
            seed=seed,
        ),
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )


def _build_batch_sampler(dataset, *, training_config: TrainingConfig, max_tokens_per_batch: int, shuffle: bool, seed: int):
    if training_config.packing_mode == "packed":
        return build_group_batch_sampler(
            dataset,
            max_tokens_per_batch=max_tokens_per_batch,
            shuffle=shuffle,
            seed=seed,
        )
    batch_size = (
        training_config.per_device_train_batch_size
        if shuffle
        else training_config.per_device_eval_batch_size
    )
    return build_fixed_batch_size_sampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def _estimate_effective_tokens_per_step(training_config: TrainingConfig) -> int:
    if training_config.packing_mode == "packed":
        return training_config.max_tokens_per_batch * training_config.gradient_accumulation_steps
    return (
        training_config.per_device_train_batch_size
        * training_config.max_seq_length
        * training_config.gradient_accumulation_steps
    )


def _release_cuda_memory(torch, device) -> None:
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            return


def _stop_requested(training_config: TrainingConfig) -> bool:
    return training_config.stop_file is not None and training_config.stop_file.exists()


@dataclass
class _StepTransition:
    global_step: int
    latest_metrics: dict[str, float]
    should_run_eval: bool
    should_save_checkpoint: bool


def _run_train_batch(
    *,
    batch,
    model,
    device,
    autocast_dtype,
    torch,
    training_config: TrainingConfig,
    optimizer,
    scheduler,
    wandb,
    start_time: float,
    global_step: int,
    target_train_steps: int,
    accumulation_step: int,
    accum_segment_count: int,
    accum_packed_row_count: int,
    accum_packed_tokens: int,
    accum_padded_tokens: int,
) -> tuple[_StepTransition | None, int, int, int, int, int, int, float]:
    assert isinstance(batch, (PackedBatch, UnpackedBatch))
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    labels = batch.labels.to(device)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        smoothed_loss = _compute_causal_lm_loss(
            logits=outputs.logits,
            labels=labels,
            label_smoothing=training_config.label_smoothing,
            torch=torch,
        )
        raw_loss = outputs.loss if smoothed_loss is None else smoothed_loss
        loss = raw_loss / training_config.gradient_accumulation_steps
    raw_loss_value = float(raw_loss.detach().cpu())
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step + 1}: {raw_loss_value}")
    loss.backward()
    accumulation_step += 1
    accum_segment_count += batch.stats.segment_count
    accum_packed_row_count += batch.stats.packed_row_count
    accum_packed_tokens += batch.stats.packed_tokens
    accum_padded_tokens += batch.stats.padded_tokens
    transition = None
    if accumulation_step % training_config.gradient_accumulation_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        latest_metrics, should_run_eval, should_save_checkpoint = _finalize_optimizer_step(
            global_step=global_step,
            target_train_steps=target_train_steps,
            latest_metrics={},
            train_loss_value=raw_loss_value,
            step_stats=PackedGroupStats(
                segment_count=accum_segment_count,
                packed_row_count=accum_packed_row_count,
                packed_tokens=accum_packed_tokens,
                padded_tokens=accum_padded_tokens,
            ),
            grad_norm=grad_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            wandb=wandb,
            start_time=start_time,
        )
        transition = _StepTransition(
            global_step=global_step,
            latest_metrics=latest_metrics,
            should_run_eval=should_run_eval,
            should_save_checkpoint=should_save_checkpoint,
        )
        accumulation_step = 0
        accum_segment_count = 0
        accum_packed_row_count = 0
        accum_packed_tokens = 0
        accum_padded_tokens = 0
    del input_ids, attention_mask, labels, outputs, smoothed_loss, raw_loss, loss, batch
    return (
        transition,
        global_step,
        accumulation_step,
        accum_segment_count,
        accum_packed_row_count,
        accum_packed_tokens,
        accum_padded_tokens,
        raw_loss_value,
    )


def _flush_partial_optimizer_step(
    *,
    model,
    optimizer,
    scheduler,
    training_config: TrainingConfig,
    torch,
    wandb,
    start_time: float,
    global_step: int,
    target_train_steps: int,
    latest_train_loss_value: float,
    accum_segment_count: int,
    accum_packed_row_count: int,
    accum_packed_tokens: int,
    accum_padded_tokens: int,
) -> _StepTransition:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    global_step += 1
    latest_metrics, should_run_eval, should_save_checkpoint = _finalize_optimizer_step(
        global_step=global_step,
        target_train_steps=target_train_steps,
        latest_metrics={},
        train_loss_value=latest_train_loss_value,
        step_stats=PackedGroupStats(
            segment_count=accum_segment_count,
            packed_row_count=accum_packed_row_count,
            packed_tokens=accum_packed_tokens,
            padded_tokens=accum_padded_tokens,
        ),
        grad_norm=grad_norm,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        wandb=wandb,
        start_time=start_time,
    )
    return _StepTransition(
        global_step=global_step,
        latest_metrics=latest_metrics,
        should_run_eval=should_run_eval,
        should_save_checkpoint=should_save_checkpoint,
    )


def _run_eval_if_needed(
    *,
    global_step: int,
    target_train_steps: int,
    latest_metrics: dict[str, float],
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    training_config: TrainingConfig,
    model_config: TinyGPT2Config | TinyQwen3Config,
    split_metadata: dict,
    eval_dataset,
    collator,
    DataLoader,
    device,
    autocast_dtype,
    torch,
    wandb,
) -> tuple[dict[str, float], bool]:
    if eval_dataset is None or (
        global_step % training_config.eval_interval != 0 and global_step != target_train_steps
    ):
        return latest_metrics, False

    saved_checkpoint_this_step = False
    if training_config.use_separate_eval_process and split_metadata.get("eval_dataset_dirs"):
        checkpoint_step_dir = _checkpoint_dir(output_dir, global_step)
        _save_checkpoint(
            output_dir=output_dir,
            step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            model_config=model_config,
        )
        saved_checkpoint_this_step = True
        if device.type == "cuda":
            _release_cuda_memory(torch, device)
        eval_metrics = _evaluate_via_subprocess(
            checkpoint_model_dir=checkpoint_step_dir / "model",
            training_config=training_config,
            eval_dataset_dirs=tuple(Path(path) for path in split_metadata["eval_dataset_dirs"]),
        )
    else:
        _release_cuda_memory(torch, device)
        if isinstance(eval_dataset, list):
            eval_loaders = [
                _build_eval_loader(
                    eval_dataset=dataset,
                    training_config=training_config,
                    collator=collator,
                    device=device,
                    seed=training_config.seed + dataset_idx,
                    DataLoader=DataLoader,
                )
                for dataset_idx, dataset in enumerate(eval_dataset)
            ]
            eval_metrics = _evaluate_over_dataloaders(model, eval_loaders, device, autocast_dtype, torch)
            del eval_loaders
        else:
            eval_loader = _build_eval_loader(
                eval_dataset=eval_dataset,
                training_config=training_config,
                collator=collator,
                device=device,
                seed=training_config.seed,
                DataLoader=DataLoader,
            )
            eval_metrics = _evaluate(model, eval_loader, device, autocast_dtype, torch)
            del eval_loader
        _release_cuda_memory(torch, device)

    latest_metrics.update(eval_metrics)
    _wandb_log_if_available(wandb, eval_metrics, step=global_step)
    print(json.dumps({"step": global_step, **eval_metrics}, ensure_ascii=False))
    return latest_metrics, saved_checkpoint_this_step


def _update_early_stopping(
    *,
    training_config: TrainingConfig,
    latest_metrics: dict[str, float],
    best_metric: float | None,
    non_improving_evals: int,
    global_step: int,
    wandb,
) -> tuple[float | None, int, bool]:
    metric_name = training_config.early_stopping_metric
    patience = training_config.early_stopping_patience
    if not metric_name or patience <= 0:
        return best_metric, non_improving_evals, False
    if metric_name not in latest_metrics:
        return best_metric, non_improving_evals, False

    metric_value = float(latest_metrics[metric_name])
    improved = best_metric is None or metric_value < best_metric
    if improved:
        best_metric = metric_value
        non_improving_evals = 0
    else:
        non_improving_evals += 1

    _wandb_summary_update_if_available(
        wandb,
        {
            "early_stopping/metric": metric_name,
            "early_stopping/best": best_metric,
            "early_stopping/non_improving_evals": non_improving_evals,
            "early_stopping/patience": patience,
        },
    )

    should_stop = non_improving_evals >= patience
    if should_stop:
        print(
            json.dumps(
                {
                    "status": "early_stopping_triggered",
                    "step": global_step,
                    "metric": metric_name,
                    "best": best_metric,
                    "current": metric_value,
                    "non_improving_evals": non_improving_evals,
                    "patience": patience,
                },
                ensure_ascii=False,
            )
        )
    return best_metric, non_improving_evals, should_stop


def _finalize_optimizer_step(
    *,
    global_step: int,
    target_train_steps: int,
    latest_metrics: dict[str, float],
    train_loss_value: float,
    step_stats: PackedGroupStats,
    grad_norm,
    training_config: TrainingConfig,
    optimizer,
    scheduler,
    wandb,
    start_time: float,
) -> tuple[dict[str, float], bool, bool]:
    latest_metrics = {
        "train/loss": float(train_loss_value),
        "train/grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
        "train/packing_efficiency": step_stats.packing_efficiency,
        "train/packed_rows": float(step_stats.packed_row_count),
        "train/segments": float(step_stats.segment_count),
        "train/tokens_per_step": float(step_stats.packed_tokens),
        "train/elapsed_sec": time.time() - start_time,
    }
    latest_metrics.update(
        _collect_optimizer_lrs(
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
        )
    )
    _wandb_log_if_available(wandb, latest_metrics, step=global_step)
    if global_step % training_config.log_interval == 0 or global_step == target_train_steps:
        print(json.dumps({"step": global_step, **latest_metrics}, ensure_ascii=False))

    should_run_eval = global_step % training_config.eval_interval == 0 or global_step == target_train_steps
    should_save_checkpoint = global_step % training_config.save_interval == 0
    return latest_metrics, should_run_eval, should_save_checkpoint


def _evaluate_via_subprocess(
    *,
    checkpoint_model_dir: Path,
    training_config: TrainingConfig,
    eval_dataset_dirs: tuple[Path, ...],
) -> dict[str, float]:
    eval_attn_implementation = training_config.attn_implementation
    if training_config.eval_device == "cpu" and training_config.attn_implementation == "flash_attention_2":
        eval_attn_implementation = "sdpa"
    command = [
        sys.executable,
        "-u",
        str(Path("scripts") / "eval_gpt2.py"),
        "--model-dir",
        str(checkpoint_model_dir),
        "--tokenizer-dir",
        str(training_config.tokenizer_dir),
        "--packing-mode",
        training_config.packing_mode,
        "--attn-implementation",
        eval_attn_implementation,
        "--max-seq-length",
        str(training_config.max_seq_length),
        "--pad-to-multiple-of",
        str(training_config.pad_to_multiple_of),
        "--eval-max-tokens-per-batch",
        str(training_config.eval_max_tokens_per_batch),
        "--per-device-eval-batch-size",
        str(training_config.per_device_eval_batch_size),
        "--max-eval-groups",
        str(training_config.max_eval_groups),
        "--eval-device",
        training_config.eval_device,
    ]
    for dataset_dir in eval_dataset_dirs:
        command.extend(["--dataset-dir", str(dataset_dir)])
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "eval subprocess failed with "
            f"returncode={exc.returncode}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc
    stdout = completed.stdout.strip().splitlines()
    if not stdout:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"eval subprocess returned no output. stderr={stderr}")
    return json.loads(stdout[-1])


def _prepare_train_eval_datasets(training_config: TrainingConfig):
    train_dataset = load_grouped_dataset(training_config.dataset_dirs)
    limited_before_split = False
    used_cached_metadata = False
    if not training_config.eval_dataset_dirs and (training_config.max_train_groups > 0 or training_config.max_eval_groups > 0):
        # Smoke and ablation runs should trim the source dataset before any expensive split/cache work.
        train_dataset = limit_groups(
            train_dataset,
            training_config.max_train_groups + max(1, training_config.max_eval_groups),
        )
        limited_before_split = True

    if training_config.validate_dataset:
        validate_grouped_dataset(train_dataset)

    split_metadata: dict[str, int | float | str] = {
        "cache_hit": "none",
        "train_rows": len(train_dataset),
        "train_groups": count_unique_groups(train_dataset),
        "eval_rows": 0,
        "eval_groups": 0,
    }
    if training_config.eval_dataset_dirs:
        eval_dataset = load_grouped_dataset(training_config.eval_dataset_dirs)
        split_metadata["eval_dataset_dirs"] = [str(path) for path in training_config.eval_dataset_dirs]
        if training_config.max_eval_groups > 0:
            eval_dataset = limit_groups(eval_dataset, training_config.max_eval_groups)
        if training_config.validate_dataset:
            validate_grouped_dataset(eval_dataset)
        split_metadata.update(
            {
                "eval_rows": len(eval_dataset),
                "eval_groups": count_unique_groups(eval_dataset),
            }
        )
    else:
        if training_config.train_split_eval_ratio == 0.0:
            train_dataset = limit_groups(train_dataset, training_config.max_train_groups)
            if training_config.validate_dataset:
                validate_grouped_dataset(train_dataset)
            split_metadata.update(
                {
                    "train_rows": len(train_dataset),
                    "train_groups": count_unique_groups(train_dataset),
                }
            )
            return train_dataset, None, split_metadata
        cached_split = None
        if not limited_before_split and training_config.max_train_groups <= 0 and training_config.max_eval_groups <= 0:
            cached_split = load_cached_split(
                training_config.dataset_dirs,
                eval_ratio=training_config.train_split_eval_ratio,
                seed=training_config.split_seed,
                cache_dir=training_config.cache_dir,
            )
        if cached_split is not None:
            train_dataset, eval_dataset, cached_metadata = cached_split
            _train_cache_path, eval_cache_path, _metadata_path = build_split_cache_paths(
                training_config.dataset_dirs,
                eval_ratio=training_config.train_split_eval_ratio,
                seed=training_config.split_seed,
                cache_dir=training_config.cache_dir,
            )
            split_metadata = {
                "cache_hit": "split",
                "eval_dataset_dirs": [str(eval_cache_path)],
                **cached_metadata,
            }
            used_cached_metadata = True
        else:
            train_dataset, eval_dataset = split_grouped_dataset(
                train_dataset,
                eval_ratio=training_config.train_split_eval_ratio,
                seed=training_config.split_seed,
            )
            if limited_before_split:
                split_metadata = {
                    "cache_hit": "limited",
                    "eval_dataset_dirs": [],
                    "train_rows": len(train_dataset),
                    "eval_rows": len(eval_dataset),
                    "train_groups": count_unique_groups(train_dataset),
                    "eval_groups": count_unique_groups(eval_dataset),
                }
            else:
                split_metadata = {
                    "cache_hit": "miss",
                    "eval_dataset_dirs": [
                        str(
                            build_split_cache_paths(
                                training_config.dataset_dirs,
                                eval_ratio=training_config.train_split_eval_ratio,
                                seed=training_config.split_seed,
                                cache_dir=training_config.cache_dir,
                            )[1]
                        )
                    ],
                    **save_split_cache(
                        train_dataset,
                        eval_dataset,
                        dataset_dirs=training_config.dataset_dirs,
                        eval_ratio=training_config.train_split_eval_ratio,
                        seed=training_config.split_seed,
                        cache_dir=training_config.cache_dir,
                    ),
                }
    train_dataset = limit_groups(train_dataset, training_config.max_train_groups)
    eval_dataset = limit_groups(eval_dataset, training_config.max_eval_groups) if eval_dataset is not None else None
    if training_config.validate_dataset:
        validate_grouped_dataset(train_dataset)
        if eval_dataset is not None:
            validate_grouped_dataset(eval_dataset)
    if limited_before_split or training_config.max_train_groups > 0 or training_config.max_eval_groups > 0 or not used_cached_metadata:
        split_metadata.update(
            {
                "train_rows": len(train_dataset),
                "train_groups": count_unique_groups(train_dataset),
                "eval_rows": 0 if eval_dataset is None else len(eval_dataset),
                "eval_groups": 0 if eval_dataset is None else count_unique_groups(eval_dataset),
            }
        )
    return train_dataset, eval_dataset, split_metadata


def _prepare_yearly_train_eval_datasets(training_config: TrainingConfig):
    yearly_train_datasets: list[tuple[Path, object]] = []
    yearly_eval_datasets: list[tuple[Path, object]] = []
    aggregate_metadata: dict[str, int | float | str | list[str]] = {
        "cache_hit": "per_year",
        "train_rows": 0,
        "train_groups": 0,
        "eval_rows": 0,
        "eval_groups": 0,
        "eval_dataset_dirs": [],
    }
    for dataset_dir in training_config.dataset_dirs:
        yearly_config = replace(
            training_config,
            dataset_dirs=(dataset_dir,),
            eval_dataset_dirs=(),
        )
        train_dataset, eval_dataset, split_metadata = _prepare_train_eval_datasets(yearly_config)
        yearly_train_datasets.append((dataset_dir, train_dataset))
        if eval_dataset is not None:
            yearly_eval_datasets.append((dataset_dir, eval_dataset))
        aggregate_metadata["train_rows"] += int(split_metadata["train_rows"])
        aggregate_metadata["train_groups"] += int(split_metadata["train_groups"])
        aggregate_metadata["eval_rows"] += int(split_metadata["eval_rows"])
        aggregate_metadata["eval_groups"] += int(split_metadata["eval_groups"])
        aggregate_metadata["eval_dataset_dirs"].extend(split_metadata.get("eval_dataset_dirs", []))
    return yearly_train_datasets, yearly_eval_datasets, aggregate_metadata


def train(
    *,
    model_config: TinyGPT2Config | TinyQwen3Config,
    training_config: TrainingConfig,
) -> dict[str, float]:
    torch, DataLoader, _unused_scheduler = _require_training_deps()
    model_config.validate()
    training_config.validate()
    model_context = (
        model_config.n_positions
        if isinstance(model_config, TinyGPT2Config)
        else model_config.max_position_embeddings
    )
    if training_config.max_seq_length > model_context:
        raise ValueError(
            f"max_seq_length={training_config.max_seq_length} exceeds model context limit={model_context}"
        )
    if training_config.packing_mode == "packed" and training_config.attn_implementation in {
        "flash_attention_2",
        "flash_attention_3",
    }:
        raise ValueError(
            f"{training_config.attn_implementation} is not compatible with the current PackedGroupCollator attention mask. "
            "The packed training path builds a dense block-diagonal attention mask, but FlashAttention 2 "
            "expects a different varlen/unpadded attention layout. Use sdpa for the MahjongLM packed training path, "
            "or switch to --packing-mode unpadded for a non-packed experiment."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("src/gpt2/train.py currently supports single-process training only")
    if training_config.use_separate_eval_process and sys.platform.startswith("win") and training_config.eval_device == "cuda":
        raise ValueError(
            "use_separate_eval_process with eval_device=cuda is disabled on Windows because CUDA + subprocess eval "
            "has been unstable in this workspace. Use eval_device=cpu for subprocess eval on Windows."
        )
    _set_global_seed(training_config.seed)
    if training_config.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resume_checkpoint_dir = None
    if training_config.resume_from_checkpoint is not None:
        resume_checkpoint_dir = training_config.resume_from_checkpoint
    elif training_config.resume_latest_checkpoint:
        resume_checkpoint_dir = _latest_checkpoint_dir(training_config.output_dir)
    if resume_checkpoint_dir is not None:
        resume_checkpoint_dir = Path(resume_checkpoint_dir)

    resume_trainer_state = None
    resume_run_id = training_config.wandb_resume_run_id
    if resume_checkpoint_dir is not None:
        resume_trainer_state = _load_trainer_state(resume_checkpoint_dir, torch)
        if resume_run_id is None:
            resume_run_id = resume_trainer_state.get("wandb_run_id")

    early_run_metadata = {
        "device": "pending",
        "train_group_count": "pending",
        "eval_group_count": "pending",
        "parameter_count": "pending",
        "attn_implementation": training_config.attn_implementation,
        "packing_mode": training_config.packing_mode,
    }
    wandb = _init_wandb(training_config, model_config, early_run_metadata, resume_run_id=resume_run_id)
    _wandb_summary_update_if_available(wandb, {"status/stage": "preparing_dataset"})

    tokenizer = load_hf_tokenizer(training_config.tokenizer_dir)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None or tokenizer.bos_token_id is None:
        raise ValueError("tokenizer must define bos/eos/pad token ids")

    if len(training_config.dataset_dirs) > 1 and not training_config.eval_dataset_dirs:
        yearly_train_datasets, yearly_eval_datasets, split_metadata = _prepare_yearly_train_eval_datasets(training_config)
        train_dataset_plan = yearly_train_datasets
        eval_dataset = [dataset for _dataset_dir, dataset in yearly_eval_datasets] if yearly_eval_datasets else None
    else:
        train_dataset, eval_dataset, split_metadata = _prepare_train_eval_datasets(training_config)
        train_dataset_plan = [(training_config.dataset_dirs[0], train_dataset)]
    _wandb_summary_update_if_available(
        wandb,
        {
            "status/stage": "dataset_ready",
            "dataset/train_group_count": split_metadata["train_groups"],
            "dataset/eval_group_count": split_metadata["eval_groups"],
            "dataset/train_rows": split_metadata["train_rows"],
            "dataset/eval_rows": split_metadata["eval_rows"],
            "dataset/cache_hit": split_metadata["cache_hit"],
        },
    )

    device = _select_device(torch)
    autocast_dtype = _resolve_dtype(torch, training_config, device)
    if resume_checkpoint_dir is not None:
        model = load_saved_causal_lm(
            model_dir=resume_checkpoint_dir / "model",
            attn_implementation=training_config.attn_implementation,
            dtype=autocast_dtype,
        )
    elif training_config.model_family == "gpt2":
        assert isinstance(model_config, TinyGPT2Config)
        model = build_tiny_gpt2_model(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=training_config.attn_implementation,
            dtype=autocast_dtype,
            config=model_config,
        )
    elif training_config.model_family == "qwen3":
        assert isinstance(model_config, TinyQwen3Config)
        model = build_tiny_qwen3_model(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=training_config.attn_implementation,
            dtype=autocast_dtype,
            config=model_config,
        )
    else:
        raise ValueError(f"unsupported model_family: {training_config.model_family}")
    if training_config.gradient_checkpointing:
        gradient_checkpointing_kwargs = None
        if training_config.model_family == "qwen3" and getattr(model_config, "use_mamba3_hybrid", False):
            # Official Mamba-3 kernels are not compatible with the default
            # non-reentrant HF checkpointing path in our stack.
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    if training_config.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.to(device)
    model._mahjonglm_label_smoothing = training_config.label_smoothing

    optimizer = _build_optimizer(
        model=model,
        training_config=training_config,
        device=device,
        torch=torch,
    )
    if training_config.packing_mode == "packed":
        collator = PackedGroupCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=training_config.max_seq_length,
            pad_to_multiple_of=training_config.pad_to_multiple_of,
            return_tensors="pt",
        )
    else:
        collator = UnpackedCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=training_config.max_seq_length,
            pad_to_multiple_of=training_config.pad_to_multiple_of,
            return_tensors="pt",
        )

    batches_per_epoch = sum(
        len(
            _build_batch_sampler(
                dataset,
                training_config=training_config,
                max_tokens_per_batch=training_config.max_tokens_per_batch,
                shuffle=True,
                seed=training_config.seed + dataset_index,
            )
        )
        for dataset_index, (_dataset_dir, dataset) in enumerate(train_dataset_plan)
    )
    if training_config.train_steps > 0:
        optimizer_steps_per_epoch = sum(
            _count_optimizer_steps_for_epoch(
                train_dataset=dataset,
                training_config=training_config,
                epoch_index=dataset_index,
            )
            for dataset_index, (_dataset_dir, dataset) in enumerate(train_dataset_plan)
        )
        target_train_steps = training_config.train_steps
    else:
        per_epoch_optimizer_steps = [
            sum(
                _count_optimizer_steps_for_epoch(
                    train_dataset=dataset,
                    training_config=training_config,
                    epoch_index=epoch_index * max(1, len(train_dataset_plan)) + dataset_index,
                )
                for dataset_index, (_dataset_dir, dataset) in enumerate(train_dataset_plan)
            )
            for epoch_index in range(training_config.train_epochs)
        ]
        optimizer_steps_per_epoch = per_epoch_optimizer_steps[0] if per_epoch_optimizer_steps else 0
        target_train_steps = sum(per_epoch_optimizer_steps)
    effective_tokens_per_step = _estimate_effective_tokens_per_step(training_config)
    resolved_warmup_steps = training_config.warmup_steps
    if training_config.lr_scheduler_type == "linear":
        scheduler = _build_linear_scheduler(
            optimizer,
            warmup_steps=resolved_warmup_steps,
            total_steps=target_train_steps,
            torch=torch,
        )
    else:
        scheduler = _build_cosine_scheduler_with_floor(
            optimizer,
            warmup_steps=resolved_warmup_steps,
            total_steps=target_train_steps,
            min_lr_ratio=training_config.min_learning_rate_ratio,
            torch=torch,
        )

    latest_metrics: dict[str, float] = {}

    run_metadata = {
        "train_group_count": split_metadata["train_groups"],
        "eval_group_count": split_metadata["eval_groups"],
        "parameter_count": count_parameters(model),
        "device": str(device),
        "attn_implementation": training_config.attn_implementation,
        "packing_mode": training_config.packing_mode,
        "model_family": training_config.model_family,
    }
    _wandb_summary_update_if_available(
        wandb,
        {
            "status/stage": "training",
            "run/train_group_count": run_metadata["train_group_count"],
            "run/eval_group_count": run_metadata["eval_group_count"],
            "run/parameter_count": run_metadata["parameter_count"],
            "run/device": run_metadata["device"],
            "run/model_family": run_metadata["model_family"],
            "run/packing_mode": run_metadata["packing_mode"],
            "run/optimizer_name": training_config.optimizer_name,
            "run/batches_per_epoch": batches_per_epoch,
            "run/optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "run/target_train_steps": target_train_steps,
            "run/effective_tokens_per_step": effective_tokens_per_step,
            "run/resolved_warmup_steps": resolved_warmup_steps,
        },
    )

    output_dir = training_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "training_config.json", json.loads(training_config.to_json()))
    _save_json(output_dir / "model_config.json", asdict(model_config))

    global_step = 0
    accumulation_step = 0
    accum_segment_count = 0
    accum_packed_row_count = 0
    accum_packed_tokens = 0
    accum_padded_tokens = 0
    latest_train_loss_value = 0.0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    interrupted = False
    early_stopped = False
    failed_exception: Exception | None = None
    best_early_stopping_metric: float | None = None
    non_improving_evals = 0
    resume_epoch = 0
    resume_dataset_index = 0
    resume_batches_seen_in_dataset = 0
    current_runtime_state = {
        "epoch": 0,
        "dataset_index": 0,
        "batches_seen_in_dataset": 0,
        "accumulation_step": 0,
        "accum_segment_count": 0,
        "accum_packed_row_count": 0,
        "accum_packed_tokens": 0,
        "accum_padded_tokens": 0,
        "latest_train_loss_value": 0.0,
        "best_early_stopping_metric": None,
        "non_improving_evals": 0,
    }

    if resume_trainer_state is not None:
        optimizer.load_state_dict(resume_trainer_state["optimizer"])
        scheduler.load_state_dict(resume_trainer_state["scheduler"])
        global_step = int(resume_trainer_state.get("step", 0))
        _restore_rng_state(trainer_state=resume_trainer_state, torch=torch, device=device)
        inferred_runtime_state = _infer_resume_runtime_state(
            trainer_state=resume_trainer_state,
            train_dataset_plan=train_dataset_plan,
            training_config=training_config,
        )
        resume_epoch = int(inferred_runtime_state.get("epoch", 0))
        resume_dataset_index = int(inferred_runtime_state.get("dataset_index", 0))
        resume_batches_seen_in_dataset = int(inferred_runtime_state.get("batches_seen_in_dataset", 0))
        accumulation_step = int(inferred_runtime_state.get("accumulation_step", 0))
        accum_segment_count = int(inferred_runtime_state.get("accum_segment_count", 0))
        accum_packed_row_count = int(inferred_runtime_state.get("accum_packed_row_count", 0))
        accum_packed_tokens = int(inferred_runtime_state.get("accum_packed_tokens", 0))
        accum_padded_tokens = int(inferred_runtime_state.get("accum_padded_tokens", 0))
        latest_train_loss_value = float(inferred_runtime_state.get("latest_train_loss_value", 0.0))
        best_early_stopping_metric = inferred_runtime_state.get("best_early_stopping_metric")
        non_improving_evals = int(inferred_runtime_state.get("non_improving_evals", 0))
        _wandb_summary_update_if_available(
            wandb,
            {
                "status/stage": "resumed_training",
                "resume/checkpoint_dir": str(resume_checkpoint_dir),
                "resume/global_step": global_step,
                "resume/epoch": resume_epoch,
                "resume/dataset_index": resume_dataset_index,
                "resume/batches_seen_in_dataset": resume_batches_seen_in_dataset,
            },
        )

    try:
        if training_config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if training_config.train_steps > 0:
            epoch_indices = range(10**9)
        else:
            epoch_indices = range(training_config.train_epochs)

        for epoch in epoch_indices:
            if global_step >= target_train_steps or early_stopped or _stop_requested(training_config):
                interrupted = interrupted or _stop_requested(training_config)
                break
            for dataset_index, (_dataset_dir, train_dataset) in enumerate(train_dataset_plan):
                if epoch < resume_epoch:
                    continue
                if epoch == resume_epoch and dataset_index < resume_dataset_index:
                    continue
                if global_step >= target_train_steps or early_stopped or _stop_requested(training_config):
                    interrupted = interrupted or _stop_requested(training_config)
                    break
                dataset_seed_index = _dataset_seed_index(
                    epoch_index=epoch,
                    dataset_index=dataset_index,
                    dataset_count=len(train_dataset_plan),
                )
                dataset_batch_skip = (
                    resume_batches_seen_in_dataset
                    if epoch == resume_epoch and dataset_index == resume_dataset_index
                    else 0
                )
                current_runtime_state = {
                    "epoch": epoch,
                    "dataset_index": dataset_index,
                    "batches_seen_in_dataset": dataset_batch_skip,
                    "accumulation_step": accumulation_step,
                    "accum_segment_count": accum_segment_count,
                    "accum_packed_row_count": accum_packed_row_count,
                    "accum_packed_tokens": accum_packed_tokens,
                    "accum_padded_tokens": accum_padded_tokens,
                    "latest_train_loss_value": latest_train_loss_value,
                    "best_early_stopping_metric": best_early_stopping_metric,
                    "non_improving_evals": non_improving_evals,
                }
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=_build_batch_sampler(
                        train_dataset,
                        training_config=training_config,
                        max_tokens_per_batch=training_config.max_tokens_per_batch,
                        shuffle=True,
                        seed=training_config.seed + dataset_seed_index,
                    ),
                    collate_fn=collator,
                    num_workers=training_config.dataloader_num_workers,
                    pin_memory=training_config.pin_memory and device.type == "cuda",
                )
                for batch_index, batch in enumerate(train_loader):
                    if batch_index < dataset_batch_skip:
                        continue
                    current_runtime_state["batches_seen_in_dataset"] = batch_index
                    if global_step >= target_train_steps or early_stopped or _stop_requested(training_config):
                        interrupted = interrupted or _stop_requested(training_config)
                        break
                    (
                        step_transition,
                        global_step,
                        accumulation_step,
                        accum_segment_count,
                        accum_packed_row_count,
                        accum_packed_tokens,
                        accum_padded_tokens,
                        latest_train_loss_value,
                    ) = _run_train_batch(
                        batch=batch,
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        torch=torch,
                        training_config=training_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        wandb=wandb,
                        start_time=start_time,
                        global_step=global_step,
                        target_train_steps=target_train_steps,
                        accumulation_step=accumulation_step,
                        accum_segment_count=accum_segment_count,
                        accum_packed_row_count=accum_packed_row_count,
                        accum_packed_tokens=accum_packed_tokens,
                        accum_padded_tokens=accum_padded_tokens,
                    )
                    current_runtime_state.update(
                        {
                            "batches_seen_in_dataset": batch_index + 1,
                            "accumulation_step": accumulation_step,
                            "accum_segment_count": accum_segment_count,
                            "accum_packed_row_count": accum_packed_row_count,
                            "accum_packed_tokens": accum_packed_tokens,
                            "accum_padded_tokens": accum_padded_tokens,
                            "latest_train_loss_value": latest_train_loss_value,
                            "best_early_stopping_metric": best_early_stopping_metric,
                            "non_improving_evals": non_improving_evals,
                        }
                    )
                    if step_transition is None:
                        continue
                    latest_metrics = step_transition.latest_metrics
                    saved_checkpoint_this_step = False
                    if eval_dataset is not None and step_transition.should_run_eval:
                        latest_metrics, saved_checkpoint_this_step = _run_eval_if_needed(
                            global_step=step_transition.global_step,
                            target_train_steps=target_train_steps,
                            latest_metrics=latest_metrics,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            output_dir=output_dir,
                            training_config=training_config,
                            model_config=model_config,
                            split_metadata=split_metadata,
                            eval_dataset=eval_dataset,
                            collator=collator,
                            DataLoader=DataLoader,
                            device=device,
                            autocast_dtype=autocast_dtype,
                            torch=torch,
                            wandb=wandb,
                        )
                        best_early_stopping_metric, non_improving_evals, early_stopped = _update_early_stopping(
                            training_config=training_config,
                            latest_metrics=latest_metrics,
                            best_metric=best_early_stopping_metric,
                            non_improving_evals=non_improving_evals,
                            global_step=step_transition.global_step,
                            wandb=wandb,
                        )
                        current_runtime_state["best_early_stopping_metric"] = best_early_stopping_metric
                        current_runtime_state["non_improving_evals"] = non_improving_evals
                    if step_transition.should_save_checkpoint and not saved_checkpoint_this_step:
                        _save_checkpoint(
                            output_dir=output_dir,
                            step=step_transition.global_step,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=training_config,
                            model_config=model_config,
                            runtime_state=current_runtime_state,
                            wandb_run_id=getattr(wandb, "id", None),
                        )
                    if early_stopped:
                        break
                del train_loader
                resume_batches_seen_in_dataset = 0
            if accumulation_step % training_config.gradient_accumulation_steps != 0 and global_step < target_train_steps and not interrupted:
                current_runtime_state = {
                    "epoch": epoch,
                    "dataset_index": dataset_index,
                    "batches_seen_in_dataset": current_runtime_state.get("batches_seen_in_dataset", 0),
                    "accumulation_step": accumulation_step,
                    "accum_segment_count": accum_segment_count,
                    "accum_packed_row_count": accum_packed_row_count,
                    "accum_packed_tokens": accum_packed_tokens,
                    "accum_padded_tokens": accum_padded_tokens,
                    "latest_train_loss_value": latest_train_loss_value,
                    "best_early_stopping_metric": best_early_stopping_metric,
                    "non_improving_evals": non_improving_evals,
                }
                step_transition = _flush_partial_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_config=training_config,
                    torch=torch,
                    wandb=wandb,
                    start_time=start_time,
                    global_step=global_step,
                    target_train_steps=target_train_steps,
                    latest_train_loss_value=latest_train_loss_value,
                    accum_segment_count=accum_segment_count,
                    accum_packed_row_count=accum_packed_row_count,
                    accum_packed_tokens=accum_packed_tokens,
                    accum_padded_tokens=accum_padded_tokens,
                )
                global_step = step_transition.global_step
                latest_metrics = step_transition.latest_metrics
                accumulation_step = 0
                accum_segment_count = 0
                accum_packed_row_count = 0
                accum_packed_tokens = 0
                accum_padded_tokens = 0
                saved_checkpoint_this_step = False
                if eval_dataset is not None and step_transition.should_run_eval:
                    latest_metrics, saved_checkpoint_this_step = _run_eval_if_needed(
                        global_step=step_transition.global_step,
                        target_train_steps=target_train_steps,
                        latest_metrics=latest_metrics,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        training_config=training_config,
                        model_config=model_config,
                        split_metadata=split_metadata,
                        eval_dataset=eval_dataset,
                        collator=collator,
                        DataLoader=DataLoader,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        torch=torch,
                        wandb=wandb,
                    )
                    best_early_stopping_metric, non_improving_evals, early_stopped = _update_early_stopping(
                        training_config=training_config,
                        latest_metrics=latest_metrics,
                        best_metric=best_early_stopping_metric,
                        non_improving_evals=non_improving_evals,
                        global_step=step_transition.global_step,
                        wandb=wandb,
                    )
                if step_transition.should_save_checkpoint and not saved_checkpoint_this_step:
                    _save_checkpoint(
                        output_dir=output_dir,
                        step=step_transition.global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=training_config,
                        model_config=model_config,
                        runtime_state=current_runtime_state,
                        wandb_run_id=getattr(wandb, "id", None),
                    )
                if early_stopped:
                    break
            accumulation_step = 0
            resume_dataset_index = 0
            if training_config.train_steps == 0:
                continue
    except KeyboardInterrupt:
        interrupted = True
        print(json.dumps({"status": "interrupt_requested", "step": global_step}, ensure_ascii=False))
    except Exception as exc:
        failed_exception = exc
        _wandb_summary_update_if_available(wandb, {"status/stage": "failed"})
        raise
    finally:
        if interrupted:
            _wandb_summary_update_if_available(wandb, {"status/stage": "stopped_by_request"})
        elif early_stopped:
            _wandb_summary_update_if_available(
                wandb,
                {
                    "status/stage": "early_stopped",
                    "early_stopping/best": best_early_stopping_metric,
                    "early_stopping/non_improving_evals": non_improving_evals,
                },
            )

        if failed_exception is None and not interrupted:
            final_dir = output_dir / "final_model"
            model.save_pretrained(str(final_dir))
            tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        else:
            print(
                json.dumps(
                    {
                        "status": "skip_final_save",
                        "reason": "failure" if failed_exception is not None else "interrupt",
                        "step": global_step,
                    },
                    ensure_ascii=False,
                )
            )

        if torch.cuda.is_available() and device.type == "cuda":
            try:
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                print(f"Peak CUDA memory allocated: {peak_allocated / (1024**3):.2f} GiB")
                print(f"Peak CUDA memory reserved: {peak_reserved / (1024**3):.2f} GiB")
                # Release PyTorch's cached CUDA blocks so smoke tests and repeated runs leave less residue.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as cleanup_exc:
                print(
                    json.dumps(
                        {
                            "status": "skip_cuda_cleanup",
                            "reason": type(cleanup_exc).__name__,
                        },
                        ensure_ascii=False,
                    )
                )

        if wandb is not None:
            wandb.finish(exit_code=0 if failed_exception is None and not interrupted else 1)
    return latest_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-2 or Qwen3 style model on MahjongLM.")
    parser.add_argument("--dataset-dir", action="append", type=Path, default=[])
    parser.add_argument("--eval-dataset-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpt2-mahjong-8192"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--model-family", type=str, default="gpt2", choices=["gpt2", "qwen3"])
    parser.add_argument(
        "--packing-mode",
        type=str,
        default="packed",
        choices=["packed", "unpadded"],
        help="Use packed group batches or plain per-row causal LM batches.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        help="Use sdpa for packed training. flash_attention_* is intended for --packing-mode unpadded.",
    )
    parser.add_argument("--arch", type=str, default="A", help="GPT-2 preset: A-D, or custom.")
    parser.add_argument("--qwen-arch", type=str, default="Q1", help="Qwen3 preset: Q0-Q4, or custom.")
    parser.add_argument("--n-layer", type=int, default=20)
    parser.add_argument("--n-head", type=int, default=10)
    parser.add_argument("--n-embd", type=int, default=640)
    parser.add_argument(
        "--n-inner",
        type=int,
        default=0,
        help="Set to 0 to use the GPT-2 default expansion factor.",
    )
    parser.add_argument("--n-positions", type=int, default=8192)
    parser.add_argument("--qwen-hidden-size", type=int, default=1024)
    parser.add_argument("--qwen-intermediate-size", type=int, default=3072)
    parser.add_argument("--qwen-num-hidden-layers", type=int, default=32)
    parser.add_argument("--qwen-num-attention-heads", type=int, default=16)
    parser.add_argument("--qwen-num-key-value-heads", type=int, default=8)
    parser.add_argument("--qwen-head-dim", type=int, default=128)
    parser.add_argument("--qwen-max-position-embeddings", type=int, default=8192)
    parser.add_argument("--use-exclusive-self-attention", action="store_true")
    parser.add_argument("--use-gated-attention", action="store_true")
    parser.add_argument("--use-zero-centered-rmsnorm", action="store_true")
    parser.add_argument("--use-rescaled-residual", action="store_true")
    parser.add_argument("--use-mamba3-hybrid", action="store_true")
    parser.add_argument("--mamba3-with-mlp-block", action="store_true")
    parser.add_argument("--mamba3-attention-period", type=int, default=4)
    parser.add_argument("--mamba3-d-state", type=int, default=128)
    parser.add_argument("--mamba3-expand", type=int, default=2)
    parser.add_argument("--mamba3-headdim", type=int, default=64)
    parser.add_argument("--mamba3-ngroups", type=int, default=1)
    parser.add_argument("--mamba3-is-mimo", action="store_true")
    parser.add_argument("--mamba3-mimo-rank", type=int, default=4)
    parser.add_argument("--mamba3-rope-fraction", type=float, default=0.5)
    parser.add_argument("--mamba3-chunk-size", type=int, default=0)
    parser.add_argument("--mamba3-outproj-norm", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--max-tokens-per-batch", type=int, default=65536)
    parser.add_argument(
        "--eval-max-tokens-per-batch",
        type=int,
        default=65536,
        help="Evaluation token budget. Independent from --max-tokens-per-batch. Mainly used in packed mode.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Micro-batch size used in --packing-mode unpadded.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Eval micro-batch size used in --packing-mode unpadded.",
    )
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument("--max-eval-groups", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=0, help="Use fixed optimizer steps. Set 0 to use exact epoch mode.")
    parser.add_argument("--train-epochs", type=int, default=1, help="Exact epoch count used when --train-steps is 0.")
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--early-stopping-metric", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--optimizer-name", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--lr-scheduler-type", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--muon-aux-learning-rate", type=float, default=3e-4)
    parser.add_argument("--muon-aux-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--use-muon-plus", action="store_true")
    parser.add_argument("--muonplus-norm-eps", type=float, default=1e-7)
    parser.add_argument("--no-muon-nesterov", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-split-eval-ratio", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--no-tf32", action="store_false", dest="use_tf32", default=True)
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mahjongLM_gpt2")
    parser.add_argument("--wandb-entity", type=str, default="a21-3jck-")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", action="append", default=[])
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--no-save-optimizer-state", action="store_false", dest="save_optimizer_state", default=True)
    parser.add_argument("--separate-eval-process", action="store_true", dest="use_separate_eval_process", default=False)
    parser.add_argument("--eval-device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-latest-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_options = {token for token in sys.argv[1:] if token.startswith("--")}

    def _apply_preset_value(option: str, attr: str, value) -> None:
        if option not in cli_options:
            setattr(args, attr, value)

    if not args.dataset_dir:
        processed_root = Path("data/processed")
        auto_dataset_dirs = sorted(
            path
            for path in processed_root.iterdir()
            if path.is_dir() and path.name != ".ingested" and (path / "dataset_info.json").is_file()
        ) if processed_root.is_dir() else []
        if not auto_dataset_dirs:
            args.dataset_dir = [Path("data/processed/2021")]
        else:
            args.dataset_dir = auto_dataset_dirs
    if args.model_family == "gpt2":
        if args.arch != "custom":
            if args.arch not in ARCH_PRESETS:
                raise ValueError(f"unknown GPT-2 arch preset: {args.arch}")
            preset = ARCH_PRESETS[args.arch]
            args.n_layer = preset.n_layer
            args.n_head = preset.n_head
            args.n_embd = preset.n_embd
            args.n_inner = 0 if preset.n_inner is None else preset.n_inner

        model_config = TinyGPT2Config(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_inner=None if args.n_inner == 0 else args.n_inner,
            n_positions=args.n_positions,
        )
    else:
        if args.qwen_arch != "custom":
            if args.qwen_arch not in QWEN3_ARCH_PRESETS:
                raise ValueError(f"unknown Qwen3 arch preset: {args.qwen_arch}")
            preset = QWEN3_ARCH_PRESETS[args.qwen_arch]
            _apply_preset_value("--qwen-hidden-size", "qwen_hidden_size", preset.hidden_size)
            _apply_preset_value("--qwen-intermediate-size", "qwen_intermediate_size", preset.intermediate_size)
            _apply_preset_value("--qwen-num-hidden-layers", "qwen_num_hidden_layers", preset.num_hidden_layers)
            _apply_preset_value("--qwen-num-attention-heads", "qwen_num_attention_heads", preset.num_attention_heads)
            _apply_preset_value("--qwen-num-key-value-heads", "qwen_num_key_value_heads", preset.num_key_value_heads)
            _apply_preset_value("--qwen-head-dim", "qwen_head_dim", preset.head_dim)
            _apply_preset_value("--qwen-max-position-embeddings", "qwen_max_position_embeddings", preset.max_position_embeddings)
            _apply_preset_value("--use-exclusive-self-attention", "use_exclusive_self_attention", preset.use_exclusive_self_attention)
            _apply_preset_value("--use-gated-attention", "use_gated_attention", preset.use_gated_attention)
            _apply_preset_value("--use-zero-centered-rmsnorm", "use_zero_centered_rmsnorm", preset.use_zero_centered_rmsnorm)
            _apply_preset_value("--use-rescaled-residual", "use_rescaled_residual", preset.use_rescaled_residual)
            _apply_preset_value("--use-mamba3-hybrid", "use_mamba3_hybrid", preset.use_mamba3_hybrid)
            _apply_preset_value("--mamba3-with-mlp-block", "mamba3_with_mlp_block", preset.mamba3_with_mlp_block)
            _apply_preset_value("--mamba3-attention-period", "mamba3_attention_period", preset.mamba3_attention_period)
            _apply_preset_value("--mamba3-d-state", "mamba3_d_state", preset.mamba3_d_state)
            _apply_preset_value("--mamba3-expand", "mamba3_expand", preset.mamba3_expand)
            _apply_preset_value("--mamba3-headdim", "mamba3_headdim", preset.mamba3_headdim)
            _apply_preset_value("--mamba3-ngroups", "mamba3_ngroups", preset.mamba3_ngroups)
            _apply_preset_value("--mamba3-is-mimo", "mamba3_is_mimo", preset.mamba3_is_mimo)
            _apply_preset_value("--mamba3-mimo-rank", "mamba3_mimo_rank", preset.mamba3_mimo_rank)
            _apply_preset_value("--mamba3-rope-fraction", "mamba3_rope_fraction", preset.mamba3_rope_fraction)
            _apply_preset_value("--mamba3-chunk-size", "mamba3_chunk_size", preset.mamba3_chunk_size)
            _apply_preset_value("--mamba3-outproj-norm", "mamba3_outproj_norm", preset.mamba3_is_outproj_norm)

        resolved_mamba3_chunk_size = args.mamba3_chunk_size
        if resolved_mamba3_chunk_size == 0:
            resolved_mamba3_chunk_size = 8 if args.mamba3_is_mimo else 64
        model_config = TinyQwen3Config(
            hidden_size=args.qwen_hidden_size,
            intermediate_size=args.qwen_intermediate_size,
            num_hidden_layers=args.qwen_num_hidden_layers,
            num_attention_heads=args.qwen_num_attention_heads,
            num_key_value_heads=args.qwen_num_key_value_heads,
            head_dim=args.qwen_head_dim,
            max_position_embeddings=args.qwen_max_position_embeddings,
            use_exclusive_self_attention=args.use_exclusive_self_attention,
            use_gated_attention=args.use_gated_attention,
            use_zero_centered_rmsnorm=args.use_zero_centered_rmsnorm,
            use_rescaled_residual=args.use_rescaled_residual,
            use_mamba3_hybrid=args.use_mamba3_hybrid,
            mamba3_with_mlp_block=args.mamba3_with_mlp_block,
            mamba3_attention_period=args.mamba3_attention_period,
            mamba3_d_state=args.mamba3_d_state,
            mamba3_expand=args.mamba3_expand,
            mamba3_headdim=args.mamba3_headdim,
            mamba3_ngroups=args.mamba3_ngroups,
            mamba3_is_mimo=args.mamba3_is_mimo,
            mamba3_mimo_rank=args.mamba3_mimo_rank,
            mamba3_rope_fraction=args.mamba3_rope_fraction,
            mamba3_chunk_size=resolved_mamba3_chunk_size,
            mamba3_is_outproj_norm=args.mamba3_outproj_norm,
        )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        dataset_dirs=tuple(args.dataset_dir),
        eval_dataset_dirs=tuple(args.eval_dataset_dir),
        tokenizer_dir=args.tokenizer_dir,
        model_family=args.model_family,
        packing_mode=args.packing_mode,
        attn_implementation=args.attn_implementation,
        max_seq_length=args.max_seq_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
        max_tokens_per_batch=args.max_tokens_per_batch,
        eval_max_tokens_per_batch=args.eval_max_tokens_per_batch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_train_groups=args.max_train_groups,
        max_eval_groups=args.max_eval_groups,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_steps=args.train_steps,
        train_epochs=args.train_epochs,
        eval_interval=args.eval_interval,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_patience=args.early_stopping_patience,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        optimizer_name=args.optimizer_name,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        muon_aux_learning_rate=args.muon_aux_learning_rate,
        muon_aux_weight_decay=args.muon_aux_weight_decay,
        muon_momentum=args.muon_momentum,
        muon_nesterov=not args.no_muon_nesterov,
        muon_ns_steps=args.muon_ns_steps,
        use_muon_plus=args.use_muon_plus,
        muonplus_norm_eps=args.muonplus_norm_eps,
        max_grad_norm=args.max_grad_norm,
        label_smoothing=args.label_smoothing,
        train_split_eval_ratio=args.train_split_eval_ratio,
        split_seed=args.split_seed,
        dataloader_num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        torch_compile=args.torch_compile,
        gradient_checkpointing=args.gradient_checkpointing,
        use_bf16=args.bf16,
        use_tf32=args.use_tf32,
        require_wandb=args.require_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume_run_id=args.wandb_run_id,
        wandb_mode=args.wandb_mode,
        wandb_tags=tuple(args.wandb_tags),
        stop_file=args.stop_file,
        keep_last_checkpoints=args.keep_last_checkpoints,
        detect_anomaly=args.detect_anomaly,
        save_optimizer_state=args.save_optimizer_state,
        use_separate_eval_process=args.use_separate_eval_process,
        eval_device=args.eval_device,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_latest_checkpoint=args.resume_latest_checkpoint,
    )
    metrics = train(model_config=model_config, training_config=training_config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
