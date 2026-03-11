from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TinyGPT2Config:
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    n_inner: int = 256
    n_positions: int = 8192
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_by_inverse_layer_idx: bool = True
    reorder_and_upcast_attn: bool = False

    def validate(self) -> None:
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_positions < 32:
            raise ValueError("n_positions is too small for practical training")
        if self.n_inner < self.n_embd:
            raise ValueError("n_inner should be >= n_embd")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    dataset_dirs: tuple[Path, ...]
    eval_dataset_dirs: tuple[Path, ...] = ()
    tokenizer_dir: Path = Path("tokenizer")
    max_seq_length: int = 8192
    pad_to_multiple_of: int = 8
    max_tokens_per_batch: int = 65536
    eval_max_tokens_per_batch: int = 65536
    gradient_accumulation_steps: int = 1
    train_steps: int = 1000
    eval_interval: int = 200
    save_interval: int = 200
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    train_split_eval_ratio: float = 0.01
    split_seed: int = 1337
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    seed: int = 1337
    torch_compile: bool = False
    gradient_checkpointing: bool = False
    use_bf16: bool = True
    use_tf32: bool = True
    require_wandb: bool = False
    wandb_project: str = "mahjonglm-gpt2"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)
    keep_last_checkpoints: int = 3
    detect_anomaly: bool = False
    save_optimizer_state: bool = True

    def validate(self) -> None:
        if not self.dataset_dirs:
            raise ValueError("dataset_dirs must not be empty")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        if self.max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive")
        if self.eval_max_tokens_per_batch <= 0:
            raise ValueError("eval_max_tokens_per_batch must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 < self.min_learning_rate_ratio <= 1.0):
            raise ValueError("min_learning_rate_ratio must be in (0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not (0.0 <= self.train_split_eval_ratio < 1.0):
            raise ValueError("train_split_eval_ratio must be in [0, 1)")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        if self.keep_last_checkpoints <= 0:
            raise ValueError("keep_last_checkpoints must be positive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be one of: online, offline, disabled")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["dataset_dirs"] = [str(path) for path in self.dataset_dirs]
        payload["eval_dataset_dirs"] = [str(path) for path in self.eval_dataset_dirs]
        payload["tokenizer_dir"] = str(self.tokenizer_dir)
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
