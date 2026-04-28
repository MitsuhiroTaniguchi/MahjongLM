from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TinyGPT2Config:
    n_layer: int = 20
    n_head: int = 10
    n_embd: int = 640
    n_inner: int | None = None
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
        if self.n_inner is not None and self.n_inner < self.n_embd:
            raise ValueError("n_inner should be >= n_embd")


@dataclass(frozen=True)
class TinyQwen3Config:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 8192
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    use_sliding_window: bool = False
    sliding_window: int | None = None
    use_exclusive_self_attention: bool = False
    use_gated_attention: bool = False
    use_zero_centered_rmsnorm: bool = False
    use_rescaled_residual: bool = False
    use_attention_residuals: bool = False
    attention_residual_num_blocks: int = 8
    attention_residual_recency_bias_init: float = 0.0
    attention_residual_mode: str = "block"
    attention_residual_gate_type: str = "bias"
    use_mamba3_hybrid: bool = False
    mamba3_with_mlp_block: bool = False
    mamba3_attention_period: int = 4
    mamba3_d_state: int = 128
    mamba3_expand: int = 2
    mamba3_headdim: int = 64
    mamba3_ngroups: int = 1
    mamba3_is_mimo: bool = False
    mamba3_mimo_rank: int = 4
    mamba3_rope_fraction: float = 0.5
    mamba3_chunk_size: int = 0
    mamba3_is_outproj_norm: bool = False

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= self.hidden_size:
            raise ValueError("intermediate_size must be greater than hidden_size")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads must be <= num_attention_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.max_position_embeddings < 32:
            raise ValueError("max_position_embeddings is too small for practical training")
        if self.hidden_act != "silu":
            raise ValueError("TinyQwen3Config currently supports hidden_act='silu' only")
        if self.use_exclusive_self_attention and self.use_gated_attention:
            raise ValueError("use_exclusive_self_attention and use_gated_attention are mutually exclusive")
        if self.use_rescaled_residual and self.use_attention_residuals:
            raise ValueError("use_rescaled_residual and use_attention_residuals are mutually exclusive")
        if self.attention_residual_num_blocks <= 0:
            raise ValueError("attention_residual_num_blocks must be positive")
        if self.attention_residual_mode not in {"block", "full"}:
            raise ValueError("attention_residual_mode must be 'block' or 'full'")
        if self.attention_residual_gate_type not in {"bias", "sigmoid_scalar", "sigmoid_vector"}:
            raise ValueError("attention_residual_gate_type must be 'bias', 'sigmoid_scalar', or 'sigmoid_vector'")
        if self.mamba3_attention_period <= 0:
            raise ValueError("mamba3_attention_period must be positive")
        if self.mamba3_d_state <= 0:
            raise ValueError("mamba3_d_state must be positive")
        if self.mamba3_expand <= 0:
            raise ValueError("mamba3_expand must be positive")
        if self.mamba3_headdim <= 0:
            raise ValueError("mamba3_headdim must be positive")
        if self.mamba3_ngroups <= 0:
            raise ValueError("mamba3_ngroups must be positive")
        if self.mamba3_mimo_rank <= 0:
            raise ValueError("mamba3_mimo_rank must be positive")
        if self.mamba3_rope_fraction not in {0.5, 1.0}:
            raise ValueError("mamba3_rope_fraction must be 0.5 or 1.0")
        if self.mamba3_chunk_size < 0:
            raise ValueError("mamba3_chunk_size must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    dataset_dirs: tuple[Path, ...]
    eval_dataset_dirs: tuple[Path, ...] = ()
    tokenizer_dir: Path = Path("tokenizer")
    cache_dir: Path = Path("data/cache")
    model_family: str = "gpt2"
    packing_mode: str = "packed"
    attn_implementation: str = "sdpa"
    max_seq_length: int = 8192
    pad_to_multiple_of: int = 8
    max_tokens_per_batch: int = 65536
    eval_max_tokens_per_batch: int = 65536
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    max_train_groups: int = 0
    max_eval_groups: int = 0
    gradient_accumulation_steps: int = 1
    train_steps: int = 0
    train_epochs: float = 1.0
    eval_interval: int = 200
    early_stopping_metric: str | None = None
    early_stopping_patience: int = 0
    save_interval: int = 200
    log_interval: int = 10
    optimizer_name: str = "adamw"
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    muon_aux_learning_rate: float = 3e-4
    muon_aux_weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    use_muon_plus: bool = False
    muonplus_norm_eps: float = 1e-7
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    train_split_eval_ratio: float = 0.01
    split_seed: int = 1337
    data_seed: int | None = None
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    seed: int = 1337
    torch_compile: bool = False
    gradient_checkpointing: bool = False
    use_bf16: bool = True
    use_tf32: bool = True
    require_wandb: bool = False
    wandb_project: str = "mahjongLM_gpt2"
    wandb_entity: str | None = "a21-3jck-"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)
    stop_file: Path | None = None
    keep_last_checkpoints: int = 3
    detect_anomaly: bool = False
    save_optimizer_state: bool = True
    validate_dataset: bool = False
    use_separate_eval_process: bool = False
    eval_device: str = "cuda"
    resume_from_checkpoint: Path | None = None
    resume_latest_checkpoint: bool = False
    wandb_resume_run_id: str | None = None
    teacher_model_dir: Path | None = None
    distillation_alpha: float = 0.0
    distillation_temperature: float = 1.0
    distillation_teacher_device: str = "same"

    def validate(self) -> None:
        if not self.dataset_dirs:
            raise ValueError("dataset_dirs must not be empty")
        if self.model_family not in {"gpt2", "qwen3"}:
            raise ValueError("model_family must be one of: gpt2, qwen3")
        if self.packing_mode not in {"packed", "unpadded"}:
            raise ValueError("packing_mode must be one of: packed, unpadded")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.attn_implementation not in {"eager", "sdpa", "flash_attention_2", "flash_attention_3"}:
            raise ValueError("attn_implementation must be one of: eager, sdpa, flash_attention_2, flash_attention_3")
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        if self.max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive")
        if self.eval_max_tokens_per_batch <= 0:
            raise ValueError("eval_max_tokens_per_batch must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("per_device_eval_batch_size must be positive")
        if self.max_train_groups < 0:
            raise ValueError("max_train_groups must be non-negative")
        if self.max_eval_groups < 0:
            raise ValueError("max_eval_groups must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.train_steps < 0:
            raise ValueError("train_steps must be non-negative")
        if self.train_epochs < 0:
            raise ValueError("train_epochs must be non-negative")
        if self.train_steps == 0 and self.train_epochs <= 0:
            raise ValueError("either train_steps or train_epochs must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_patience > 0 and not self.early_stopping_metric:
            raise ValueError("early_stopping_metric must be set when early_stopping_patience is positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.optimizer_name not in {"adamw", "muon"}:
            raise ValueError("optimizer_name must be one of: adamw, muon")
        if self.lr_scheduler_type not in {"cosine", "linear"}:
            raise ValueError("lr_scheduler_type must be one of: cosine, linear")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 < self.min_learning_rate_ratio <= 1.0):
            raise ValueError("min_learning_rate_ratio must be in (0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.muon_aux_learning_rate <= 0:
            raise ValueError("muon_aux_learning_rate must be positive")
        if self.muon_aux_weight_decay < 0:
            raise ValueError("muon_aux_weight_decay must be non-negative")
        if not (0.0 < self.muon_momentum < 1.0):
            raise ValueError("muon_momentum must be in (0, 1)")
        if self.muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive")
        if self.muonplus_norm_eps <= 0:
            raise ValueError("muonplus_norm_eps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        if not (0.0 <= self.train_split_eval_ratio < 1.0):
            raise ValueError("train_split_eval_ratio must be in [0, 1)")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        if self.keep_last_checkpoints <= 0:
            raise ValueError("keep_last_checkpoints must be positive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be one of: online, offline, disabled")
        if self.eval_device not in {"cpu", "cuda"}:
            raise ValueError("eval_device must be one of: cpu, cuda")
        if self.resume_from_checkpoint is not None and self.resume_latest_checkpoint:
            raise ValueError("resume_from_checkpoint and resume_latest_checkpoint are mutually exclusive")
        if self.teacher_model_dir is None and self.distillation_alpha > 0.0:
            raise ValueError("teacher_model_dir must be set when distillation_alpha is positive")
        if not (0.0 <= self.distillation_alpha <= 1.0):
            raise ValueError("distillation_alpha must be in [0, 1]")
        if self.distillation_temperature <= 0.0:
            raise ValueError("distillation_temperature must be positive")
        if self.distillation_teacher_device not in {"same", "cpu", "cuda"}:
            raise ValueError("distillation_teacher_device must be one of: same, cpu, cuda")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["dataset_dirs"] = [str(path) for path in self.dataset_dirs]
        payload["eval_dataset_dirs"] = [str(path) for path in self.eval_dataset_dirs]
        payload["tokenizer_dir"] = str(self.tokenizer_dir)
        payload["cache_dir"] = str(self.cache_dir)
        payload["stop_file"] = str(self.stop_file) if self.stop_file is not None else None
        payload["resume_from_checkpoint"] = (
            str(self.resume_from_checkpoint) if self.resume_from_checkpoint is not None else None
        )
        payload["teacher_model_dir"] = str(self.teacher_model_dir) if self.teacher_model_dir is not None else None
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
