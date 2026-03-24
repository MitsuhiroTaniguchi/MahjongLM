from __future__ import annotations

from pathlib import Path

from .config import TinyGPT2Config, TinyQwen3Config


def build_tiny_gpt2_model(
    *,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    attn_implementation: str | None = None,
    dtype=None,
    config: TinyGPT2Config | None = None,
):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("transformers with torch support is required for training") from exc

    model_config = config if config is not None else TinyGPT2Config()
    model_config.validate()
    hf_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=model_config.n_positions,
        n_ctx=model_config.n_positions,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_inner=model_config.n_inner,
        resid_pdrop=model_config.resid_pdrop,
        embd_pdrop=model_config.embd_pdrop,
        attn_pdrop=model_config.attn_pdrop,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        initializer_range=model_config.initializer_range,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        scale_attn_by_inverse_layer_idx=model_config.scale_attn_by_inverse_layer_idx,
        reorder_and_upcast_attn=model_config.reorder_and_upcast_attn,
        use_cache=False,
    )
    if dtype is not None:
        hf_config.dtype = dtype
    if attn_implementation is not None:
        hf_config._attn_implementation = attn_implementation
    model = GPT2LMHeadModel(hf_config)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def build_tiny_qwen3_model(
    *,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    attn_implementation: str | None = None,
    dtype=None,
    config: TinyQwen3Config | None = None,
):
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("transformers with torch support is required for training") from exc

    model_config = config if config is not None else TinyQwen3Config()
    model_config.validate()
    hf_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        max_position_embeddings=model_config.max_position_embeddings,
        max_window_layers=model_config.num_hidden_layers,
        attention_bias=model_config.attention_bias,
        attention_dropout=model_config.attention_dropout,
        hidden_act=model_config.hidden_act,
        rms_norm_eps=model_config.rms_norm_eps,
        rope_theta=model_config.rope_theta,
        tie_word_embeddings=model_config.tie_word_embeddings,
        use_sliding_window=model_config.use_sliding_window,
        sliding_window=model_config.sliding_window,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
    )
    if dtype is not None:
        hf_config.dtype = dtype
    if attn_implementation is not None:
        hf_config._attn_implementation = attn_implementation
    model = Qwen3ForCausalLM(hf_config)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def _load_pretrained_state_dict(model_dir: Path):
    try:
        from safetensors.torch import load_file as load_safetensors_file
    except ModuleNotFoundError:
        load_safetensors_file = None
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    pytorch_bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.is_file():
        if load_safetensors_file is None:
            raise ModuleNotFoundError("safetensors is required to load model.safetensors checkpoints")
        return load_safetensors_file(str(safetensors_path))
    if pytorch_bin_path.is_file():
        import torch

        return torch.load(pytorch_bin_path, map_location="cpu")
    raise FileNotFoundError(f"no supported weight file found in {model_dir}")


def load_saved_causal_lm(
    *,
    model_dir: Path,
    attn_implementation: str | None = None,
    dtype=None,
):
    try:
        from transformers import AutoConfig, GPT2LMHeadModel, Qwen3ForCausalLM
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("transformers with torch support is required for loading checkpoints") from exc

    model_dir = Path(model_dir)
    hf_config = AutoConfig.from_pretrained(str(model_dir))
    if dtype is not None:
        hf_config.dtype = dtype
    if attn_implementation is not None:
        hf_config._attn_implementation = attn_implementation

    if hf_config.model_type == "gpt2":
        model = GPT2LMHeadModel(hf_config)
    elif hf_config.model_type == "qwen3":
        model = Qwen3ForCausalLM(hf_config)
    else:
        raise ValueError(f"unsupported saved model_type: {hf_config.model_type}")

    state_dict = _load_pretrained_state_dict(model_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch for {model_dir}: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
