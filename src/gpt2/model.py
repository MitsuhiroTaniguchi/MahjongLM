from __future__ import annotations

from pathlib import Path

from .config import TinyGPT2Config, TinyQwen3Config
from .mamba3_adapter import OfficialMamba3Block


def _patch_qwen3_with_xsa(model) -> None:
    import torch
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        Qwen3Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
        repeat_kv,
    )

    class Qwen3ExclusiveSelfAttention(Qwen3Attention):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.xsa_eps = 1e-6

        def _apply_xsa(self, attn_output: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, num_heads, head_dim = attn_output.shape
            base_value_states = value_states.transpose(1, 2)
            if base_value_states.shape[1] != seq_len:
                base_value_states = base_value_states[:, -seq_len:, :, :]
            if num_heads != base_value_states.shape[2] * self.num_key_value_groups:
                repeated_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2)
                if repeated_value_states.shape[1] != seq_len:
                    repeated_value_states = repeated_value_states[:, -seq_len:, :, :]
                denom = repeated_value_states.square().sum(dim=-1, keepdim=True, dtype=torch.float32).clamp_min(self.xsa_eps)
                dot = (attn_output * repeated_value_states).sum(dim=-1, keepdim=True, dtype=torch.float32)
                coeff = (dot / denom).to(dtype=attn_output.dtype)
                return attn_output - coeff * repeated_value_states

            grouped_attn_output = attn_output.reshape(
                batch_size,
                seq_len,
                base_value_states.shape[2],
                self.num_key_value_groups,
                head_dim,
            )
            grouped_value_states = base_value_states.unsqueeze(3)
            denom = (
                base_value_states.square().sum(dim=-1, keepdim=True, dtype=torch.float32).unsqueeze(3).clamp_min(self.xsa_eps)
            )
            dot = (grouped_attn_output * grouped_value_states).sum(dim=-1, keepdim=True, dtype=torch.float32)
            coeff = (dot / denom).to(dtype=attn_output.dtype)
            corrected = grouped_attn_output - coeff * grouped_value_states
            return corrected.reshape_as(attn_output)

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            attn_output = self._apply_xsa(attn_output, value_states)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        new_attn = Qwen3ExclusiveSelfAttention(model.config, layer_idx)
        new_attn.load_state_dict(old_attn.state_dict())
        new_attn.to(device=old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
        layer.self_attn = new_attn


def _patch_qwen3_with_gated_attention(model) -> None:
    import torch
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        Qwen3Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    class Qwen3GatedAttention(Qwen3Attention):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.q_proj = torch.nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim * 2,
                bias=config.attention_bias,
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        new_attn = Qwen3GatedAttention(model.config, layer_idx)
        new_attn.to(device=old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
        with torch.no_grad():
            q_out = old_attn.q_proj.weight.shape[0]
            new_attn.q_proj.weight.zero_()
            new_attn.q_proj.weight[:q_out].copy_(old_attn.q_proj.weight)
            if old_attn.q_proj.bias is not None and new_attn.q_proj.bias is not None:
                new_attn.q_proj.bias.zero_()
                new_attn.q_proj.bias[:q_out].copy_(old_attn.q_proj.bias)
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
            new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
        layer.self_attn = new_attn


def _patch_qwen3_with_mamba3_hybrid(model, model_config: TinyQwen3Config) -> None:
    from transformers.modeling_layers import GradientCheckpointingLayer

    class Qwen3Mamba3DecoderLayer(GradientCheckpointingLayer):
        attention_type = "full_attention"

        def __init__(self, config, layer_idx: int, reference_layer):
            super().__init__()
            self.layer_idx = layer_idx
            self.input_layernorm = reference_layer.input_layernorm
            self.mamba3 = OfficialMamba3Block(
                d_model=config.hidden_size,
                d_state=model_config.mamba3_d_state,
                expand=model_config.mamba3_expand,
                headdim=model_config.mamba3_headdim,
                ngroups=model_config.mamba3_ngroups,
                is_mimo=model_config.mamba3_is_mimo,
                mimo_rank=model_config.mamba3_mimo_rank,
                rope_fraction=model_config.mamba3_rope_fraction,
                is_outproj_norm=model_config.mamba3_is_outproj_norm,
                chunk_size=model_config.mamba3_chunk_size,
                layer_idx=layer_idx,
                device=reference_layer.self_attn.q_proj.weight.device,
                dtype=reference_layer.self_attn.q_proj.weight.dtype,
            )

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            del position_embeddings, attention_mask, cache_position, kwargs
            if past_key_values is not None:
                raise NotImplementedError("Mamba3 hybrid currently does not support KV-cache inference.")
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.mamba3(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

    attention_layer_idx = [
        idx for idx in range(model_config.num_hidden_layers) if (idx + 1) % model_config.mamba3_attention_period == 0
    ]
    model.config.mamba3_attention_layer_idx = attention_layer_idx
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in attention_layer_idx:
            continue
        model.model.layers[layer_idx] = Qwen3Mamba3DecoderLayer(model.config, layer_idx, layer)


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
    hf_config.use_exclusive_self_attention = model_config.use_exclusive_self_attention
    hf_config.use_gated_attention = model_config.use_gated_attention
    hf_config.use_mamba3_hybrid = model_config.use_mamba3_hybrid
    hf_config.mamba3_attention_period = model_config.mamba3_attention_period
    hf_config.mamba3_d_state = model_config.mamba3_d_state
    hf_config.mamba3_expand = model_config.mamba3_expand
    hf_config.mamba3_headdim = model_config.mamba3_headdim
    hf_config.mamba3_ngroups = model_config.mamba3_ngroups
    hf_config.mamba3_is_mimo = model_config.mamba3_is_mimo
    hf_config.mamba3_mimo_rank = model_config.mamba3_mimo_rank
    hf_config.mamba3_rope_fraction = model_config.mamba3_rope_fraction
    hf_config.mamba3_chunk_size = model_config.mamba3_chunk_size
    hf_config.mamba3_is_outproj_norm = model_config.mamba3_is_outproj_norm
    model = Qwen3ForCausalLM(hf_config)
    if model_config.use_exclusive_self_attention:
        _patch_qwen3_with_xsa(model)
    if model_config.use_gated_attention:
        _patch_qwen3_with_gated_attention(model)
    if model_config.use_mamba3_hybrid:
        _patch_qwen3_with_mamba3_hybrid(model, model_config)
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
        if getattr(hf_config, "use_exclusive_self_attention", False):
            _patch_qwen3_with_xsa(model)
        if getattr(hf_config, "use_gated_attention", False):
            _patch_qwen3_with_gated_attention(model)
        if getattr(hf_config, "use_mamba3_hybrid", False):
            model_config = TinyQwen3Config(
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=hf_config.head_dim,
                max_position_embeddings=hf_config.max_position_embeddings,
                use_exclusive_self_attention=getattr(hf_config, "use_exclusive_self_attention", False),
                use_gated_attention=getattr(hf_config, "use_gated_attention", False),
                use_mamba3_hybrid=getattr(hf_config, "use_mamba3_hybrid", False),
                mamba3_attention_period=getattr(hf_config, "mamba3_attention_period", 4),
                mamba3_d_state=getattr(hf_config, "mamba3_d_state", 128),
                mamba3_expand=getattr(hf_config, "mamba3_expand", 2),
                mamba3_headdim=getattr(hf_config, "mamba3_headdim", 64),
                mamba3_ngroups=getattr(hf_config, "mamba3_ngroups", 1),
                mamba3_is_mimo=getattr(hf_config, "mamba3_is_mimo", False),
                mamba3_mimo_rank=getattr(hf_config, "mamba3_mimo_rank", 4),
                mamba3_rope_fraction=getattr(hf_config, "mamba3_rope_fraction", 0.5),
                mamba3_chunk_size=getattr(hf_config, "mamba3_chunk_size", 64),
                mamba3_is_outproj_norm=getattr(hf_config, "mamba3_is_outproj_norm", False),
            )
            _patch_qwen3_with_mamba3_hybrid(model, model_config)
    else:
        raise ValueError(f"unsupported saved model_type: {hf_config.model_type}")

    state_dict = _load_pretrained_state_dict(model_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    tied_lm_head_missing = (
        hf_config.model_type == "qwen3"
        and getattr(hf_config, "tie_word_embeddings", True)
        and missing_keys == ["lm_head.weight"]
    )
    if tied_lm_head_missing:
        model.tie_weights()
        missing_keys = []
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch for {model_dir}: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
