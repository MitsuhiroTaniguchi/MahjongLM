from __future__ import annotations

from .config import TinyGPT2Config


def build_tiny_gpt2_model(
    *,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    config: TinyGPT2Config | None = None,
):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ModuleNotFoundError as exc:
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
    return GPT2LMHeadModel(hf_config)


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
