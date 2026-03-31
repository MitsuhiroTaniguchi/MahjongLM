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
    import copy
    import torch
    from types import SimpleNamespace
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        qwen35_attention_config = SimpleNamespace(
            hidden_size=model.config.hidden_size,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            attention_dropout=model.config.attention_dropout,
            attention_bias=model.config.attention_bias,
            rms_norm_eps=model.config.rms_norm_eps,
            _attn_implementation=model.config._attn_implementation,
        )
        new_attn = Qwen3_5Attention(qwen35_attention_config, layer_idx)
        new_attn.to(device=old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
        with torch.no_grad():
            new_attn.q_proj.weight.zero_()
            old_q_weight = old_attn.q_proj.weight.view(model.config.num_attention_heads, model.config.head_dim, model.config.hidden_size)
            new_q_weight = new_attn.q_proj.weight.view(model.config.num_attention_heads, model.config.head_dim * 2, model.config.hidden_size)
            new_q_weight[:, : model.config.head_dim, :].copy_(old_q_weight)
            if old_attn.q_proj.bias is not None and new_attn.q_proj.bias is not None:
                new_attn.q_proj.bias.zero_()
                old_q_bias = old_attn.q_proj.bias.view(model.config.num_attention_heads, model.config.head_dim)
                new_q_bias = new_attn.q_proj.bias.view(model.config.num_attention_heads, model.config.head_dim * 2)
                new_q_bias[:, : model.config.head_dim].copy_(old_q_bias)
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
        # Keep the Qwen3.5 gated attention path, but retain Qwen3's standard RMSNorm
        # for q/k normalization so gated attention can be tested independently of
        # zero-centered norm changes.
        new_attn.q_norm = copy.deepcopy(old_attn.q_norm)
        new_attn.k_norm = copy.deepcopy(old_attn.k_norm)
        layer.self_attn = new_attn


def _patch_qwen3_with_zero_centered_rmsnorm(model) -> None:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    def _convert_norm(old_norm):
        new_norm = Qwen3_5RMSNorm(old_norm.weight.shape[0], eps=old_norm.variance_epsilon)
        new_norm.to(device=old_norm.weight.device, dtype=old_norm.weight.dtype)
        with torch.no_grad():
            new_norm.weight.copy_(old_norm.weight - 1.0)
        new_norm.weight._force_weight_decay = True
        return new_norm

    model.model.norm = _convert_norm(model.model.norm)
    for layer in model.model.layers:
        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm = _convert_norm(layer.input_layernorm)
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm = _convert_norm(layer.post_attention_layernorm)


def _patch_qwen3_with_attention_residuals(model, model_config: TinyQwen3Config) -> None:
    import copy
    from types import MethodType

    import torch
    import torch.nn as nn
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def _make_norm_like(reference_norm):
        norm_cls = reference_norm.__class__
        eps = getattr(reference_norm, "variance_epsilon", getattr(reference_norm, "eps", model_config.rms_norm_eps))
        norm = norm_cls(reference_norm.weight.shape[0], eps=eps)
        norm.to(device=reference_norm.weight.device, dtype=reference_norm.weight.dtype)
        return norm

    def _init_linear(linear: nn.Linear) -> None:
        std = float(getattr(model.config, "initializer_range", 0.02))
        nn.init.normal_(linear.weight, mean=0.0, std=std)

    def _direct_rmsnorm_query(proj: nn.Linear, norm: nn.Module) -> tuple[torch.Tensor, float] | tuple[None, None]:
        eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
        weight = getattr(norm, "weight", None)
        if eps is None or weight is None:
            return None, None
        scaled_weight = weight
        if norm.__class__.__name__ == "Qwen3_5RMSNorm":
            scaled_weight = 1.0 + scaled_weight
        query = proj.weight.view(-1)
        return query * scaled_weight.to(dtype=query.dtype), float(eps)

    def _normalize_attnres_source(source: torch.Tensor, rms_eps: float) -> torch.Tensor:
        source_fp32 = source.to(torch.float32)
        inv_rms = torch.rsqrt(source_fp32.square().mean(dim=-1, keepdim=True) + rms_eps)
        return (source_fp32 * inv_rms).to(dtype=source.dtype)

    def _empty_block_cache_like(source: torch.Tensor) -> torch.Tensor:
        return source.new_empty((0, *source.shape))

    def _append_block_cache(
        completed_blocks: torch.Tensor,
        completed_blocks_norm: torch.Tensor,
        block: torch.Tensor,
        *,
        rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = block.unsqueeze(0)
        next_blocks = torch.cat((completed_blocks, block), dim=0)
        normalized_block = _normalize_attnres_source(block.squeeze(0), rms_eps).unsqueeze(0)
        next_blocks_norm = torch.cat((completed_blocks_norm, normalized_block), dim=0)
        return next_blocks, next_blocks_norm

    def _single_source_attn_stats(
        source: torch.Tensor,
        proj: nn.Linear,
        norm: nn.Module,
        bias: torch.Tensor | nn.Parameter | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direct_query, rms_eps = _direct_rmsnorm_query(proj, norm)
        if direct_query is None:
            logits = nn.functional.linear(norm(source), proj.weight).squeeze(-1)
        else:
            logits = nn.functional.linear(
                _normalize_attnres_source(source, rms_eps),
                direct_query.to(dtype=source.dtype).unsqueeze(0),
            ).squeeze(-1)
        if bias is not None:
            logits = logits + bias
        output = source
        max_logits = logits.to(torch.float32)
        lse = torch.ones_like(max_logits)
        return output, max_logits, lse

    def _merge_attn_stats(
        inter_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        intra_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        intra_output, intra_max, intra_lse = intra_stats
        if inter_stats[0].numel() == 0:
            return intra_output.to(dtype=target_dtype)

        inter_output, inter_max, inter_lse = inter_stats
        merged_max = torch.maximum(inter_max, intra_max)
        inter_scale = torch.exp(inter_max - merged_max)
        intra_scale = torch.exp(intra_max - merged_max)
        numerator = inter_scale.unsqueeze(-1) * inter_output.to(torch.float32) + intra_scale.unsqueeze(-1) * intra_output.to(torch.float32)
        denominator = inter_scale * inter_lse + intra_scale * intra_lse
        return (numerator / denominator.unsqueeze(-1)).to(dtype=target_dtype)

    def _empty_inter_stats(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = source.new_empty(0)
        return empty, empty, empty

    def _precompute_interblock_attn(
        completed_blocks: torch.Tensor,
        completed_blocks_norm: torch.Tensor,
        layers: list["Qwen3AttentionResidualLayer"],
    ) -> list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]]:
        if completed_blocks.shape[0] == 0:
            return [(_empty_inter_stats(completed_blocks), _empty_inter_stats(completed_blocks)) for _ in layers]

        query_specs: list[tuple[int, int, torch.Tensor, float]] = []
        for layer_idx, layer in enumerate(layers):
            first_query, first_eps = _direct_rmsnorm_query(layer.first_res_proj, layer.first_res_norm)
            if first_query is None:
                return [(_empty_inter_stats(completed_blocks), _empty_inter_stats(completed_blocks)) for _ in layers]
            query_specs.append((layer_idx, 0, first_query, first_eps))
            if layer.with_mlp_block:
                second_query, second_eps = _direct_rmsnorm_query(layer.second_res_proj, layer.second_res_norm)
                if second_query is None:
                    return [(_empty_inter_stats(completed_blocks), _empty_inter_stats(completed_blocks)) for _ in layers]
                query_specs.append((layer_idx, 1, second_query, second_eps))

        eps_values = {eps for _, _, _, eps in query_specs}
        if len(eps_values) != 1:
            return [(_empty_inter_stats(completed_blocks), _empty_inter_stats(completed_blocks)) for _ in layers]

        num_blocks, batch_size, seq_len, hidden_size = completed_blocks.shape
        num_queries = len(query_specs)

        queries = torch.stack([query.to(dtype=completed_blocks.dtype) for _, _, query, _ in query_specs], dim=0)
        flat_keys = completed_blocks_norm.permute(1, 2, 0, 3).contiguous().view(batch_size * seq_len, num_blocks, hidden_size)
        flat_values = completed_blocks.permute(1, 2, 0, 3).contiguous().view(batch_size * seq_len, num_blocks, hidden_size)

        flat_logits = torch.matmul(flat_keys, queries.t())  # [B*T, N, Q]
        flat_max_logits = flat_logits.amax(dim=1)
        flat_logits = flat_logits - flat_max_logits.unsqueeze(1)
        flat_logits.exp_()
        flat_lse = flat_logits.sum(dim=1)
        flat_outputs = torch.bmm(flat_logits.permute(0, 2, 1).to(flat_values.dtype), flat_values)  # [B*T, Q, D]

        max_logits = flat_max_logits.view(batch_size, seq_len, num_queries).permute(2, 0, 1).contiguous()
        lse = flat_lse.view(batch_size, seq_len, num_queries).permute(2, 0, 1).contiguous()
        outputs = flat_outputs.view(batch_size, seq_len, num_queries, hidden_size).permute(2, 0, 1, 3).contiguous()

        per_layer: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]] = [[None, None] for _ in layers]
        for query_idx, (layer_idx, branch_idx, _, _) in enumerate(query_specs):
            per_layer[layer_idx][branch_idx] = (outputs[query_idx], max_logits[query_idx], lse[query_idx])
        return [(branch_stats[0], branch_stats[1]) for branch_stats in per_layer]

    def block_attn_res(
        completed_blocks: torch.Tensor,
        completed_blocks_norm: torch.Tensor,
        partial_block: torch.Tensor,
        proj: nn.Linear,
        norm: nn.Module,
        recency_bias: nn.Parameter,
        return_entropy: bool = False,
    ):
        direct_query, rms_eps = _direct_rmsnorm_query(proj, norm)
        if direct_query is not None:
            direct_query = direct_query.to(dtype=partial_block.dtype)
        if direct_query is None:
            completed_logits = (
                nn.functional.linear(norm(completed_blocks), proj.weight).squeeze(-1)
                if completed_blocks.shape[0] > 0
                else partial_block.new_empty((0, partial_block.shape[0], partial_block.shape[1]))
            )
            partial_logits = nn.functional.linear(norm(partial_block), proj.weight).squeeze(-1)
        else:
            completed_logits = (
                nn.functional.linear(completed_blocks_norm, direct_query.unsqueeze(0)).squeeze(-1)
                if completed_blocks.shape[0] > 0
                else partial_block.new_empty((0, partial_block.shape[0], partial_block.shape[1]))
            )
            partial_logits = nn.functional.linear(_normalize_attnres_source(partial_block, rms_eps), direct_query.unsqueeze(0)).squeeze(-1)
        logits = torch.cat((completed_logits, (partial_logits + recency_bias).unsqueeze(0)), dim=0)
        weights = logits.softmax(dim=0)
        hidden = weights[-1].to(dtype=partial_block.dtype).unsqueeze(-1) * partial_block
        if completed_blocks.shape[0] > 0:
            hidden = hidden + (weights[:-1].to(dtype=partial_block.dtype).unsqueeze(-1) * completed_blocks).sum(dim=0)
        if return_entropy:
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
            return hidden, entropy
        return hidden

    class Qwen3AttentionResidualLayer(GradientCheckpointingLayer):
        def __init__(self, reference_layer, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx
            self.attention_type = getattr(reference_layer, "attention_type", "full_attention")
            self.attnres_mode = model_config.attention_residual_mode
            self.gate_type = model_config.attention_residual_gate_type
            self.layers_per_block = max(
                1,
                (model_config.num_hidden_layers + model_config.attention_residual_num_blocks - 1)
                // model_config.attention_residual_num_blocks,
            )
            self.is_mamba_layer = hasattr(reference_layer, "mamba3")

            self.input_layernorm = reference_layer.input_layernorm

            if self.is_mamba_layer:
                self.mamba3 = reference_layer.mamba3
                self.with_mlp_block = getattr(reference_layer, "with_mlp_block", False)
                if self.with_mlp_block:
                    self.post_attention_layernorm = reference_layer.post_attention_layernorm
                    self.mlp = reference_layer.mlp
            else:
                self.self_attn = reference_layer.self_attn
                self.mlp = reference_layer.mlp
                self.post_attention_layernorm = reference_layer.post_attention_layernorm
                self.with_mlp_block = True

            self.first_res_proj = nn.Linear(model_config.hidden_size, 1, bias=False)
            self.first_res_norm = _make_norm_like(self.input_layernorm)
            _init_linear(self.first_res_proj)

            self.second_res_proj = None
            self.second_res_norm = None
            if self.with_mlp_block:
                self.second_res_proj = nn.Linear(model_config.hidden_size, 1, bias=False)
                self.second_res_norm = _make_norm_like(self.post_attention_layernorm)
                _init_linear(self.second_res_proj)

            bias_init = float(model_config.attention_residual_recency_bias_init)
            if self.gate_type == "sigmoid_scalar":
                self.first_gate_logit = nn.Parameter(torch.tensor(-10.0))
                self.second_gate_logit = nn.Parameter(torch.tensor(-10.0)) if self.with_mlp_block else None
                self.first_res_bias = nn.Parameter(torch.tensor(0.0))
                self.second_res_bias = nn.Parameter(torch.tensor(0.0)) if self.with_mlp_block else None
            elif self.gate_type == "sigmoid_vector":
                self.first_gate_proj = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=True)
                nn.init.zeros_(self.first_gate_proj.weight)
                nn.init.constant_(self.first_gate_proj.bias, -10.0)
                self.second_gate_proj = None
                if self.with_mlp_block:
                    self.second_gate_proj = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=True)
                    nn.init.zeros_(self.second_gate_proj.weight)
                    nn.init.constant_(self.second_gate_proj.bias, -10.0)
                self.first_res_bias = nn.Parameter(torch.tensor(0.0))
                self.second_res_bias = nn.Parameter(torch.tensor(0.0)) if self.with_mlp_block else None
            else:
                self.first_res_bias = nn.Parameter(torch.tensor(bias_init))
                self.second_res_bias = nn.Parameter(torch.tensor(bias_init)) if self.with_mlp_block else None

        @property
        def is_block_boundary(self) -> bool:
            return (self.layer_idx + 1) % self.layers_per_block == 0

        def _apply_gate(self, hidden_states, attnres_hidden, *, branch_idx: int):
            if self.gate_type == "sigmoid_scalar":
                gate_logit = self.first_gate_logit if branch_idx == 0 else self.second_gate_logit
                gate = torch.sigmoid(gate_logit)
                return (1 - gate) * hidden_states + gate * attnres_hidden
            if self.gate_type == "sigmoid_vector":
                gate_proj = self.first_gate_proj if branch_idx == 0 else self.second_gate_proj
                gate = torch.sigmoid(gate_proj(hidden_states))
                return (1 - gate) * hidden_states + gate * attnres_hidden
            return attnres_hidden

        def _run_first_branch(
            self,
            partial_block,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
            position_embeddings,
        ):
            if self.is_mamba_layer:
                return self.mamba3(self.input_layernorm(partial_block))
            branch_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(partial_block),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            return branch_out

        def _run_second_branch(self, partial_block):
            return self.mlp(self.post_attention_layernorm(partial_block))

        def forward(
            self,
            completed_blocks: torch.Tensor,
            completed_blocks_norm: torch.Tensor,
            partial_block: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            use_cache: bool | None = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            first_inter_stats=None,
            second_inter_stats=None,
            entropy_accum=None,
            block_rms_eps=None,
            **kwargs,
        ):
            if "entropy_accum" in kwargs:
                entropy_accum = kwargs.pop("entropy_accum")
            if "block_rms_eps" in kwargs:
                block_rms_eps = kwargs.pop("block_rms_eps")

            if entropy_accum is not None:
                attnres_hidden, entropy = block_attn_res(
                    completed_blocks,
                    completed_blocks_norm,
                    partial_block,
                    self.first_res_proj,
                    self.first_res_norm,
                    self.first_res_bias,
                    return_entropy=True,
                )
                entropy_accum.append(entropy)
            else:
                if first_inter_stats[0].numel() == 0:
                    attnres_hidden = block_attn_res(
                        completed_blocks,
                        completed_blocks_norm,
                        partial_block,
                        self.first_res_proj,
                        self.first_res_norm,
                        self.first_res_bias,
                    )
                else:
                    attnres_hidden = _merge_attn_stats(
                        first_inter_stats,
                        _single_source_attn_stats(partial_block, self.first_res_proj, self.first_res_norm, self.first_res_bias),
                        target_dtype=partial_block.dtype,
                    )
            hidden = self._apply_gate(partial_block, attnres_hidden, branch_idx=0)

            if self.attnres_mode == "block" and self.is_block_boundary:
                completed_blocks, completed_blocks_norm = _append_block_cache(
                    completed_blocks,
                    completed_blocks_norm,
                    partial_block,
                    rms_eps=block_rms_eps,
                )
                partial_block = torch.zeros_like(partial_block)

            first_branch_out = self._run_first_branch(
                hidden,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings,
            )
            partial_block = partial_block + first_branch_out

            if self.attnres_mode == "full":
                completed_blocks, completed_blocks_norm = _append_block_cache(
                    completed_blocks,
                    completed_blocks_norm,
                    partial_block,
                    rms_eps=block_rms_eps,
                )

            if not self.with_mlp_block:
                if self.attnres_mode == "full":
                    completed_blocks, completed_blocks_norm = _append_block_cache(
                        completed_blocks,
                        completed_blocks_norm,
                        partial_block,
                        rms_eps=block_rms_eps,
                    )
                return completed_blocks, completed_blocks_norm, partial_block

            if entropy_accum is not None:
                attnres_hidden, entropy = block_attn_res(
                    completed_blocks,
                    completed_blocks_norm,
                    partial_block,
                    self.second_res_proj,
                    self.second_res_norm,
                    self.second_res_bias,
                    return_entropy=True,
                )
                entropy_accum.append(entropy)
            else:
                if second_inter_stats[0].numel() == 0:
                    attnres_hidden = block_attn_res(
                        completed_blocks,
                        completed_blocks_norm,
                        partial_block,
                        self.second_res_proj,
                        self.second_res_norm,
                        self.second_res_bias,
                    )
                else:
                    attnres_hidden = _merge_attn_stats(
                        second_inter_stats,
                        _single_source_attn_stats(partial_block, self.second_res_proj, self.second_res_norm, self.second_res_bias),
                        target_dtype=partial_block.dtype,
                    )
            hidden = self._apply_gate(partial_block, attnres_hidden, branch_idx=1)
            partial_block = partial_block + self._run_second_branch(hidden)

            if self.attnres_mode == "full":
                completed_blocks, completed_blocks_norm = _append_block_cache(
                    completed_blocks,
                    completed_blocks_norm,
                    partial_block,
                    rms_eps=block_rms_eps,
                )
            return completed_blocks, completed_blocks_norm, partial_block

    for layer_idx, layer in enumerate(model.model.layers):
        model.model.layers[layer_idx] = Qwen3AttentionResidualLayer(layer, layer_idx)

    def attnres_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(self, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        completed_blocks = _empty_block_cache_like(inputs_embeds)
        completed_blocks_norm = _empty_block_cache_like(inputs_embeds)
        partial_block = inputs_embeds

        entropy_lambda = kwargs.pop("entropy_lambda", 0.0)
        entropy_accum = [] if entropy_lambda > 0 else None
        block_rms_eps = model_config.rms_norm_eps
        if self.layers:
            first_layer = self.layers[0]
            _, detected_rms_eps = _direct_rmsnorm_query(first_layer.first_res_proj, first_layer.first_res_norm)
            if detected_rms_eps is not None:
                block_rms_eps = detected_rms_eps

        layer_idx = 0
        while layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            block_span = getattr(layer, "layers_per_block", 1)
            block_layers = list(self.layers[layer_idx : layer_idx + block_span])
            precomputed_inter = _precompute_interblock_attn(completed_blocks, completed_blocks_norm, block_layers)

            for block_offset, layer in enumerate(block_layers):
                first_inter_stats = None
                second_inter_stats = None
                first_inter_stats, second_inter_stats = precomputed_inter[block_offset]

                if self.gradient_checkpointing and self.training:
                    completed_blocks, completed_blocks_norm, partial_block = self._gradient_checkpointing_func(
                        layer.forward,
                        completed_blocks,
                        completed_blocks_norm,
                        partial_block,
                        causal_mask_mapping[layer.attention_type],
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        first_inter_stats,
                        second_inter_stats,
                        entropy_accum,
                        block_rms_eps,
                    )
                else:
                    completed_blocks, completed_blocks_norm, partial_block = layer(
                        completed_blocks=completed_blocks,
                        completed_blocks_norm=completed_blocks_norm,
                        partial_block=partial_block,
                        attention_mask=causal_mask_mapping[layer.attention_type],
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        entropy_accum=entropy_accum,
                        first_inter_stats=first_inter_stats,
                        second_inter_stats=second_inter_stats,
                        block_rms_eps=block_rms_eps,
                    )
            layer_idx += len(block_layers)

        hidden_states = self.norm(partial_block)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        if entropy_accum:
            outputs.attnres_entropy = torch.stack(entropy_accum).mean()
        else:
            outputs.attnres_entropy = None
        return outputs

    model.model.forward = MethodType(attnres_forward, model.model)


def _rescaled_residual(hidden_states, residual, *, residual_idx: int):
    depth = max(int(residual_idx), 1)
    residual_scale = depth / (depth + 1)
    branch_scale = 1.0 / (depth + 1)
    return residual * residual_scale + hidden_states * branch_scale


def _residual_op_index(model_config: TinyQwen3Config, *, layer_idx: int, branch_idx: int, is_attention_layer: bool) -> int:
    if not model_config.use_mamba3_hybrid:
        return layer_idx * 2 + branch_idx + 1

    residuals_per_mamba_layer = 2 if model_config.mamba3_with_mlp_block else 1
    attention_period = max(int(model_config.mamba3_attention_period), 1)
    total = 0
    for prev_layer_idx in range(layer_idx):
        prev_is_attention = prev_layer_idx % attention_period == attention_period - 1
        total += 2 if prev_is_attention else residuals_per_mamba_layer
    return total + branch_idx + 1


def _patch_qwen3_with_rescaled_residual(model) -> None:
    from transformers.modeling_layers import GradientCheckpointingLayer

    class Qwen3RescaledDecoderLayer(GradientCheckpointingLayer):
        def __init__(self, reference_layer, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx
            self.self_attn = reference_layer.self_attn
            self.mlp = reference_layer.mlp
            self.input_layernorm = reference_layer.input_layernorm
            self.post_attention_layernorm = reference_layer.post_attention_layernorm
            self.attention_type = reference_layer.attention_type

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = _rescaled_residual(
                hidden_states,
                residual,
                residual_idx=_residual_op_index(
                    model.config,
                    layer_idx=self.layer_idx,
                    branch_idx=0,
                    is_attention_layer=True,
                ),
            )

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _rescaled_residual(
                hidden_states,
                residual,
                residual_idx=_residual_op_index(
                    model.config,
                    layer_idx=self.layer_idx,
                    branch_idx=1,
                    is_attention_layer=True,
                ),
            )
            return hidden_states

    for layer_idx, layer in enumerate(model.model.layers):
        model.model.layers[layer_idx] = Qwen3RescaledDecoderLayer(layer, layer_idx)


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
            self.with_mlp_block = model_config.mamba3_with_mlp_block
            if self.with_mlp_block:
                self.post_attention_layernorm = reference_layer.post_attention_layernorm
                self.mlp = reference_layer.mlp

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
            if model_config.use_rescaled_residual:
                hidden_states = _rescaled_residual(
                    hidden_states,
                    residual,
                    residual_idx=_residual_op_index(
                        model_config,
                        layer_idx=self.layer_idx,
                        branch_idx=0,
                        is_attention_layer=False,
                    ),
                )
            else:
                hidden_states = residual + hidden_states
            if self.with_mlp_block:
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                if model_config.use_rescaled_residual:
                    hidden_states = _rescaled_residual(
                        hidden_states,
                        residual,
                        residual_idx=_residual_op_index(
                            model_config,
                            layer_idx=self.layer_idx,
                            branch_idx=1,
                            is_attention_layer=False,
                        ),
                    )
                else:
                    hidden_states = residual + hidden_states
            return hidden_states

    attention_layer_idx = [
        idx
        for idx in range(model_config.num_hidden_layers)
        if idx % model_config.mamba3_attention_period == model_config.mamba3_attention_period - 1
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
    hf_config.use_zero_centered_rmsnorm = model_config.use_zero_centered_rmsnorm
    hf_config.use_rescaled_residual = model_config.use_rescaled_residual
    hf_config.use_attention_residuals = model_config.use_attention_residuals
    hf_config.attention_residual_num_blocks = model_config.attention_residual_num_blocks
    hf_config.attention_residual_recency_bias_init = model_config.attention_residual_recency_bias_init
    hf_config.attention_residual_mode = model_config.attention_residual_mode
    hf_config.attention_residual_gate_type = model_config.attention_residual_gate_type
    hf_config.use_mamba3_hybrid = model_config.use_mamba3_hybrid
    hf_config.mamba3_with_mlp_block = model_config.mamba3_with_mlp_block
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
    if model_config.use_zero_centered_rmsnorm:
        _patch_qwen3_with_zero_centered_rmsnorm(model)
    if model_config.use_mamba3_hybrid:
        _patch_qwen3_with_mamba3_hybrid(model, model_config)
    if model_config.use_attention_residuals:
        _patch_qwen3_with_attention_residuals(model, model_config)
    if model_config.use_rescaled_residual:
        _patch_qwen3_with_rescaled_residual(model)
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
        if getattr(hf_config, "use_zero_centered_rmsnorm", False):
            _patch_qwen3_with_zero_centered_rmsnorm(model)
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
                use_zero_centered_rmsnorm=getattr(hf_config, "use_zero_centered_rmsnorm", False),
                use_rescaled_residual=getattr(hf_config, "use_rescaled_residual", False),
                use_attention_residuals=getattr(hf_config, "use_attention_residuals", False),
                attention_residual_num_blocks=getattr(hf_config, "attention_residual_num_blocks", 8),
                attention_residual_recency_bias_init=getattr(hf_config, "attention_residual_recency_bias_init", 0.0),
                attention_residual_mode=getattr(hf_config, "attention_residual_mode", "block"),
                attention_residual_gate_type=getattr(hf_config, "attention_residual_gate_type", "bias"),
                use_mamba3_hybrid=getattr(hf_config, "use_mamba3_hybrid", False),
                mamba3_with_mlp_block=getattr(hf_config, "mamba3_with_mlp_block", False),
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
            if getattr(hf_config, "use_attention_residuals", False):
                _patch_qwen3_with_attention_residuals(model, model_config)
        elif getattr(hf_config, "use_attention_residuals", False):
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
                use_zero_centered_rmsnorm=getattr(hf_config, "use_zero_centered_rmsnorm", False),
                use_rescaled_residual=getattr(hf_config, "use_rescaled_residual", False),
                use_attention_residuals=getattr(hf_config, "use_attention_residuals", False),
                attention_residual_num_blocks=getattr(hf_config, "attention_residual_num_blocks", 8),
                attention_residual_recency_bias_init=getattr(hf_config, "attention_residual_recency_bias_init", 0.0),
                attention_residual_mode=getattr(hf_config, "attention_residual_mode", "block"),
                attention_residual_gate_type=getattr(hf_config, "attention_residual_gate_type", "bias"),
            )
            _patch_qwen3_with_attention_residuals(model, model_config)
        if getattr(hf_config, "use_rescaled_residual", False):
            _patch_qwen3_with_rescaled_residual(model)
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
