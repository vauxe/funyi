# coding=utf-8
"""
Fuse q/k/v linears and gate/up linears into single matmuls per layer.

Matmul collapse is exact at fp32 level - the only reason numerics may drift
is that cuBLAS/CUTLASS may pick a different kernel variant for a larger
matmul. In practice bf16 deltas on qkv and gate/up fusions are zero on the
local decode-shape validation set.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _fuse_linears(*linears: nn.Linear) -> nn.Linear:
    """Concatenate N linears with the same input size along the output dim."""
    assert len(linears) >= 2
    in_features = linears[0].in_features
    dtype = linears[0].weight.dtype
    device = linears[0].weight.device
    bias = linears[0].bias is not None
    for lin in linears[1:]:
        assert lin.in_features == in_features
        assert (lin.bias is not None) == bias

    out_features = sum(lin.out_features for lin in linears)
    fused = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        offset = 0
        for lin in linears:
            n = lin.out_features
            fused.weight[offset : offset + n].copy_(lin.weight)
            if bias:
                fused.bias[offset : offset + n].copy_(lin.bias)
            offset += n
    return fused


def _make_fused_attention_forward(
    self_attn: nn.Module,
    q_size: int,
    kv_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Build a replacement forward that uses self.qkv_proj instead of q/k/v."""
    config = self_attn.config
    layer_idx = self_attn.layer_idx
    scaling = self_attn.scaling

    def fused_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,  # accepted but not used (rope computed externally)
        use_cache=False,
        **kwargs,
    ):
        # Single fused matmul -> split into Q/K/V
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from qwen3_asr_runtime.hf_qwen3_asr.modeling_qwen3_asr import (
            eager_attention_forward,
            apply_rotary_pos_emb,
        )

        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q_view = q.view(*input_shape, num_q_heads, head_dim)
        k_view = k.view(*input_shape, num_kv_heads, head_dim)
        v_view = v.view(*input_shape, num_kv_heads, head_dim)

        query_states = self.q_norm(q_view).transpose(1, 2)
        key_states = self.k_norm(k_view).transpose(1, 2)
        value_states = v_view.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return fused_forward


def _make_fused_mlp_forward(mlp: nn.Module):
    def fused_forward(self, x):
        gu = self.gate_up_proj(x)
        gate, up = gu.split([self.intermediate_size, self.intermediate_size], dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    return fused_forward


def patch_model_fused_linears(model) -> dict:
    """Fuse q/k/v into qkv_proj and gate/up into gate_up_proj on every text layer.

    Returns a summary dict with counts.
    """
    # Find the text model layers (thinker.model.layers)
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        raise RuntimeError("model has no `thinker` attribute; cannot fuse")
    layers = thinker.model.layers
    attn_count = 0
    mlp_count = 0
    for layer in layers:
        # self_attn: fuse q/k/v
        attn = layer.self_attn
        if not hasattr(attn, "qkv_proj"):
            head_dim = attn.head_dim
            num_q_heads = attn.q_proj.out_features // head_dim
            num_kv_heads = attn.k_proj.out_features // head_dim
            q_size = num_q_heads * head_dim
            kv_size = num_kv_heads * head_dim
            attn.qkv_proj = _fuse_linears(attn.q_proj, attn.k_proj, attn.v_proj)
            # drop the originals to free memory
            del attn.q_proj
            del attn.k_proj
            del attn.v_proj
            fwd = _make_fused_attention_forward(
                attn, q_size, kv_size, num_q_heads, num_kv_heads, head_dim
            )
            attn.forward = fwd.__get__(attn, type(attn))
            attn_count += 1

        mlp = layer.mlp
        if not hasattr(mlp, "gate_up_proj"):
            mlp.gate_up_proj = _fuse_linears(mlp.gate_proj, mlp.up_proj)
            del mlp.gate_proj
            del mlp.up_proj
            mlp.forward = _make_fused_mlp_forward(mlp).__get__(mlp, type(mlp))
            mlp_count += 1

    return {"attn_fused": attn_count, "mlp_fused": mlp_count}


__all__ = ["patch_model_fused_linears"]
