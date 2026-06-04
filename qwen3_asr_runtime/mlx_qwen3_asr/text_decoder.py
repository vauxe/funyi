# coding=utf-8
"""MLX text decoder for Qwen3-ASR (mirrors Qwen3ASRThinkerTextModel).

Inference-speed optimizations (standard MLX patterns):
  * a pre-allocated, step-growing KV cache (no per-token concatenate);
  * fused mx.fast.scaled_dot_product_attention (GQA handled natively);
  * plain nn.RoPE(traditional=False) with a cache offset.

The last point is valid because Qwen3-ASR has no separate temporal/height/width
MRoPE grid: get_rope_index assigns the same position to all three rope
dimensions, so the interleaved MRoPE reduces to standard RoPE. Output parity
against the upstream transformers model is verified by tools/parity_mlx_vs_hf.py.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..mlx_common.cache import KVCache, create_additive_causal_mask
from ..mlx_common.layers import RMSNorm, SwiGLUMLP
from .config import MLXTextConfig


class TextAttention(nn.Module):
    def __init__(self, cfg: MLXTextConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.scale = self.head_dim ** -0.5
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        # ASR MRoPE collapses to standard RoPE (identical position per grid dim).
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, cache: Optional[KVCache]):
        b, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, length, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, length, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Pure prefill (offset==0) uses MLX's fused "causal" mask -- bit-identical to the
        # additive mask but avoids materializing an L*L float mask in every layer. The
        # offset>0 multi-token case (streaming prefill with a populated cache) keeps the
        # explicit additive mask; single-token decode needs no mask.
        if length == 1:
            mask = None
        elif offset == 0:
            mask = "causal"
        else:
            mask = create_additive_causal_mask(length, offset).astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: MLXTextConfig):
        super().__init__()
        self.self_attn = TextAttention(cfg)
        self.mlp = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, cache: Optional[KVCache]):
        x = x + self.self_attn(self.input_layernorm(x), cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TextModel(nn.Module):
    def __init__(self, cfg: MLXTextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array, caches: Optional[List[KVCache]]):
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, caches[i] if caches is not None else None)
        return self.norm(h)

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.layers))]
