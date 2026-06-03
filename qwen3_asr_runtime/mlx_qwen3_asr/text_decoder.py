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

from .config import MLXTextConfig
from .layers import RMSNorm, SwiGLUMLP

NEG_INF = -1e9


def create_additive_causal_mask(seq_len: int, offset: int = 0) -> mx.array:
    """Additive causal mask of shape (seq_len, offset+seq_len) in float32."""
    rinds = mx.arange(offset + seq_len)
    linds = mx.arange(offset, offset + seq_len) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return (mask * NEG_INF).astype(mx.float32)


class KVCache:
    """Pre-allocated KV cache that grows in fixed steps (mlx-lm style).

    Avoids reallocating the whole cache on every decoded token; only every
    ``step`` tokens does it allocate a new block.
    """

    def __init__(self, step: int = 256) -> None:
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        added = keys.shape[2]
        if self.keys is None or (prev + added) > self.keys.shape[2]:
            b, n_kv, _, head_dim = keys.shape
            v_dim = values.shape[3]
            n_steps = (self.step + added - 1) // self.step
            new_k = mx.zeros((b, n_kv, n_steps * self.step, head_dim), keys.dtype)
            new_v = mx.zeros((b, n_kv, n_steps * self.step, v_dim), values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        self.offset += added
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


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

        mask = create_additive_causal_mask(length, offset).astype(q.dtype) if length > 1 else None
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
