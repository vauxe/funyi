# coding=utf-8
"""Shared MLX decode primitives: a step-growing KV cache and the causal mask."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

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
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
