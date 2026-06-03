# coding=utf-8
"""Shared MLX building blocks mirroring modeling_qwen3_asr.py.

RMSNorm keeps the reference's float32 variance accumulation. Attention itself
uses ``mx.fast.scaled_dot_product_attention`` inline in the text/audio modules.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Qwen3 RMSNorm via the fused mx.fast.rms_norm kernel (float32 accumulation).

    Numerically equivalent to the reference (verified token-identical); ~1.23x
    faster decode than a hand-rolled variant because there are ~113 RMSNorm calls
    per token (input/post norms + q/k norms across 28 layers + final norm), so the
    fused kernel saves a large number of launches.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
