# coding=utf-8
"""Compute-dtype resolution shared by the MLX model layers."""

from __future__ import annotations

import mlx.core as mx

_DTYPES = {
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
    "float16": mx.float16,
    "fp16": mx.float16,
    "half": mx.float16,
    "float32": mx.float32,
    "fp32": mx.float32,
    "float": mx.float32,
}


def resolve_dtype(name: str) -> mx.Dtype:
    key = str(name).lower().replace("torch.", "").strip()
    if key not in _DTYPES:
        raise ValueError(f"unsupported dtype: {name}")
    return _DTYPES[key]
