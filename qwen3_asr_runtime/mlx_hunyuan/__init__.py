# coding=utf-8
"""Ground-up MLX (Apple Silicon) reimplementation of the Hunyuan dense v1
decoder used by the HY-MT translation model (tencent/Hy-MT2-1.8B).

Does not depend on the upstream `transformers` model code at runtime. Shapes are
read from the checkpoint ``config.json``. Quality is gated by chrF against the
stock-transformers golden, not by byte-for-byte parity -- see ``AGENTS.md`` and
``tools/parity_mlx_translation.py``.
"""

from __future__ import annotations

from .config import MLXHunyuanConfig
from .model import MLXHunyuanForCausalLM, load_mlx_hunyuan

__all__ = [
    "MLXHunyuanConfig",
    "MLXHunyuanForCausalLM",
    "load_mlx_hunyuan",
]
