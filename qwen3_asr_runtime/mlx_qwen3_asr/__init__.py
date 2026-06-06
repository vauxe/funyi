# coding=utf-8
"""Ground-up MLX (Apple Silicon) reimplementation of the Qwen3-ASR forward pass.

This package does not depend on the upstream `transformers` model code at
runtime. Shapes are read from the checkpoint ``config.json`` (never the config
class defaults, which differ from the actual checkpoints). Quality is gated by
CER against an official-code golden, not by byte-for-byte parity -- see
``AGENTS.md``.

The model layer mirrors ``qwen3_asr_runtime/hf_qwen3_asr/modeling_qwen3_asr.py``
and is gated for correctness against that upstream model
(``tools/parity_mlx_vs_hf.py``). The inference-speed optimizations use standard
MLX patterns (pre-allocated step-growing KV cache, fused kernels).
"""

from __future__ import annotations

from .config import MLXQwen3ASRConfig
from .model import MLXQwen3ASRForConditionalGeneration, load_mlx_qwen3_asr

__all__ = [
    "MLXQwen3ASRConfig",
    "MLXQwen3ASRForConditionalGeneration",
    "load_mlx_qwen3_asr",
]
