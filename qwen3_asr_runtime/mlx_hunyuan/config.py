# coding=utf-8
"""Dataclass config for the MLX Hunyuan dense v1 decoder.

The non-obvious field is ``rope_base``: Hunyuan dynamic RoPE uses
``rope_theta * alpha ** (head_dim / (head_dim - 2))`` instead of raw
``rope_theta``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _effective_rope_base(rope_theta: float, head_dim: int, rope_scaling: Optional[dict]) -> float:
    if not rope_scaling:
        return float(rope_theta)
    rope_type = str(rope_scaling.get("type") or rope_scaling.get("rope_type") or "default").lower()
    alpha = rope_scaling.get("alpha")
    if rope_type == "dynamic" and alpha:
        return float(rope_theta) * float(alpha) ** (head_dim / (head_dim - 2))
    return float(rope_theta)


@dataclass
class MLXHunyuanConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_base: float
    tie_word_embeddings: bool
    use_qk_norm: bool
    attention_bias: bool
    eos_token_ids: List[int] = field(default_factory=lambda: [120020])
    quantization: Optional[dict] = None  # {"group_size", "bits"} for pre-quantized checkpoints

    @classmethod
    def from_dict(cls, raw: dict, generation: Optional[dict] = None) -> "MLXHunyuanConfig":
        head_dim = int(raw.get("head_dim") or raw.get("attention_head_dim") or (raw["hidden_size"] // raw["num_attention_heads"]))
        rope_base = _effective_rope_base(
            float(raw.get("rope_theta", 10000.0)), head_dim, raw.get("rope_scaling")
        )

        eos = _resolve_eos(raw.get("eos_token_id"))
        if generation and generation.get("eos_token_id") is not None:
            eos = _resolve_eos(generation.get("eos_token_id"))

        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw.get("num_key_value_heads", raw["num_attention_heads"])),
            head_dim=head_dim,
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_base=rope_base,
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            use_qk_norm=bool(raw.get("use_qk_norm", False)),
            attention_bias=bool(raw.get("attention_bias", False)),
            eos_token_ids=eos,
            quantization=raw.get("quantization"),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "MLXHunyuanConfig":
        d = Path(model_dir)
        raw = json.loads((d / "config.json").read_text())
        gen: Optional[dict] = None
        gen_path = d / "generation_config.json"
        if gen_path.exists():
            gen = json.loads(gen_path.read_text())
        return cls.from_dict(raw, gen)


def _resolve_eos(value: object) -> List[int]:
    if value is None:
        return [120020]
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)) and value:
        return [int(x) for x in value]
    return [120020]
