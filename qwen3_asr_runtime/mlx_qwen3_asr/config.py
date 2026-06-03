# coding=utf-8
"""Plain-dataclass config parsed directly from a checkpoint ``config.json``.

Independent of ``transformers.PretrainedConfig`` so the MLX model layer stays
torch-free. Holds only the fields the model layer reads; nothing is hardcoded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MLXTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool

    @classmethod
    def from_dict(cls, d: dict) -> "MLXTextConfig":
        head_dim = d.get("head_dim") or (d["hidden_size"] // d["num_attention_heads"])
        return cls(
            vocab_size=int(d["vocab_size"]),
            hidden_size=int(d["hidden_size"]),
            intermediate_size=int(d["intermediate_size"]),
            num_hidden_layers=int(d["num_hidden_layers"]),
            num_attention_heads=int(d["num_attention_heads"]),
            num_key_value_heads=int(d.get("num_key_value_heads", d["num_attention_heads"])),
            head_dim=int(head_dim),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            rope_theta=float(d.get("rope_theta", 1e6)),
            attention_bias=bool(d.get("attention_bias", False)),
        )


@dataclass
class MLXAudioConfig:
    num_mel_bins: int
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    output_dim: int
    downsample_hidden_size: int
    n_window: int
    n_window_infer: int
    conv_chunksize: int
    max_source_positions: int

    @classmethod
    def from_dict(cls, d: dict) -> "MLXAudioConfig":
        return cls(
            num_mel_bins=int(d.get("num_mel_bins", 128)),
            d_model=int(d["d_model"]),
            encoder_layers=int(d.get("encoder_layers", d.get("num_hidden_layers"))),
            encoder_attention_heads=int(d["encoder_attention_heads"]),
            encoder_ffn_dim=int(d["encoder_ffn_dim"]),
            output_dim=int(d["output_dim"]),
            downsample_hidden_size=int(d.get("downsample_hidden_size", 480)),
            n_window=int(d.get("n_window", 50)),
            n_window_infer=int(d.get("n_window_infer", 800)),
            conv_chunksize=int(d.get("conv_chunksize", 500)),
            max_source_positions=int(d.get("max_source_positions", 1500)),
        )


@dataclass
class MLXQwen3ASRConfig:
    text: MLXTextConfig
    audio: MLXAudioConfig
    audio_token_id: int
    eos_token_ids: List[int] = field(default_factory=lambda: [151645, 151643])
    quantization: Optional[dict] = None  # {"group_size", "bits"} for pre-quantized checkpoints

    @classmethod
    def from_dict(cls, raw: dict, generation: Optional[dict] = None) -> "MLXQwen3ASRConfig":
        thinker = raw.get("thinker_config", raw)
        text = MLXTextConfig.from_dict(thinker["text_config"])
        audio = MLXAudioConfig.from_dict(thinker["audio_config"])
        audio_token_id = thinker.get("audio_token_id", raw.get("audio_token_id", 151646))

        eos = [151645, 151643]
        if generation:
            g_eos = generation.get("eos_token_id")
            if isinstance(g_eos, int):
                eos = [g_eos]
            elif isinstance(g_eos, (list, tuple)) and g_eos:
                eos = [int(x) for x in g_eos]

        return cls(
            text=text,
            audio=audio,
            audio_token_id=int(audio_token_id),
            eos_token_ids=eos,
            quantization=raw.get("quantization"),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "MLXQwen3ASRConfig":
        d = Path(model_dir)
        raw = json.loads((d / "config.json").read_text())
        gen: Optional[dict] = None
        gen_path = d / "generation_config.json"
        if gen_path.exists():
            gen = json.loads(gen_path.read_text())
        return cls.from_dict(raw, gen)
