# coding=utf-8
"""Tests for the MLX (Apple Silicon) Qwen3 forced-aligner backend.

Pure-CPU structural tests always run (guarded by mlx availability). The
end-to-end test runs only when FUNYI_MLX_ALIGNER_MODEL points at a local
forced-aligner checkpoint (e.g. an mlx-community 4bit dir), so CI without
MLX/weights stays green.
"""
from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

from qwen3_asr_runtime.mlx_qwen3_asr.config import MLXQwen3ASRConfig

MODEL = os.environ.get("FUNYI_MLX_ALIGNER_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set FUNYI_MLX_ALIGNER_MODEL to a local forced-aligner checkpoint")


def _aligner_config_dict() -> dict:
    # Mirrors mlx-community/Qwen3-ForcedAligner-0.6B-4bit config.json.
    return {
        "model_type": "qwen3_asr",
        "timestamp_token_id": 151705,
        "timestamp_segment_time": 80,
        "thinker_config": {
            "model_type": "qwen3_forced_aligner",
            "classify_num": 5000,
            "audio_token_id": 151676,
            "audio_config": {
                "num_mel_bins": 128, "d_model": 1024, "encoder_layers": 24,
                "encoder_attention_heads": 16, "encoder_ffn_dim": 4096, "output_dim": 1024,
                "downsample_hidden_size": 480, "n_window": 50, "n_window_infer": 800,
                "max_source_positions": 1500,
            },
            "text_config": {
                "vocab_size": 152064, "hidden_size": 1024, "intermediate_size": 3072,
                "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "rope_theta": 1000000, "rms_norm_eps": 1e-6,
            },
        },
    }


def test_forced_aligner_config_parses_head_and_timestamps():
    cfg = MLXQwen3ASRConfig.from_dict(_aligner_config_dict())
    assert cfg.is_forced_aligner is True
    assert cfg.classify_num == 5000
    assert cfg.timestamp_token_id == 151705
    assert cfg.timestamp_segment_time == 80.0
    assert cfg.audio_token_id == 151676
    assert cfg.audio.d_model == 1024 and cfg.audio.encoder_layers == 24


def test_plain_asr_config_is_not_aligner():
    d = _aligner_config_dict()
    d["thinker_config"]["model_type"] = "qwen3_asr"
    d["thinker_config"].pop("classify_num")
    d.pop("timestamp_token_id")
    d.pop("timestamp_segment_time")
    cfg = MLXQwen3ASRConfig.from_dict(d)
    assert cfg.is_forced_aligner is False
    assert cfg.classify_num is None and cfg.timestamp_token_id is None


@needs_model
def test_end_to_end_align_mlx():
    import numpy as np

    from qwen3_asr_runtime.mlx_forced_aligner import MLXForcedAlignerBackend

    aligner = MLXForcedAlignerBackend.from_pretrained(MODEL, dtype="bfloat16")
    wav = (0.01 * np.sin(2 * np.pi * 220 * np.arange(16000 * 2) / 16000)).astype(np.float32)
    results = aligner.align((wav, 16000), "hello world", "English")
    assert results and len(results) == 1
    for it in results[0].items:
        assert it.end_time >= it.start_time
