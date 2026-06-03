# coding=utf-8
"""Tests for the MLX (Apple Silicon) Qwen3-ASR backend.

Pure-CPU structural tests always run (guarded by mlx availability). The
end-to-end test runs only when FUNYI_MLX_TEST_MODEL points at a local 0.6B
checkpoint, so CI without MLX/weights stays green.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from qwen3_asr_runtime.mlx_qwen3_asr.audio_encoder import AudioEncoder, feat_extract_output_length
from qwen3_asr_runtime.mlx_qwen3_asr.config import MLXQwen3ASRConfig

MODEL = os.environ.get("FUNYI_MLX_TEST_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set FUNYI_MLX_TEST_MODEL to a local 0.6B checkpoint")
MODEL_4BIT = os.environ.get("FUNYI_MLX_4BIT_MODEL")
needs_4bit = pytest.mark.skipif(not MODEL_4BIT, reason="set FUNYI_MLX_4BIT_MODEL to a local 4-bit checkpoint")


def _synthetic_config_dict() -> dict:
    return {
        "model_type": "qwen3_asr",
        "thinker_config": {
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "dtype": "bfloat16",
            "audio_config": {
                "num_mel_bins": 128, "d_model": 896, "encoder_layers": 18,
                "encoder_attention_heads": 14, "encoder_ffn_dim": 3584, "output_dim": 1024,
                "downsample_hidden_size": 480, "n_window": 50, "n_window_infer": 800,
                "max_source_positions": 1500,
            },
            "text_config": {
                "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
                "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "rope_theta": 1000000, "rms_norm_eps": 1e-6,
                "tie_word_embeddings": True,
                "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20]},
            },
        },
    }


def test_config_parses_nested_fields():
    cfg = MLXQwen3ASRConfig.from_dict(_synthetic_config_dict())
    assert cfg.audio_token_id == 151676
    assert cfg.text.num_attention_heads == 16 and cfg.text.num_key_value_heads == 8
    assert cfg.text.head_dim == 128 and cfg.text.rope_theta == 1000000.0
    assert cfg.audio.d_model == 896 and cfg.audio.output_dim == 1024


def test_feat_length_matches_reference_formula():
    from qwen3_asr_runtime.hf_qwen3_asr.processing_qwen3_asr import _get_feat_extract_output_lengths

    for length in [1, 2, 50, 99, 100, 101, 200, 313, 999, 3001]:
        assert feat_extract_output_length(length) == int(_get_feat_extract_output_lengths(length))


def test_block_mask_coverage_and_pattern():
    mask = AudioEncoder._block_mask([2, 3], 5)
    m = np.asarray(mask)[0, 0]
    assert m.shape == (5, 5)
    assert m[0, 1] == 0.0 and m[3, 4] == 0.0  # within-block attends
    assert m[0, 2] < -1e8 and m[2, 0] < -1e8  # across-block blocked


def test_block_mask_rejects_wrong_coverage():
    with pytest.raises(AssertionError):
        AudioEncoder._block_mask([2, 2], 5)


@needs_model
def test_end_to_end_transcribe_mlx():
    from qwen3_asr_runtime.model import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(MODEL, backend="mlx", dtype="bfloat16").eval()
    wav_path = "local_data/e2e_en_espeak_20260522T201810.wav"
    if not os.path.exists(wav_path):
        pytest.skip("validation wav not present")
    out = model.transcribe(wav_path)
    assert out and out[0].text.strip()
    assert out[0].language == "English"


@needs_4bit
def test_quantized_4bit_loads_and_transcribes():
    """Pre-quantized MLX checkpoint: tied LM head + per-module quantize + NHWC conv."""
    from qwen3_asr_runtime.mlx_qwen3_asr import load_mlx_qwen3_asr
    from qwen3_asr_runtime.model import Qwen3ASRModel

    _, cfg = load_mlx_qwen3_asr(MODEL_4BIT, dtype="bfloat16")
    assert cfg.quantization and int(cfg.quantization["bits"]) == 4

    model = Qwen3ASRModel.from_pretrained(MODEL_4BIT, backend="mlx", dtype="bfloat16").eval()
    wav_path = "local_data/e2e_en_espeak_20260522T201810.wav"
    if not os.path.exists(wav_path):
        pytest.skip("validation wav not present")
    out = model.transcribe(wav_path)
    assert out and "transcription" in out[0].text.lower()
