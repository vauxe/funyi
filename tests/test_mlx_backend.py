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

from qwen3_asr_runtime.mlx_qwen3_asr.audio_encoder import (
    AudioEncoder,
    feat_extract_output_length,
)
from qwen3_asr_runtime.mlx_qwen3_asr.config import MLXQwen3ASRConfig
from qwen3_asr_runtime.mlx_common.weights import load_weights_dict

MODEL = os.environ.get("FUNYI_MLX_TEST_MODEL")
needs_model = pytest.mark.skipif(
    not MODEL, reason="set FUNYI_MLX_TEST_MODEL to a local 0.6B checkpoint"
)
MODEL_4BIT = os.environ.get("FUNYI_MLX_4BIT_MODEL")
needs_4bit = pytest.mark.skipif(
    not MODEL_4BIT, reason="set FUNYI_MLX_4BIT_MODEL to a local 4-bit checkpoint"
)


def _synthetic_config_dict() -> dict:
    return {
        "model_type": "qwen3_asr",
        "thinker_config": {
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "dtype": "bfloat16",
            "audio_config": {
                "num_mel_bins": 128,
                "d_model": 896,
                "encoder_layers": 18,
                "encoder_attention_heads": 14,
                "encoder_ffn_dim": 3584,
                "output_dim": 1024,
                "downsample_hidden_size": 480,
                "n_window": 50,
                "n_window_infer": 800,
                "max_source_positions": 1500,
            },
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rope_theta": 1000000,
                "rms_norm_eps": 1e-6,
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
    from qwen3_asr_runtime.hf_qwen3_asr.processing_qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    for length in [1, 2, 50, 99, 100, 101, 200, 313, 999, 3001]:
        assert feat_extract_output_length(length) == int(
            _get_feat_extract_output_lengths(length)
        )


def test_block_mask_coverage_and_pattern():
    mask = AudioEncoder._block_mask([2, 3], 5)
    m = np.asarray(mask)[0, 0]
    assert m.shape == (5, 5)
    assert m[0, 1] == 0.0 and m[3, 4] == 0.0  # within-block attends
    assert m[0, 2] < -1e8 and m[2, 0] < -1e8  # across-block blocked


def test_block_mask_rejects_wrong_coverage():
    with pytest.raises(AssertionError):
        AudioEncoder._block_mask([2, 2], 5)


def test_load_weights_dict_merges_sharded_safetensors(tmp_path):
    mx.save_safetensors(
        str(tmp_path / "model-00001-of-00002.safetensors"), {"b.weight": mx.array([2])}
    )
    mx.save_safetensors(
        str(tmp_path / "model-00002-of-00002.safetensors"), {"a.weight": mx.array([1])}
    )

    weights = load_weights_dict(str(tmp_path))

    assert sorted(weights) == ["a.weight", "b.weight"]
    assert int(weights["a.weight"][0].item()) == 1
    assert int(weights["b.weight"][0].item()) == 2


def _tiny_asr_model():
    """Small random-weight model: enough to exercise generate/draft/align paths."""
    from qwen3_asr_runtime.mlx_qwen3_asr.model import (
        MLXQwen3ASRForConditionalGeneration,
    )

    mx.random.seed(0)
    cfg = MLXQwen3ASRConfig.from_dict(
        {
            "model_type": "qwen3_asr",
            "thinker_config": {
                "audio_token_id": 120,
                "audio_config": {
                    "num_mel_bins": 8,
                    "d_model": 32,
                    "encoder_layers": 1,
                    "encoder_attention_heads": 2,
                    "encoder_ffn_dim": 64,
                    "output_dim": 48,
                    "downsample_hidden_size": 16,
                    "n_window": 50,
                    "n_window_infer": 800,
                    "max_source_positions": 100,
                },
                "text_config": {
                    "vocab_size": 128,
                    "hidden_size": 48,
                    "intermediate_size": 96,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 24,
                    "rope_theta": 10000,
                    "rms_norm_eps": 1e-6,
                },
            },
        }
    )
    model = MLXQwen3ASRForConditionalGeneration(cfg, tie_lm_head=False)
    mx.eval(model.parameters())
    return model, cfg


def test_kv_cache_crop_rewinds_offset():
    from qwen3_asr_runtime.mlx_common.cache import KVCache

    cache = KVCache(step=4)
    k = mx.random.normal((1, 1, 10, 3))
    v = mx.random.normal((1, 1, 10, 3))
    cache.update_and_fetch(k, v)
    assert cache.offset == 10

    cache.crop(6)
    assert cache.offset == 6
    k1 = mx.random.normal((1, 1, 1, 3))
    v1 = mx.random.normal((1, 1, 1, 3))
    keys, values = cache.update_and_fetch(k1, v1)
    assert keys.shape[2] == 7 and values.shape[2] == 7
    assert mx.allclose(keys[..., :6, :], k[..., :6, :]).item()
    assert mx.allclose(keys[..., 6:, :], k1).item()
    cache.crop(100)  # crop never grows the offset
    assert cache.offset == 7

    # Riskiest branch: crop to a non-step-aligned offset, then force a
    # reallocation -- update_and_fetch must trim the stale region before
    # concatenating new blocks so cropped-away entries never resurface.
    cache.crop(6)
    k2 = mx.random.normal((1, 1, 8, 3))
    v2 = mx.random.normal((1, 1, 8, 3))
    keys, values = cache.update_and_fetch(k2, v2)
    assert cache.offset == 14
    assert keys.shape[2] == 14 and values.shape[2] == 14
    assert mx.allclose(keys[..., :6, :], k[..., :6, :]).item()
    assert mx.allclose(keys[..., 6:, :], k2).item()


def test_generate_with_draft_invariants():
    model, cfg = _tiny_asr_model()
    ids = mx.array([[3, 14, 15, 92, 65, 35]], dtype=mx.int32)

    plain = model.generate(ids, max_new_tokens=8, eos_token_ids=cfg.eos_token_ids)
    assert len(plain) == 8  # eos ids sit outside the tiny vocab

    # A fully rejected draft must reduce to exactly the plain greedy output.
    garbage = [(t + 1) % 128 for t in plain[:5]]
    stats: dict = {}
    rejected = model.generate_with_draft(
        ids,
        draft_ids=garbage,
        max_new_tokens=8,
        eos_token_ids=cfg.eos_token_ids,
        stats=stats,
    )
    assert stats == {"draft_tokens": 5, "accepted_tokens": 0}
    assert rejected == plain

    # A true-continuation draft is accepted and leaves the output unchanged.
    stats = {}
    accepted = model.generate_with_draft(
        ids,
        draft_ids=plain[:5],
        max_new_tokens=8,
        eos_token_ids=cfg.eos_token_ids,
        stats=stats,
    )
    assert stats["accepted_tokens"] == 5
    assert accepted == plain

    # A draft longer than the budget is clipped before verification.
    stats = {}
    clipped = model.generate_with_draft(
        ids,
        draft_ids=plain,
        max_new_tokens=3,
        eos_token_ids=cfg.eos_token_ids,
        stats=stats,
    )
    assert stats["draft_tokens"] == 3
    assert len(clipped) <= 3

    with pytest.raises(ValueError):
        model.generate_with_draft(
            ids, draft_ids=[], max_new_tokens=8, eos_token_ids=cfg.eos_token_ids
        )

    # Zero budget returns [] before the draft is validated (CUDA-path parity).
    assert (
        model.generate_with_draft(
            ids, draft_ids=plain[:5], max_new_tokens=0, eos_token_ids=cfg.eos_token_ids
        )
        == []
    )


def test_align_logits_positions_matches_full_sequence():
    model, cfg = _tiny_asr_model()
    n_audio = feat_extract_output_length(100)
    ids = [1, 2] + [cfg.audio_token_id] * n_audio + [7, 8, 9]
    input_ids = mx.array([ids], dtype=mx.int32)
    feats = mx.random.normal((1, 8, 100)).astype(mx.float32)

    full = model.align_logits(input_ids, feats, [100])
    pos = [2, len(ids) - 2, len(ids) - 1]
    subset = model.align_logits(input_ids, feats, [100], positions=pos)

    assert subset.shape == (1, len(pos), full.shape[-1])
    assert mx.allclose(full[:, mx.array(pos), :], subset, atol=1e-5).item()


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

    model = Qwen3ASRModel.from_pretrained(
        MODEL_4BIT, backend="mlx", dtype="bfloat16"
    ).eval()
    wav_path = "local_data/e2e_en_espeak_20260522T201810.wav"
    if not os.path.exists(wav_path):
        pytest.skip("validation wav not present")
    out = model.transcribe(wav_path)
    assert out and "transcription" in out[0].text.lower()
