# coding=utf-8
"""Tests for the MLX (Apple Silicon) Hunyuan / HY-MT translation backend.

Pure-CPU structural tests always run (guarded by mlx availability). The
end-to-end test runs only when FUNYI_MLX_HYMT_MODEL points at a local HY-MT
checkpoint (e.g. an mlx-community 4bit dir), so CI without MLX/weights stays
green.
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

from qwen3_asr_runtime.mlx_hunyuan.config import MLXHunyuanConfig, _effective_rope_base

MODEL = os.environ.get("FUNYI_MLX_HYMT_MODEL")
needs_model = pytest.mark.skipif(
    not MODEL, reason="set FUNYI_MLX_HYMT_MODEL to a local HY-MT checkpoint"
)


def _hymt_config_dict() -> dict:
    # Mirrors tencent/Hy-MT2-1.8B / mlx-community/Hy-MT2-1.8B-4bit config.json.
    return {
        "model_type": "hunyuan_v1_dense",
        "vocab_size": 120818,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "dynamic",
            "alpha": 1000.0,
            "factor": 1.0,
            "mscale": 1.0,
        },
        "use_qk_norm": True,
        "attention_bias": False,
        "tie_word_embeddings": True,
        "eos_token_id": 120020,
    }


def test_config_parses_nested_fields():
    cfg = MLXHunyuanConfig.from_dict(_hymt_config_dict())
    assert (
        cfg.hidden_size == 2048
        and cfg.num_attention_heads == 16
        and cfg.num_key_value_heads == 4
    )
    assert (
        cfg.head_dim == 128
        and cfg.intermediate_size == 6144
        and cfg.num_hidden_layers == 32
    )
    assert cfg.use_qk_norm is True and cfg.tie_word_embeddings is True
    assert cfg.eos_token_ids == [120020]


def test_effective_rope_base_uses_alpha():
    # HunYuan DynamicNTKAlpha: base = rope_theta * alpha**(head_dim/(head_dim-2)).
    base = _effective_rope_base(10000.0, 128, {"type": "dynamic", "alpha": 1000.0})
    assert base == pytest.approx(10000.0 * 1000.0 ** (128 / 126), rel=1e-9)
    assert base == pytest.approx(11158839.925, rel=1e-6)
    # No scaling -> plain theta.
    assert _effective_rope_base(10000.0, 128, None) == 10000.0
    # Non-dynamic / no alpha -> plain theta.
    assert (
        _effective_rope_base(10000.0, 128, {"type": "linear", "factor": 2.0}) == 10000.0
    )


def test_head_dim_falls_back_to_hidden_over_heads():
    d = dict(_hymt_config_dict())
    d.pop("head_dim")
    d.pop("attention_head_dim", None)
    cfg = MLXHunyuanConfig.from_dict(d)
    assert cfg.head_dim == 2048 // 16


@needs_model
def test_end_to_end_translate_mlx():
    from qwen3_asr_runtime.mlx_translation import MLXHYMTTranslator

    translator = MLXHYMTTranslator(MODEL, dtype="bfloat16")
    out = translator.translate(
        "Hello, world.", target_language="Chinese", source_language="English"
    )
    assert isinstance(out, str) and out.strip()
