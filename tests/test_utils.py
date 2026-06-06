# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from qwen3_asr_runtime.utils import float_range_normalize


class TestFloatRangeNormalize:
    @pytest.mark.parametrize(
        ("audio", "expected"),
        [
            (
                np.array([-1.0, -0.25, 0.0, 0.75, 1.0], dtype=np.float32),
                np.array([-1.0, -0.25, 0.0, 0.75, 1.0], dtype=np.float32),
            ),
            (
                np.array([-2.0, 0.0, 1.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.5], dtype=np.float32),
            ),
            (
                np.array([-2, 0, 1], dtype=np.int16),
                np.array([-1.0, 0.0, 0.5], dtype=np.float32),
            ),
        ],
    )
    def test_finite_audio_is_normalized_to_float32_unit_range(
        self,
        audio: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        original = audio.copy()

        normalized = float_range_normalize(audio)

        assert normalized.dtype == np.float32
        np.testing.assert_allclose(normalized, expected)
        if audio.dtype == np.float32 and np.max(np.abs(original)) <= 1.0:
            np.testing.assert_array_equal(normalized, audio)

    def test_nonfinite_peak_keeps_previous_edge_case_semantics(self) -> None:
        with np.errstate(invalid="ignore"):
            inf_normalized = float_range_normalize(
                np.array([np.inf, 1.0], dtype=np.float32)
            )
        nan_normalized = float_range_normalize(
            np.array([np.nan, 2.0], dtype=np.float32)
        )

        assert np.isnan(inf_normalized[0])
        assert inf_normalized[1] == 0.0
        assert np.isnan(nan_normalized[0])
        assert nan_normalized[1] == 1.0
