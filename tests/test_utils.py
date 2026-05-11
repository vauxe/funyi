# coding=utf-8
from __future__ import annotations

import unittest

import numpy as np

from qwen3_asr_runtime.utils import float_range_normalize


class FloatRangeNormalizeTest(unittest.TestCase):
    def test_float32_unit_range_audio_is_unchanged(self) -> None:
        audio = np.array([-1.0, -0.25, 0.0, 0.75, 1.0], dtype=np.float32)

        normalized = float_range_normalize(audio)

        np.testing.assert_array_equal(normalized, audio)

    def test_out_of_range_audio_is_scaled_into_unit_range(self) -> None:
        audio = np.array([-2.0, 0.0, 1.0], dtype=np.float32)

        normalized = float_range_normalize(audio)

        np.testing.assert_allclose(normalized, np.array([-1.0, 0.0, 0.5], dtype=np.float32))

    def test_integer_audio_is_converted_to_float32(self) -> None:
        audio = np.array([-2, 0, 1], dtype=np.int16)

        normalized = float_range_normalize(audio)

        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized, np.array([-1.0, 0.0, 0.5], dtype=np.float32))

    def test_nonfinite_peak_keeps_previous_edge_case_semantics(self) -> None:
        with np.errstate(invalid="ignore"):
            inf_normalized = float_range_normalize(np.array([np.inf, 1.0], dtype=np.float32))
        nan_normalized = float_range_normalize(np.array([np.nan, 2.0], dtype=np.float32))

        self.assertTrue(np.isnan(inf_normalized[0]))
        self.assertEqual(inf_normalized[1], 0.0)
        self.assertTrue(np.isnan(nan_normalized[0]))
        self.assertEqual(nan_normalized[1], 1.0)


if __name__ == "__main__":
    unittest.main()
