# coding=utf-8
from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from qwen3_asr_runtime.forced_aligner import (
    Qwen3ForceAlignTextProcessor,
    Qwen3ForcedAlignerBackend,
    ForcedAlignTextSegment,
    normalize_forced_align_language,
)
from qwen3_asr_runtime.hf_qwen3_asr.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen3_asr_runtime.hf_qwen3_asr.modeling_qwen3_asr import Qwen3ASRAudioEncoder


TIMESTAMP_TOKEN_ID = 151705


def _reference_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _reference_pack_audio(
    encoder: Qwen3ASRAudioEncoder,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_width = encoder.n_window * 2
    chunk_num = torch.ceil(feature_lens / chunk_width).long()
    chunk_lengths = torch.tensor(
        [chunk_width] * int(chunk_num.sum().item()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_width
    chunk_lengths[chunk_lengths == 0] = chunk_width

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _reference_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    return padded_feature, padded_mask_after_cnn


def _official_slow_fix_timestamp(values: list[int]) -> list[int]:
    data = list(values)
    n = len(data)
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_length = max(dp)
    max_idx = dp.index(max_length)

    lis_indices = []
    idx = max_idx
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()

    is_normal = [False] * n
    for idx in lis_indices:
        is_normal[idx] = True

    result = data.copy()
    i = 0
    while i < n:
        if is_normal[i]:
            i += 1
            continue
        j = i
        while j < n and not is_normal[j]:
            j += 1

        anomaly_count = j - i
        left_val = None
        for k in range(i - 1, -1, -1):
            if is_normal[k]:
                left_val = result[k]
                break

        right_val = None
        for k in range(j, n):
            if is_normal[k]:
                right_val = result[k]
                break

        if anomaly_count <= 2:
            for k in range(i, j):
                if left_val is None:
                    result[k] = right_val
                elif right_val is None:
                    result[k] = left_val
                else:
                    result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
        elif left_val is not None and right_val is not None:
            step = (right_val - left_val) / (anomaly_count + 1)
            for k in range(i, j):
                result[k] = left_val + step * (k - i + 1)
        elif left_val is not None:
            for k in range(i, j):
                result[k] = left_val
        elif right_val is not None:
            for k in range(i, j):
                result[k] = right_val

        i = j

    return [int(res) for res in result]


class FakeProcessor:
    def __init__(self, *, emit_timestamps: bool = True) -> None:
        self.emit_timestamps = emit_timestamps

    def __call__(
        self,
        *,
        text: list[str],
        audio: list[np.ndarray],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        rows: list[list[int]] = []
        for prompt in text:
            count = prompt.count("<timestamp>")
            row: list[int] = [11]
            if self.emit_timestamps:
                for idx in range(count):
                    row.extend([TIMESTAMP_TOKEN_ID, 20 + idx])
            rows.append(row)
        max_len = max(len(row) for row in rows)
        padded = [row + [0] * (max_len - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class FakeThinker:
    def __init__(self, timestamp_classes: list[int]) -> None:
        self.timestamp_classes = timestamp_classes
        self.calls = 0
        self.use_cache_values: list[object] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        use_cache: object = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        self.calls += 1
        self.use_cache_values.append(use_cache)
        vocab_size = max(8, max(self.timestamp_classes) + 1)
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab_size), -1000.0)
        logits[:, :, 7] = 999.0
        for row in range(input_ids.shape[0]):
            timestamp_positions = torch.nonzero(
                input_ids[row] == TIMESTAMP_TOKEN_ID,
                as_tuple=False,
            ).flatten()
            for idx, pos in enumerate(timestamp_positions):
                logits[row, pos, :] = -1000.0
                logits[row, pos, self.timestamp_classes[idx % len(self.timestamp_classes)]] = 1000.0
        return SimpleNamespace(logits=logits)


class FakeModel(torch.nn.Module):
    def __init__(self, timestamp_classes: list[int]) -> None:
        super().__init__()
        self.config = SimpleNamespace(timestamp_token_id=TIMESTAMP_TOKEN_ID, timestamp_segment_time=80)
        self.thinker = FakeThinker(timestamp_classes)


class ForceAlignTextProcessorTest(unittest.TestCase):
    def test_chinese_text_is_split_into_alignable_units(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()

        words, prompt = processor.encode_timestamp("现在AI，可以", "Chinese")

        self.assertEqual(words, ["现", "在", "AI", "可", "以"])
        self.assertTrue(prompt.startswith("<|audio_start|><|audio_pad|><|audio_end|>"))
        self.assertEqual(prompt.count("<timestamp>"), 10)

    def test_empty_text_keeps_official_timestamp_prompt(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()

        words, prompt = processor.encode_timestamp("", "Chinese")

        self.assertEqual(words, [])
        self.assertEqual(prompt, "<|audio_start|><|audio_pad|><|audio_end|><timestamp><timestamp>")

    def test_space_language_removes_punctuation_without_losing_words(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()

        words, _ = processor.encode_timestamp("hello, world! it's 2026.", "English")

        self.assertEqual(words, ["hello", "world", "it's", "2026"])

    def test_timestamp_repair_preserves_count_and_monotonicity(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()

        fixed = processor.fix_timestamp([0, 160, 80, 400, 320, 560])

        self.assertEqual(len(fixed), 6)
        self.assertTrue(all(left <= right for left, right in zip(fixed, fixed[1:])))
        for start, end in zip(fixed[0::2], fixed[1::2]):
            self.assertLessEqual(start, end)

    def test_timestamp_repair_leaves_monotonic_values_unchanged(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()

        fixed = processor.fix_timestamp(torch.tensor([0, 80, 160, 240]))

        self.assertEqual(fixed, [0, 80, 160, 240])

    def test_timestamp_repair_matches_official_tie_breaking(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()
        cases = [
            [0, 160, 80, 400, 320, 560],
            [0, 400, 120, 80, 700, 600, 900],
            [3, 1, 2, 2, 1, 4, 3, 5],
            [5, 5, 4, 4, 6, 6, 1, 7],
        ]

        for values in cases:
            with self.subTest(values=values):
                self.assertEqual(processor.fix_timestamp(values), _official_slow_fix_timestamp(values))

    def test_timestamp_repair_matches_official_on_seeded_sequences(self) -> None:
        processor = Qwen3ForceAlignTextProcessor()
        rng = np.random.default_rng(1234)

        for length in [2, 3, 8, 31, 128]:
            for _ in range(20):
                values = rng.integers(0, 20, size=length).tolist()
                with self.subTest(length=length, values=values):
                    self.assertEqual(processor.fix_timestamp(values), _official_slow_fix_timestamp(values))

    def test_language_validation_uses_aligner_language_set(self) -> None:
        self.assertEqual(normalize_forced_align_language("cHINese"), "Chinese")
        with self.assertRaises(ValueError):
            normalize_forced_align_language("Arabic")


class ForcedAlignerBackendTest(unittest.TestCase):
    def test_backend_aligns_timestamp_positions_with_official_call_shape(self) -> None:
        model = FakeModel(timestamp_classes=[0, 2, 3, 5])
        backend = Qwen3ForcedAlignerBackend(model=model, processor=FakeProcessor())

        result = backend.align(audio=(np.zeros(16000, dtype=np.float32), 16000), text="现在", language="Chinese")[0]

        self.assertEqual(
            [(item.text, item.start_time, item.end_time) for item in result.items],
            [
                ("现", 0.0, 0.16),
                ("在", 0.24, 0.4),
            ],
        )
        self.assertEqual(model.thinker.calls, 1)
        self.assertEqual(model.thinker.use_cache_values, [None])

    def test_empty_text_still_runs_official_model_path(self) -> None:
        model = FakeModel(timestamp_classes=[0])
        backend = Qwen3ForcedAlignerBackend(model=model, processor=FakeProcessor())

        result = backend.align(audio=(np.zeros(16000, dtype=np.float32), 16000), text="", language="Chinese")[0]

        self.assertEqual(result.items, [])
        self.assertEqual(model.thinker.calls, 1)

    def test_core_alignment_does_not_clamp_to_audio_duration(self) -> None:
        model = FakeModel(timestamp_classes=[0, 20])
        backend = Qwen3ForcedAlignerBackend(model=model, processor=FakeProcessor())

        result = backend.align(audio=(np.zeros(1600, dtype=np.float32), 16000), text="好", language="Chinese")[0]

        self.assertEqual([(item.text, item.start_time, item.end_time) for item in result.items], [("好", 0.0, 1.6)])

    def test_batch_alignment_preserves_output_order(self) -> None:
        backend = Qwen3ForcedAlignerBackend(model=FakeModel(timestamp_classes=[0, 2, 3, 5]), processor=FakeProcessor())

        results = backend.align(
            audio=[
                (np.zeros(16000 * 4, dtype=np.float32), 16000),
                (np.zeros(16000, dtype=np.float32), 16000),
                (np.zeros(16000 * 2, dtype=np.float32), 16000),
            ],
            text=["现在", "好", "可以"],
            language="Chinese",
        )

        self.assertEqual(["".join(item.text for item in result.items) for result in results], ["现在", "好", "可以"])

    def test_transcript_window_alignment_preserves_segment_order_and_offsets(self) -> None:
        backend = Qwen3ForcedAlignerBackend(
            model=FakeModel(timestamp_classes=[0, 2, 3, 5, 5, 7]),
            processor=FakeProcessor(),
        )

        results = backend.align_transcript_segments(
            audio=(np.zeros(16000 * 10, dtype=np.float32), 16000),
            segments=[
                ForcedAlignTextSegment("现在", 4.0, 5.0),
                ForcedAlignTextSegment("好", 1.0, 2.0),
                ForcedAlignTextSegment("可以", 6.0, 7.0),
            ],
            language="Chinese",
            window_sec=4.0,
            pad_sec=0.0,
        )

        self.assertEqual([result.text if result else None for result in results], ["现在", "好", "可以"])
        self.assertEqual([(result.start_time, result.end_time) if result else None for result in results], [
            (1.24, 1.56),
            (1.0, 1.16),
            (6.0, 6.4),
        ])

    def test_transcript_window_alignment_preserves_space_language_boundaries(self) -> None:
        backend = Qwen3ForcedAlignerBackend(
            model=FakeModel(timestamp_classes=[0, 2, 3, 5]),
            processor=FakeProcessor(),
        )

        results = backend.align_transcript_segments(
            audio=(np.zeros(16000 * 4, dtype=np.float32), 16000),
            segments=[
                ForcedAlignTextSegment("hello", 0.0, 1.0),
                ForcedAlignTextSegment("world", 1.0, 2.0),
            ],
            language="English",
            window_sec=4.0,
        )

        self.assertEqual([result.text if result else None for result in results], ["hello", "world"])

    def test_transcript_window_alignment_validates_window_arguments(self) -> None:
        backend = Qwen3ForcedAlignerBackend(model=FakeModel(timestamp_classes=[0]), processor=FakeProcessor())
        with self.assertRaises(ValueError):
            backend.align_transcript_segments(
                audio=(np.zeros(16000, dtype=np.float32), 16000),
                segments=[ForcedAlignTextSegment("好", 0.0, 1.0)],
                language="Chinese",
                window_sec=0,
            )
        with self.assertRaises(ValueError):
            backend.align_transcript_segments(
                audio=(np.zeros(16000, dtype=np.float32), 16000),
                segments=[ForcedAlignTextSegment("好", 0.0, 1.0)],
                language="Chinese",
                pad_sec=-0.1,
            )
        with self.assertRaises(ValueError):
            backend.align_transcript_segments(
                audio=(np.zeros(16000, dtype=np.float32), 16000),
                segments=[ForcedAlignTextSegment("好", 0.0, 1.0)],
                language="Chinese",
                window_sec=180,
                pad_sec=0.1,
            )

    def test_batch_inputs_must_have_matching_lengths(self) -> None:
        backend = Qwen3ForcedAlignerBackend(model=FakeModel(timestamp_classes=[0]), processor=FakeProcessor())

        with self.assertRaises(ValueError):
            backend.align(
                audio=[
                    (np.zeros(16000, dtype=np.float32), 16000),
                    (np.zeros(16000, dtype=np.float32), 16000),
                ],
                text=["a"],
                language=["English", "English"],
            )

    def test_batch_feature_move_casts_only_floating_tensors(self) -> None:
        backend = Qwen3ForcedAlignerBackend(model=FakeModel(timestamp_classes=[0]), processor=FakeProcessor())
        backend.model.dtype = torch.bfloat16
        inputs = BatchFeature(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "input_features": torch.ones((1, 2), dtype=torch.float32),
            }
        )

        moved = backend._move_inputs_like_official(inputs)

        self.assertEqual(moved["input_ids"].dtype, torch.long)
        self.assertEqual(moved["input_features"].dtype, torch.bfloat16)

    def test_single_full_feature_mask_is_replaced_by_lengths(self) -> None:
        inputs = BatchFeature({"feature_attention_mask": torch.ones((1, 4), dtype=torch.long)})

        Qwen3ForcedAlignerBackend._drop_single_full_feature_mask(inputs)

        self.assertIsNone(inputs["feature_attention_mask"])
        self.assertEqual(inputs["audio_feature_lengths"].tolist(), [4])

    def test_padded_feature_mask_is_preserved(self) -> None:
        inputs = BatchFeature({"feature_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long)})

        Qwen3ForcedAlignerBackend._drop_single_full_feature_mask(inputs)

        self.assertEqual(inputs["feature_attention_mask"].tolist(), [[1, 1, 0]])
        self.assertNotIn("audio_feature_lengths", inputs)


class AudioEncoderTest(unittest.TestCase):
    def test_single_audio_forward_matches_reference_chunking(self) -> None:
        torch.manual_seed(1234)
        config = Qwen3ASRAudioEncoderConfig(
            num_mel_bins=8,
            d_model=16,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=32,
            downsample_hidden_size=4,
            output_dim=16,
            n_window=5,
            n_window_infer=20,
            conv_chunksize=2,
            max_source_positions=32,
            _attn_implementation="eager",
        )
        encoder = Qwen3ASRAudioEncoder(config).eval()

        for feature_len in [7, 10, 23]:
            with self.subTest(feature_len=feature_len):
                input_features = torch.randn(config.num_mel_bins, feature_len)
                feature_lens = torch.tensor([feature_len], dtype=torch.long)

                original_pack_audio = encoder._pack_audio
                encoder._pack_audio = MethodType(_reference_pack_audio, encoder)
                try:
                    expected = encoder(input_features, feature_lens=feature_lens).last_hidden_state
                finally:
                    encoder._pack_audio = original_pack_audio
                actual = encoder(input_features, feature_lens=feature_lens).last_hidden_state

                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
