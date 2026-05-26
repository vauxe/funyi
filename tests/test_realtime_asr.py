# coding=utf-8
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from realtime_server import (
    CUDA_GRAPH_CAPTURE_LOCK,
    TranslationServiceConfig,
    WebSocketSendTimeout,
    _build_model_load,
    _build_session_config,
    _build_translation,
    _close_websocket,
    _parse_args,
    _parse_language_config_update,
    _prepare_cuda_graph_runtime,
    _publish_finish_events,
    _receive_start,
    _publish_session_events,
    _receive_or_sender_failed,
    _run_store_write,
    _send_queued_events,
    _session_translation_config,
    _translation_capture_lock,
    _validate_timestamp_start_language,
)
from qwen3_asr_runtime.language_support import (
    HYMT_MODEL_CARD_LANGUAGES,
    QWEN3_ASR_MODEL_CARD_LANGUAGES,
    QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES,
)
from qwen3_asr_runtime.realtime_session import RealtimeASRConfig, RealtimeASRSession
from qwen3_asr_runtime.streaming import RecognitionFrame, TailSelector, TextStabilizer
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.utils import SUPPORTED_LANGUAGES
from qwen3_asr_runtime.vad import SileroVadAdapter, SileroVadConfig


def _desktop_language_options(name: str) -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[1] / "desktop/ui/src/languages.ts"
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"export const {re.escape(name)} = \[(.*?)\] as const;", text, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing desktop language constant: {name}")
    values = ast.literal_eval(f"[{match.group(1)}]")
    return tuple(values)


class FakeStreamingModel:
    def __init__(
        self,
        outputs: list[object] | None = None,
        finish_text: object | None = None,
        finish_outputs: list[object] | None = None,
        chunk_size_sec: float = 1.0,
    ) -> None:
        self.outputs = list(outputs or [])
        self.finish_text = finish_text
        self.finish_outputs = list(finish_outputs or [])
        self.chunk_size_sec = float(chunk_size_sec)
        self.init_count = 0
        self.stream_calls = 0
        self.finish_calls = 0
        self.init_kwargs: list[dict[str, object]] = []
        self.stream_audio_lengths: list[int] = []
        self.audio_seen_samples = 0

    def low_latency_preset_kwargs(self) -> dict[str, object]:
        return {
            "chunk_size_sec": self.chunk_size_sec,
            "unfixed_chunk_num": 4,
            "max_window_sec": 20.0,
            "spec_decode": True,
        }

    def init_streaming_state(self, **kwargs: object) -> SimpleNamespace:
        self.init_count += 1
        self.init_kwargs.append(dict(kwargs))
        return SimpleNamespace(
            text="",
            language=kwargs.get("language") or "Chinese",
            recognition_frame=None,
        )

    def streaming_transcribe(self, audio: np.ndarray, state: SimpleNamespace) -> SimpleNamespace:
        self.stream_calls += 1
        self.stream_audio_lengths.append(int(audio.shape[0]))
        self.audio_seen_samples += int(audio.shape[0])
        if self.outputs:
            self._apply_output(state, self.outputs.pop(0))
        return state

    def finish_streaming_transcribe(self, state: SimpleNamespace) -> SimpleNamespace:
        self.finish_calls += 1
        if self.finish_outputs:
            self._apply_output(state, self.finish_outputs.pop(0))
        elif self.finish_text is not None:
            self._apply_output(state, self.finish_text)
        return state

    def _apply_output(self, state: SimpleNamespace, output: object) -> None:
        window_start_sample = 0
        if isinstance(output, tuple):
            text, window_start_sample = output
        else:
            text = output

        text_value = str(text)
        language = getattr(state, "language", "Chinese") or "Chinese"
        state.text = text_value
        state.recognition_frame = RecognitionFrame(
            window_start_sample=int(window_start_sample),
            audio_end_sample=self.audio_seen_samples,
            full_text=text_value,
            language=language,
            decoded_text=text_value,
            generated_text=text_value,
        )


def make_session(
    model: FakeStreamingModel,
    *,
    live_stability_delay_ms: int = 12_000,
    force_align_timestamps: bool = False,
) -> RealtimeASRSession:
    return RealtimeASRSession(
        model,
        config=RealtimeASRConfig(
            language="Chinese",
            live_stability_delay_ms=live_stability_delay_ms,
            force_align_timestamps=force_align_timestamps,
        ),
    )


def transcript_updates(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if event.get("type") == "transcript_update"]


def partial_texts(events: list[dict[str, object]]) -> list[str]:
    texts: list[str] = []
    for event in transcript_updates(events):
        partial = event.get("partial")
        if isinstance(partial, dict):
            texts.append(str(partial.get("text") or ""))
    return texts


def stable_appends(events: list[dict[str, object]]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for event in transcript_updates(events):
        appends = event.get("stable_appends")
        if isinstance(appends, list):
            segments.extend(segment for segment in appends if isinstance(segment, dict))
    return segments


def assert_transcript_update_invariants(test: unittest.TestCase, events: list[dict[str, object]]) -> None:
    revision = 0
    stable_count = 0
    for event in transcript_updates(events):
        test.assertGreater(int(event["revision"]), revision)
        revision = int(event["revision"])
        test.assertEqual(event["stable_base"], stable_count)
        appends = event.get("stable_appends")
        test.assertIsInstance(appends, list)
        stable_count += len(appends)
        test.assertEqual(event["stable_count"], stable_count)
        test.assertIn("partial", event)
        partial = event.get("partial")
        test.assertTrue(partial is None or isinstance(partial, dict))


class TranscriptStoreTest(unittest.TestCase):
    def test_segments_are_appended_with_monotonic_timestamps(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(text="第一句。", start_ms=0, end_ms=1200, language="Chinese")
        store.append_stable_segment(text="Second.", start_ms=1000, end_ms=2300, language="English")

        self.assertEqual([segment.text for segment in store.stable_segments], ["第一句。", "Second."])
        self.assertEqual(store.stable_segments[1].start_ms, first.end_ms)
        self.assertEqual(store.stable_count, 2)

    def test_stable_segment_text_preserves_boundary_whitespace(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(text="hello ", start_ms=0, end_ms=1000, language="English")
        second = store.append_stable_segment(text="world", start_ms=1000, end_ms=2000, language="English")

        self.assertEqual("".join(segment.text for segment in store.stable_segments), "hello world")
        event = store.update_event(stable_base=0, stable_appends=[first, second])
        self.assertEqual("".join(str(segment["text"]) for segment in event["stable_appends"]), "hello world")

    def test_update_event_uses_stable_cursor_and_replaceable_partial(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(text="稳定", start_ms=0, end_ms=1000, language="Chinese")
        event = store.update_event(stable_base=0, stable_appends=[segment])

        self.assertEqual(event["type"], "transcript_update")
        self.assertEqual(event["stable_base"], 0)
        self.assertEqual(event["stable_count"], 1)
        self.assertEqual(event["stable_appends"][0]["text"], "稳定")
        self.assertIsNone(event["partial"])

    def test_update_event_rejects_stale_stable_base(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(text="稳定", start_ms=0, end_ms=1000, language="Chinese")

        with self.assertRaises(ValueError):
            store.update_event(stable_base=1, stable_appends=[segment])

    def test_pending_stable_segment_timing_is_patched_by_segment_id(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="稳定",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        update = store.update_event(stable_base=0, stable_appends=[segment])

        self.assertIsNone(update["stable_appends"][0]["start_ms"])
        self.assertIsNone(update["stable_appends"][0]["end_ms"])
        self.assertEqual(update["stable_appends"][0]["timing_status"], "pending")

        timing_update = store.update_segment_timing(
            source_segment_id="seg_000001",
            start_ms=120,
            end_ms=860,
            timing_status="aligned",
        )
        final = store.final_event()

        self.assertEqual(
            timing_update,
            {
                "type": "transcript_timing_update",
                "source_segment_id": "seg_000001",
                "start_ms": 120,
                "end_ms": 860,
                "timing_status": "aligned",
            },
        )
        self.assertEqual(final["segments"][0]["start_ms"], 120)
        self.assertEqual(final["segments"][0]["end_ms"], 860)
        self.assertEqual(final["segments"][0]["timing_status"], "aligned")


class TextStabilizerTest(unittest.TestCase):
    def test_repeated_prefix_is_required_before_stabilizing(self) -> None:
        stabilizer = TextStabilizer()

        first = stabilizer.observe("第一秒", end_sample=16_000, can_commit=True)
        second = stabilizer.observe("第一秒第二秒", end_sample=32_000, can_commit=True)

        self.assertEqual(first.stable_text, "")
        self.assertEqual(first.partial_text, "第一秒")
        self.assertIsNone(first.stable_end_sample)
        self.assertEqual(second.stable_text, "第一秒")
        self.assertEqual(second.partial_text, "第二秒")
        self.assertEqual(second.stable_end_sample, 16_000)

    def test_stable_prefix_does_not_split_ascii_word(self) -> None:
        stabilizer = TextStabilizer("hello wor", 16_000)

        update = stabilizer.observe("hello world today", end_sample=32_000, can_commit=True)

        self.assertEqual(update.stable_text, "hello")
        self.assertEqual(update.partial_text, "world today")
        self.assertEqual(update.stable_end_sample, 16_000)


class TailSelectorTest(unittest.TestCase):
    def test_exact_prefix_projects_uncommitted_tail(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=0,
            audio_end_sample=32_000,
            full_text="前段后段",
            decoded_text="前段后段",
            generated_text="前段后段",
            language="Chinese",
        )

        tail = TailSelector.select(frame, stable_text_prefix="前段", stable_end_sample=16_000)

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "后段")

    def test_window_after_stable_cursor_keeps_decoded_mutable_tail(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=16_000,
            audio_end_sample=32_000,
            full_text="已裁掉的历史草稿后段",
            decoded_text="草稿后段",
            generated_text="后段",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix="前段已稳定",
            stable_end_sample=16_000,
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "草稿后段")

    def test_prompt_prefix_echo_is_not_returned_as_window_tail(self) -> None:
        stable = "现在已经没悬念了后段"
        frame = RecognitionFrame(
            window_start_sample=16_000,
            audio_end_sample=32_000,
            full_text="现在已经没悬念了有轻微改写后段继续",
            decoded_text="有轻微改写后段继续",
            generated_text="有轻微改写后段继续",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix=stable,
            stable_end_sample=16_000,
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "有轻微改写后段继续")

    def test_stable_suffix_overlap_removes_committed_prompt_tail(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=44_000,
            audio_end_sample=64_000,
            full_text="旧历史当年那个离开雅信，第一次出来创业",
            decoded_text="当年那个离开雅信，第一次出来创业",
            generated_text="出来创业",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix="昨天发布会之后，当年那个离开雅信，第一次",
            stable_end_sample=46_000,
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "出来创业")

    def test_stable_suffix_overlap_ignores_boundary_punctuation(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=100_000,
            audio_end_sample=120_000,
            full_text="旧历史那么大家都知道，在飞机里面经常会有一些广告",
            decoded_text="那么大家都知道，在飞机里面经常会有一些广告",
            generated_text="有一些广告",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix="杂志的事情是我大学第一次坐飞机那么大家都知道在飞机里面经常会",
            stable_end_sample=102_000,
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "有一些广告")

    def test_long_stable_context_echo_is_not_committable(self) -> None:
        stable = "昨天发布会怎么样兴奋困想睡觉一般来讲"
        decoded = "昨天发布会怎么样有轻微改写后段"
        frame = RecognitionFrame(
            window_start_sample=32_000,
            audio_end_sample=48_000,
            full_text="开头改写" + decoded,
            decoded_text=decoded,
            generated_text=decoded,
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix=stable,
            stable_end_sample=16_000,
        )

        self.assertFalse(tail.aligned)

    def test_repeated_phrase_at_window_boundary_is_not_removed_as_prompt_echo(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=16_000,
            audio_end_sample=32_000,
            full_text="旧历史谢谢大家下一句",
            decoded_text="谢谢大家下一句",
            generated_text="下一句",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix="上一句谢谢大家",
            stable_end_sample=16_000,
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "谢谢大家下一句")

    def test_empty_full_text_does_not_hide_decoded_tail(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=16_000,
            audio_end_sample=32_000,
            full_text="",
            decoded_text="后段",
            generated_text="后段",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix="前段",
            stable_end_sample=16_000,
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "后段")

    def test_visible_partial_is_not_removed_by_stable_suffix_overlap(self) -> None:
        frame = RecognitionFrame(
            window_start_sample=15_000,
            audio_end_sample=32_000,
            full_text="旧历史谢谢大家下一句",
            decoded_text="谢谢大家下一句",
            generated_text="下一句",
            language="Chinese",
        )

        tail = TailSelector.select(
            frame,
            stable_text_prefix="上一句谢谢大家",
            stable_end_sample=16_000,
            previous_partial_text="谢谢大家下一句",
        )

        self.assertTrue(tail.aligned)
        self.assertEqual(tail.text, "谢谢大家下一句")


class RealtimeServerCliTest(unittest.TestCase):
    def test_gpu_runtime_is_default(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            args = _parse_args()

        self.assertIsNone(args.device_map)
        self.assertIsNone(args.cuda_graph)
        self.assertIsNone(args.flashinfer)
        self.assertIsNone(args.fused_rmsnorm)
        self.assertIsNone(args.fused_linears)
        self.assertIsNone(args.w8a16)
        self.assertTrue(args.cuda_graph_prewarm)
        self.assertEqual(args.cuda_graph_prewarm_language, "Chinese")
        self.assertIsNone(args.timestamp_model)
        self.assertTrue(args.timestamp_local_files_only)
        self.assertEqual(args.timestamp_pad_ms, 500)
        self.assertEqual(args.timestamp_finish_timeout_ms, 30_000)
        self.assertIsNone(args.translation_model)
        self.assertEqual(args.translation_preview_debounce_ms, 700)
        self.assertEqual(args.translation_stable_batch_size, 1)
        self.assertFalse(args.translation_sample)
        self.assertEqual(args.live_stability_delay_ms, 12_000)

    def test_live_stability_delay_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["realtime_server.py", "--model", "model", "--live-stability-delay-ms", "5000"],
        ):
            self.assertEqual(_parse_args().live_stability_delay_ms, 5_000)

    def test_translation_sampling_can_be_enabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-sample"]):
            self.assertTrue(_parse_args().translation_sample)

    def test_translation_stable_batch_size_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["realtime_server.py", "--model", "model", "--translation-stable-batch-size", "4"],
        ):
            self.assertEqual(_parse_args().translation_stable_batch_size, 4)

    def test_w8a16_can_be_disabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--no-w8a16"]):
            self.assertFalse(_parse_args().w8a16)

    def test_translation_model_flag_uses_default_model_when_value_is_omitted(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-model"]):
            self.assertEqual(_parse_args().translation_model, "tencent/HY-MT1.5-1.8B")

    def test_translation_model_can_be_configured(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-model", "local/hymt"]):
            self.assertEqual(_parse_args().translation_model, "local/hymt")

    def test_translation_model_enables_translation_without_default_target(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-model", "local/hymt"]):
            args = _parse_args()

        translator = object()
        with patch("realtime_server.HYMTTranslator", return_value=translator) as translator_class:
            built_translator, config = _build_translation(args)

        self.assertIs(built_translator, translator)
        self.assertIsNotNone(config)
        self.assertEqual(translator_class.call_args.args[0], "local/hymt")

    def test_asr_supported_languages_follow_qwen_model_card(self) -> None:
        self.assertEqual(tuple(SUPPORTED_LANGUAGES), QWEN3_ASR_MODEL_CARD_LANGUAGES)

    def test_desktop_language_options_follow_backend_model_card_lists(self) -> None:
        self.assertEqual(_desktop_language_options("ASR_LANGUAGE_OPTIONS"), QWEN3_ASR_MODEL_CARD_LANGUAGES)
        self.assertEqual(_desktop_language_options("TRANSLATION_TARGET_LANGUAGE_OPTIONS"), HYMT_MODEL_CARD_LANGUAGES)

    def test_transformers_load_kwargs_are_default(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            backend, kwargs = _build_model_load(_parse_args())

        self.assertEqual(backend, "transformers")
        self.assertEqual(kwargs["device_map"], "cuda:0")
        self.assertTrue(kwargs["cuda_graph"])
        self.assertEqual(kwargs["cuda_graph_len_bucket"], 64)
        self.assertTrue(kwargs["flashinfer"])
        self.assertTrue(kwargs["fused_rmsnorm"])
        self.assertTrue(kwargs["fused_linears"])
        self.assertTrue(kwargs["quantized_linears"])

    def test_cuda_graph_prewarm_is_default_hard_gate_for_asr_only(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def prewarm_realtime_cuda_graph(self, **kwargs: object) -> bool:
                self.calls.append(dict(kwargs))
                return True

        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            args = _parse_args()
        model = FakeModel()

        _prepare_cuda_graph_runtime(model, args)

        self.assertEqual(
            model.calls,
            [{"language": "Chinese", "max_window_sec": 20.0, "max_prefix_tokens": 64}],
        )

    def test_cuda_graph_prewarm_failure_is_startup_error(self) -> None:
        class FakeModel:
            def prewarm_realtime_cuda_graph(self, **kwargs: object) -> bool:
                del kwargs
                return False

        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            args = _parse_args()

        with self.assertRaises(RuntimeError):
            _prepare_cuda_graph_runtime(FakeModel(), args)

    def test_translation_uses_capture_lock_when_asr_prewarm_is_disabled(self) -> None:
        class FakeModel:
            def prewarm_realtime_cuda_graph(self, **kwargs: object) -> bool:
                raise AssertionError("prewarm should not run")

        with patch.object(
            sys,
            "argv",
            ["realtime_server.py", "--model", "model", "--no-cuda-graph-prewarm"],
        ):
            args = _parse_args()

        lock = _translation_capture_lock(args, translation_enabled=True)

        self.assertIs(lock, CUDA_GRAPH_CAPTURE_LOCK)

    def test_translation_actor_needs_no_capture_lock_after_default_asr_cuda_graph_prewarm(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def prewarm_realtime_cuda_graph(self, **kwargs: object) -> bool:
                del kwargs
                self.calls += 1
                return True

        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            args = _parse_args()

        model = FakeModel()
        _prepare_cuda_graph_runtime(model, args)
        lock = _translation_capture_lock(args, translation_enabled=True)

        self.assertEqual(model.calls, 1)
        self.assertIsNone(lock)

    def test_start_payload_without_target_disables_session_translation(self) -> None:
        config = TranslationServiceConfig()

        self.assertIsNone(_session_translation_config({"type": "start"}, config))

    def test_start_payload_rejects_empty_translation_target(self) -> None:
        config = TranslationServiceConfig()

        with self.assertRaisesRegex(ValueError, "target_language must not be empty"):
            _session_translation_config({"type": "start", "target_language": ""}, config)

    def test_start_payload_rejects_translation_target_when_translation_is_not_configured(self) -> None:
        self.assertIsNone(_session_translation_config({"type": "start"}, None))

        with self.assertRaisesRegex(ValueError, "translation model to be configured"):
            _session_translation_config({"type": "start", "target_language": "English"}, None)

    def test_start_payload_can_enable_session_translation_target(self) -> None:
        config = TranslationServiceConfig(
            max_new_tokens=16,
        )

        session_config = _session_translation_config(
            {"type": "start", "target_language": "Japanese"},
            config,
        )
        self.assertIsNotNone(session_config)
        self.assertEqual(session_config.target_language, "Japanese")
        self.assertEqual(session_config.max_new_tokens, 16)

    def test_start_payload_rejects_translation_target_outside_hymt_model_card(self) -> None:
        config = TranslationServiceConfig()

        with self.assertRaisesRegex(ValueError, "Unsupported target_language"):
            _session_translation_config({"type": "start", "target_language": "Swedish"}, config)

    def test_start_payload_normalizes_translation_target(self) -> None:
        config = TranslationServiceConfig()

        session_config = _session_translation_config({"type": "start", "target_language": "traditional chinese"}, config)

        self.assertIsNotNone(session_config)
        self.assertEqual(session_config.target_language, "Traditional Chinese")

    def test_service_session_config_uses_start_payload_context(self) -> None:
        config = _build_session_config({"type": "start", "context": "meeting", "language": "Chinese"})

        self.assertEqual(config.context, "meeting")
        self.assertEqual(config.language, "Chinese")

    def test_set_language_command_normalizes_language_choices(self) -> None:
        update = _parse_language_config_update(
            {"type": "set_language", "language": "japanese", "target_language": "traditional chinese"},
            TranslationServiceConfig(),
        )

        self.assertEqual(update, {"language": "Japanese", "target_language": "Traditional Chinese"})

    def test_set_language_command_allows_auto_asr_and_translation_off(self) -> None:
        update = _parse_language_config_update(
            {"type": "set_language", "language": "", "target_language": None},
            None,
        )

        self.assertEqual(update, {"language": None, "target_language": None})

    def test_set_language_command_rejects_target_without_translation_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "translation model to be configured"):
            _parse_language_config_update({"type": "set_language", "target_language": "English"}, None)

    def test_set_language_command_rejects_unknown_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported set_language command field"):
            _parse_language_config_update(
                {"type": "set_language", "language": "English", "extra": True},
                TranslationServiceConfig(),
            )

    def test_timestamp_start_language_follows_forced_aligner_model_card(self) -> None:
        _validate_timestamp_start_language({"type": "start", "language": "Japanese"}, timestamps_enabled=True)
        _validate_timestamp_start_language({"type": "start"}, timestamps_enabled=True)

        with self.assertRaisesRegex(ValueError, "forced-aligner timestamps"):
            _validate_timestamp_start_language({"type": "start", "language": "Arabic"}, timestamps_enabled=True)

        self.assertIn("Japanese", QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES)
        self.assertNotIn("Arabic", QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES)

    def test_set_language_command_follows_forced_aligner_model_card(self) -> None:
        _parse_language_config_update(
            {"type": "set_language", "language": "Japanese"},
            TranslationServiceConfig(),
            timestamps_enabled=True,
        )

        with self.assertRaisesRegex(ValueError, "forced-aligner timestamps"):
            _parse_language_config_update(
                {"type": "set_language", "language": "Arabic"},
                TranslationServiceConfig(),
                timestamps_enabled=True,
            )


class RealtimeServerTranslationOrderingTest(unittest.IsolatedAsyncioTestCase):
    async def test_receive_start_normalizes_supported_language(self) -> None:
        class FakeWebSocket:
            async def receive(self) -> dict[str, object]:
                return {"text": '{"type":"start","language":"japanese"}'}

        payload = await _receive_start(FakeWebSocket())

        self.assertIsNotNone(payload)
        self.assertEqual(payload["language"], "Japanese")  # type: ignore[index]

    async def test_receive_start_rejects_unsupported_language(self) -> None:
        class FakeWebSocket:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.closed_code: int | None = None

            async def receive(self) -> dict[str, object]:
                return {"text": '{"type":"start","language":"Klingon"}'}

            async def send_text(self, text: str) -> None:
                self.sent.append(text)

            async def close(self, code: int) -> None:
                self.closed_code = code

        websocket = FakeWebSocket()

        payload = await _receive_start(websocket)

        self.assertIsNone(payload)
        self.assertEqual(websocket.closed_code, 1003)
        self.assertIn("Unsupported language", websocket.sent[0])

    async def test_receive_start_rejects_unknown_field(self) -> None:
        class FakeWebSocket:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.closed_code: int | None = None

            async def receive(self) -> dict[str, object]:
                return {"text": '{"type":"start","unsupported":true}'}

            async def send_text(self, text: str) -> None:
                self.sent.append(text)

            async def close(self, code: int) -> None:
                self.closed_code = code

        websocket = FakeWebSocket()

        payload = await _receive_start(websocket)

        self.assertIsNone(payload)
        self.assertEqual(websocket.closed_code, 1003)
        self.assertIn("Unsupported start command field", websocket.sent[0])
        self.assertIn("unsupported", websocket.sent[0])

    async def test_sender_times_out_when_client_stops_consuming_output(self) -> None:
        class HangingSendWebSocket:
            def __init__(self) -> None:
                self.send_started = asyncio.Event()

            async def send_text(self, text: str) -> None:
                del text
                self.send_started.set()
                await asyncio.Future()

        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        await queue.put({"type": "ready"})
        websocket = HangingSendWebSocket()

        with self.assertRaises(WebSocketSendTimeout):
            await _send_queued_events(websocket, queue, send_timeout_sec=0.01)

        self.assertTrue(websocket.send_started.is_set())
        self.assertEqual(queue.qsize(), 0)

    async def test_receive_wait_is_interrupted_when_sender_fails(self) -> None:
        class HangingReceiveWebSocket:
            def __init__(self) -> None:
                self.receive_cancelled = False

            async def receive(self) -> dict[str, object]:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.receive_cancelled = True
                    raise

        async def fail_sender() -> None:
            raise WebSocketSendTimeout("client did not consume output")

        websocket = HangingReceiveWebSocket()
        sender_task = asyncio.create_task(fail_sender())

        with self.assertRaises(WebSocketSendTimeout):
            await _receive_or_sender_failed(websocket, sender_task)

        self.assertTrue(websocket.receive_cancelled)

    async def test_close_websocket_times_out_when_client_stops_consuming_output(self) -> None:
        class HangingCloseWebSocket:
            def __init__(self) -> None:
                self.close_started = asyncio.Event()

            async def close(self, code: int) -> None:
                del code
                self.close_started.set()
                await asyncio.Future()

        websocket = HangingCloseWebSocket()

        await _close_websocket(websocket, code=1011, timeout_sec=0.01)

        self.assertTrue(websocket.close_started.is_set())

    async def test_pending_old_preview_is_not_queued_after_new_source_revision(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        class FakeTranslation:
            async def accept_source_event(self, event: dict[str, object]) -> None:
                self.assert_source_event(event)
                await queue.put({"type": "translation_preview", "source_revision": 1, "text": "old"})

            def assert_source_event(self, event: dict[str, object]) -> None:
                if event.get("type") != "transcript_update" or event.get("revision") != 2:
                    raise AssertionError(f"unexpected source event: {event!r}")

        await _publish_session_events(
            queue,
            FakeTranslation(),  # type: ignore[arg-type]
            [
                {
                    "type": "transcript_update",
                    "revision": 2,
                    "stable_appends": [],
                    "partial": None,
                }
            ],
        )

        first = await queue.get()
        second = await queue.get()
        self.assertEqual(first["type"], "translation_preview")
        self.assertEqual(first["source_revision"], 1)
        self.assertEqual(second["type"], "transcript_update")
        self.assertEqual(second["revision"], 2)

    async def test_finish_cancels_preview_before_publishing_finish_source_update(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        class FakeTranslation:
            async def cancel_preview(self) -> None:
                await queue.put({"type": "translation_preview", "source_revision": 1, "text": "old"})

            async def finish(self, transcript_updates: list[dict[str, object]]) -> list[dict[str, object]]:
                if [event.get("revision") for event in transcript_updates] != [2]:
                    raise AssertionError(f"unexpected finish updates: {transcript_updates!r}")
                return [
                    {
                        "type": "translation_stable",
                        "source_revision": 2,
                        "source_segment_id": "seg_000001",
                        "source_segment_index": 1,
                        "target_language": "English",
                        "text": "tail",
                    }
                ]

        await _publish_finish_events(
            queue,
            FakeTranslation(),  # type: ignore[arg-type]
            [
                {
                    "type": "transcript_update",
                    "revision": 2,
                    "stable_appends": [
                        {
                            "id": "seg_000001",
                            "index": 1,
                            "start_ms": 0,
                            "end_ms": 900,
                            "text": "tail",
                            "language": "Chinese",
                        }
                    ],
                    "partial": None,
                },
                {
                    "type": "transcript_final",
                    "segments": [
                        {
                            "id": "seg_000001",
                            "index": 1,
                            "start_ms": 0,
                            "end_ms": 900,
                            "text": "tail",
                            "language": "Chinese",
                        }
                    ],
                },
            ],
        )

        events = [await queue.get(), await queue.get(), await queue.get(), await queue.get()]
        self.assertEqual([event["type"] for event in events], [
            "translation_preview",
            "transcript_update",
            "translation_stable",
            "transcript_final",
        ])
        self.assertEqual(events[0]["source_revision"], 1)
        self.assertEqual(events[1]["revision"], 2)

    async def test_finish_publishes_timing_patch_before_fresh_final_snapshot(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        finish_seen = False

        class FakeTimestamp:
            async def finish(self, jobs: list[object]) -> list[dict[str, object]]:
                nonlocal finish_seen
                if jobs != ["job"]:
                    raise AssertionError(f"unexpected timestamp jobs: {jobs!r}")
                finish_seen = True
                return [
                    {
                        "type": "transcript_timing_update",
                        "source_segment_id": "seg_000001",
                        "start_ms": 120,
                        "end_ms": 860,
                        "timing_status": "aligned",
                    }
                ]

        def final_event() -> dict[str, object]:
            if not finish_seen:
                raise AssertionError("final snapshot was built before timestamp finish")
            return {
                "type": "transcript_final",
                "segments": [
                    {
                        "id": "seg_000001",
                        "index": 1,
                        "start_ms": 120,
                        "end_ms": 860,
                        "text": "tail",
                        "language": "Chinese",
                    }
                ],
            }

        await _publish_finish_events(
            queue,
            None,
            [
                {
                    "type": "transcript_update",
                    "revision": 2,
                    "stable_appends": [
                        {
                            "id": "seg_000001",
                            "index": 1,
                            "start_ms": None,
                            "end_ms": None,
                            "timing_status": "pending",
                            "text": "tail",
                            "language": "Chinese",
                        }
                    ],
                    "partial": None,
                }
            ],
            FakeTimestamp(),  # type: ignore[arg-type]
            ["job"],  # type: ignore[list-item]
            final_event_factory=final_event,
        )

        events = [await queue.get(), await queue.get(), await queue.get()]
        self.assertEqual([event["type"] for event in events], [
            "transcript_update",
            "transcript_timing_update",
            "transcript_final",
        ])
        self.assertEqual(events[2]["segments"][0]["start_ms"], 120)

    async def test_finish_final_snapshot_waits_for_store_lock(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        store_lock = asyncio.Lock()
        final_called = False

        def final_event() -> dict[str, object]:
            nonlocal final_called
            final_called = True
            return {"type": "transcript_final", "segments": []}

        async with store_lock:
            publish_task = asyncio.create_task(
                _publish_finish_events(
                    queue,
                    None,
                    [],
                    final_event_factory=final_event,
                    store_lock=store_lock,
                )
            )
            await asyncio.sleep(0.05)
            self.assertFalse(final_called)
            self.assertFalse(publish_task.done())

        await asyncio.wait_for(publish_task, timeout=1.0)
        self.assertTrue(final_called)
        self.assertEqual((await queue.get())["type"], "transcript_final")

    async def test_session_store_write_waits_for_store_lock(self) -> None:
        store_lock = asyncio.Lock()
        called = False

        def write_store() -> str:
            nonlocal called
            called = True
            return "done"

        async def run_inline(func, *args):
            return func(*args)

        with patch("realtime_server.asyncio.to_thread", side_effect=run_inline):
            async with store_lock:
                write_task = asyncio.create_task(_run_store_write(store_lock, write_store))
                await asyncio.sleep(0.05)
                self.assertFalse(called)
                self.assertFalse(write_task.done())

            self.assertEqual(await asyncio.wait_for(write_task, timeout=1.0), "done")


class RealtimeASRSessionTest(unittest.TestCase):
    def test_service_default_keeps_stable_history_conservative(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "前两秒", "前两秒第三秒"])
        config = _build_session_config({"type": "start", "language": "Chinese"})
        session = RealtimeASRSession(model, config=config)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        self.assertEqual(stable_appends(events), [])
        self.assertEqual(partial_texts(events)[-1], "前两秒第三秒")
        assert_transcript_update_invariants(self, events)

    def test_force_align_timestamp_mode_emits_pending_stable_segment_and_hidden_timing_job(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = make_session(model, live_stability_delay_ms=0, force_align_timestamps=True)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events = session.ingest_audio(speech) + session.ingest_audio(speech)
        stable = stable_appends(events)
        jobs = session.stable_timing_jobs_for_events(events)

        self.assertEqual(len(stable), 1)
        self.assertIsNone(stable[0]["start_ms"])
        self.assertIsNone(stable[0]["end_ms"])
        self.assertEqual(stable[0]["timing_status"], "pending")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].source_segment_id, stable[0]["id"])
        self.assertEqual(jobs[0].source_text, "第一秒")
        self.assertEqual((jobs[0].start_sample, jobs[0].end_sample), (0, 16_000))

    def test_silence_is_not_an_asr_input_gate(self) -> None:
        model = FakeStreamingModel(outputs=["低能量语音。"])
        session = make_session(model)

        events = session.ingest_audio(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(partial_texts(events), ["低能量语音。"])
        self.assertEqual(model.init_count, 1)
        self.assertEqual(model.stream_calls, 1)
        self.assertEqual(model.stream_audio_lengths, [16_000])
        assert_transcript_update_invariants(self, events)

    def test_punctuation_does_not_stabilize_while_speech_continues(self) -> None:
        model = FakeStreamingModel(outputs=["第一句。第二", "第一句。第二句。第三", "第一句。第二句。第三句"])
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events1 = session.ingest_audio(speech)
        events2 = session.ingest_audio(speech)
        events3 = session.ingest_audio(speech)

        events = events1 + events2 + events3
        self.assertEqual(stable_appends(events), [])
        self.assertEqual(partial_texts(events)[-1], "第一句。第二句。第三句")
        self.assertEqual(model.init_count, 1)
        assert_transcript_update_invariants(self, events)

    def test_asr_runs_on_server_side_half_second_cadence(self) -> None:
        model = FakeStreamingModel(outputs=["半秒"], chunk_size_sec=0.5)
        session = make_session(model)
        speech = np.ones(1_600, dtype=np.float32) * 0.2

        for _ in range(4):
            self.assertEqual(session.ingest_audio(speech), [])
            self.assertEqual(model.stream_calls, 0)

        events = session.ingest_audio(speech)

        self.assertEqual(partial_texts(events), ["半秒"])
        self.assertEqual(model.stream_calls, 1)
        self.assertEqual(model.stream_audio_lengths, [8_000])
        self.assertEqual(model.init_kwargs[0]["chunk_size_sec"], 0.5)
        self.assertEqual(model.init_kwargs[0]["unfixed_chunk_num"], 4)
        self.assertEqual(model.init_kwargs[0]["max_window_sec"], 20.0)
        self.assertEqual(model.init_kwargs[0]["max_prefix_tokens"], 64)
        self.assertTrue(model.init_kwargs[0]["spec_decode"])
        self.assertEqual(model.init_kwargs[0]["language"], "Chinese")

    def test_rewritten_partial_text_is_not_stabilized_by_punctuation(self) -> None:
        model = FakeStreamingModel(
            outputs=[
                "第一句。",
                "第一句话，也。",
                "第一句话，有补充。",
                "第一句话，有补充。下一段，",
            ]
        )
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        for _ in range(4):
            events.extend(session.ingest_audio(speech))

        self.assertEqual(stable_appends(events), [])
        self.assertEqual(partial_texts(events)[-1], "第一句话，有补充。下一段，")
        assert_transcript_update_invariants(self, events)

    def test_continuous_audio_advances_stable_cursor_without_resetting_asr(self) -> None:
        model = FakeStreamingModel(
            outputs=["第一秒", "前两秒", "前两秒第三秒"],
        )
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        updates = transcript_updates(events)
        non_empty_partials = [
            event["partial"]
            for event in updates
            if isinstance(event.get("partial"), dict)
        ]
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0]["text"], "前两秒")
        self.assertEqual(stable[0]["start_ms"], 0)
        self.assertEqual(stable[0]["end_ms"], 2_000)
        self.assertEqual(non_empty_partials[-1]["text"], "第三秒")
        self.assertEqual(non_empty_partials[-1]["start_ms"], 2_000)
        self.assertEqual(non_empty_partials[-1]["end_ms"], 3_000)
        self.assertEqual(model.init_count, 1)
        self.assertEqual(model.finish_calls, 0)
        self.assertEqual(model.stream_audio_lengths, [16_000, 16_000, 16_000])
        assert_transcript_update_invariants(self, events)

    def test_live_stability_delay_waits_for_repeated_prefix_before_stabilizing(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "前两秒"])
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        self.assertEqual(stable_appends(events), [])
        self.assertEqual(partial_texts(events)[-1], "前两秒")
        assert_transcript_update_invariants(self, events)

    def test_zero_live_stability_delay_still_requires_repeated_prefix(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        self.assertEqual([segment["text"] for segment in stable], ["第一秒"])
        self.assertEqual(partial_texts(events)[-1], "第二秒")
        assert_transcript_update_invariants(self, events)

    def test_repeated_tail_text_after_stable_prefix_is_not_dropped(self) -> None:
        model = FakeStreamingModel(outputs=["哈哈", "哈哈哈哈", "哈哈哈哈哈"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        self.assertEqual([segment["text"] for segment in stable], ["哈哈", "哈哈"])
        self.assertEqual(partial_texts(events)[-1], "哈")
        assert_transcript_update_invariants(self, events)

    def test_unaligned_live_window_still_updates_partial_without_stabilizing_it(self) -> None:
        model = FakeStreamingModel(outputs=["旧段", "旧段", "新内容", "新内容继续"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["旧段"])
        self.assertEqual(partial_texts(events)[-1], "新内容继续")
        assert_transcript_update_invariants(self, events)

    def test_tail_only_window_keeps_updating_current_partial(self) -> None:
        model = FakeStreamingModel(outputs=["旧段", "旧段", "新", "新内容"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["旧段"])
        self.assertEqual(partial_texts(events)[-1], "新内容")
        assert_transcript_update_invariants(self, events)

    def test_finish_stabilizes_confirmed_tail_only_partial(self) -> None:
        model = FakeStreamingModel(
            outputs=["旧段", "旧段", ("新", 16_000), ("新内容", 16_000)],
            finish_text=("内容", 48_000),
        )
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        for _ in range(4):
            events.extend(session.ingest_audio(speech))
        events.extend(session.finish())

        final_events = [event for event in events if event.get("type") == "transcript_final"]
        final_text = "".join(segment["text"] for segment in final_events[-1]["segments"])
        self.assertEqual(final_text, "旧段新内容")
        assert_transcript_update_invariants(self, events)

    def test_finish_stabilizes_tail_only_window_after_stable_cursor(self) -> None:
        model = FakeStreamingModel(
            outputs=["前段", "前段"],
            finish_outputs=[("后段", 16_000)],
        )
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.finish())

        final_events = [event for event in events if event.get("type") == "transcript_final"]
        self.assertEqual([segment["text"] for segment in final_events[-1]["segments"]], ["前段", "后段"])
        assert_transcript_update_invariants(self, events)

    def test_stable_prefix_does_not_split_ascii_word(self) -> None:
        model = FakeStreamingModel(outputs=["hello wor", "hello world today"])
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        self.assertEqual([segment["text"] for segment in stable], ["hello"])
        self.assertEqual(partial_texts(events)[-1], "world today")
        assert_transcript_update_invariants(self, events)

    def test_long_stable_text_is_split_into_subtitle_sized_cues(self) -> None:
        stable_text = "一二三四五六七八九十甲乙丙丁戊己庚辛。后续文本"
        model = FakeStreamingModel(outputs=[stable_text, stable_text + "后续"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        self.assertGreater(len(stable), 1)
        self.assertEqual("".join(str(segment["text"]) for segment in stable), stable_text)
        self.assertTrue(all(len(str(segment["text"])) <= 18 for segment in stable))
        self.assertEqual(stable[0]["start_ms"], 0)
        self.assertEqual(stable[-1]["end_ms"], 1_000)
        self.assertEqual(partial_texts(events)[-1], "后续")
        assert_transcript_update_invariants(self, events)

    def test_long_ascii_stable_text_split_preserves_spaces(self) -> None:
        stable_text = "hello world today again tomorrow"
        model = FakeStreamingModel(outputs=[stable_text, stable_text + " next"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        self.assertGreater(len(stable), 1)
        self.assertEqual("".join(str(segment["text"]) for segment in stable), stable_text)
        self.assertEqual(partial_texts(events)[-1], "next")
        assert_transcript_update_invariants(self, events)

    def test_flush_after_stable_cursor_stabilizes_only_tail_without_resetting_asr(self) -> None:
        model = FakeStreamingModel(
            outputs=["第一秒", "前两秒", "前两秒第三秒"],
            finish_text="前两秒第三秒",
        )
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.flush())

        stable = stable_appends(events)
        self.assertEqual([segment["text"] for segment in stable], ["前两秒", "第三秒"])
        self.assertEqual(
            [(segment["start_ms"], segment["end_ms"]) for segment in stable],
            [(0, 2_000), (2_000, 3_000)],
        )
        self.assertEqual(model.init_count, 1)
        self.assertEqual(model.finish_calls, 1)
        assert_transcript_update_invariants(self, events)

    def test_asr_rewrite_of_stable_prefix_preserves_existing_partial(self) -> None:
        model = FakeStreamingModel(
            outputs=[
                "第一秒",
                "第一秒第二秒",
                "第一秒第二秒第三秒",
                "第一秒二秒第三秒第四秒",
            ],
        )
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        for _ in range(4):
            events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        self.assertEqual([segment["text"] for segment in stable], ["第一秒", "第二秒"])
        self.assertEqual(partial_texts(events)[-1], "第三秒")
        assert_transcript_update_invariants(self, events)

    def test_finish_with_unaligned_asr_promotes_last_visible_partial(self) -> None:
        model = FakeStreamingModel(
            outputs=["第一秒", "第一秒第二秒", "第一秒第二秒第三秒"],
            finish_text="完全不同的最终结果",
        )
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        for _ in range(3):
            events.extend(session.ingest_audio(speech))
        events.extend(session.finish())

        updates = transcript_updates(events)
        final_events = [event for event in events if event.get("type") == "transcript_final"]
        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["第一秒", "第二秒", "第三秒"])
        self.assertEqual(updates[-1]["stable_appends"][0]["text"], "第三秒")
        self.assertIsNone(updates[-1]["partial"])
        self.assertEqual(final_events[-1]["stable_count"], 3)
        self.assertEqual([segment["text"] for segment in final_events[-1]["segments"]], ["第一秒", "第二秒", "第三秒"])
        assert_transcript_update_invariants(self, events)

    def test_finish_with_unaligned_tail_update_promotes_longer_final_tail(self) -> None:
        model = FakeStreamingModel(
            outputs=["第一秒", "第一秒第二秒", "第一秒第二秒第三秒"],
            finish_text="第三秒尾巴",
        )
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        for _ in range(3):
            events.extend(session.ingest_audio(speech))
        events.extend(session.finish())

        final_events = [event for event in events if event.get("type") == "transcript_final"]
        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["第一秒", "第二秒", "第三秒尾巴"])
        self.assertEqual([segment["text"] for segment in final_events[-1]["segments"]], ["第一秒", "第二秒", "第三秒尾巴"])
        assert_transcript_update_invariants(self, events)

    def test_flush_stabilizes_tail_without_resetting_streaming_state(self) -> None:
        model = FakeStreamingModel(outputs=["尾句", "尾句后续"], finish_text="尾句")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        flush_events = session.flush()
        resume_events = session.ingest_audio(speech)

        self.assertEqual([segment["text"] for segment in stable_appends(flush_events)], ["尾句"])
        self.assertEqual(partial_texts(resume_events), ["后续"])
        self.assertEqual(model.finish_calls, 1)
        self.assertEqual(model.init_count, 1)

    def test_set_language_flushes_tail_and_restarts_future_asr_state(self) -> None:
        model = FakeStreamingModel(outputs=["hello", "world"], finish_text="hello")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        switch_events = session.set_language("English")
        resume_events = session.ingest_audio(speech)

        stable = stable_appends(switch_events)
        self.assertEqual([segment["text"] for segment in stable], ["hello"])
        self.assertEqual(stable[0]["language"], "Chinese")
        self.assertEqual(session.config.language, "English")
        self.assertEqual(partial_texts(resume_events), ["world"])
        self.assertEqual(model.finish_calls, 1)
        self.assertEqual(model.init_count, 2)
        self.assertEqual(model.init_kwargs[0]["language"], "Chinese")
        self.assertEqual(model.init_kwargs[1]["language"], "English")
        assert_transcript_update_invariants(self, switch_events + resume_events)

    def test_set_language_none_returns_future_asr_to_auto_language(self) -> None:
        model = FakeStreamingModel(outputs=["hello", "world"], finish_text="hello")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        session.set_language(None)
        session.ingest_audio(speech)

        self.assertIsNone(session.config.language)
        self.assertIsNone(model.init_kwargs[1]["language"])

    def test_forced_flush_stabilizes_one_speech_segment_without_punctuation_split(self) -> None:
        model = FakeStreamingModel(outputs=[""], finish_text="第一句。第二句。尾巴")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        events = session.flush()

        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["第一句。第二句。尾巴"])

    def test_finish_feeds_received_tail_even_below_asr_cadence(self) -> None:
        model = FakeStreamingModel(outputs=["前段", "前段后段"], finish_text="前段后段")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2
        quiet_tail = np.ones(3_840, dtype=np.float32) * 0.005

        session.ingest_audio(speech)
        session.ingest_audio(quiet_tail)
        events = session.finish()

        self.assertEqual(model.stream_audio_lengths, [16_000, 3_840])
        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["前段后段"])

    def test_low_energy_audio_between_speech_is_not_dropped(self) -> None:
        model = FakeStreamingModel(outputs=["前半", "前半低能量后半"], finish_text="前半低能量后半")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2
        low_energy = np.zeros(16_000, dtype=np.float32)

        session.ingest_audio(speech)
        events = session.ingest_audio(low_energy)

        self.assertEqual(partial_texts(events), ["前半低能量后半"])
        self.assertEqual(model.stream_audio_lengths, [16_000, 16_000])

    def test_short_pause_is_promoted_when_speech_resumes(self) -> None:
        model = FakeStreamingModel(outputs=["前半", "前半后半"])
        session = make_session(model)
        speech_one = np.ones(16_000, dtype=np.float32) * 0.2
        short_pause = np.zeros(3_200, dtype=np.float32)
        speech_two = np.ones(12_800, dtype=np.float32) * 0.2

        session.ingest_audio(speech_one)
        pause_events = session.ingest_audio(short_pause)
        resume_events = session.ingest_audio(speech_two)

        self.assertEqual(partial_texts(pause_events + resume_events), ["前半后半"])
        self.assertEqual(model.stream_audio_lengths, [16_000, 16_000])

    def test_large_transport_frame_is_split_before_asr_cadence(self) -> None:
        model = FakeStreamingModel(outputs=["第一段", "第一段第二段"], finish_text="第一段第二段尾段")
        session = make_session(model)
        payload = np.concatenate(
            [
                np.ones(16_000, dtype=np.float32) * 0.2,
                np.zeros(8_000, dtype=np.float32),
                np.ones(16_000, dtype=np.float32) * 0.2,
            ],
            axis=0,
        )

        events = session.ingest_audio(payload)
        flush_events = session.flush()

        self.assertEqual([segment["text"] for segment in stable_appends(events + flush_events)], ["第一段第二段尾段"])
        self.assertEqual(model.init_count, 1)
        self.assertEqual(model.stream_audio_lengths, [16_000, 16_000, 8_000])

    def test_finish_emits_transcript_final_snapshot(self) -> None:
        model = FakeStreamingModel(outputs=["尾句"], finish_text="尾句")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        events = session.finish()

        final_events = [event for event in events if event["type"] == "transcript_final"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["stable_count"], 1)
        self.assertEqual(final_events[0]["segments"][0]["text"], "尾句")
        self.assertNotIn("final", {event["type"] for event in events})
        self.assertFalse({"partial", "committed"} & {str(event["type"]) for event in events})
        assert_transcript_update_invariants(self, events)


class FakeVadModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)
        self.calls = 0
        self.reset_count = 0

    def __call__(self, frame: object, sample_rate: int) -> np.ndarray:
        self.calls += 1
        probability = self.probabilities.pop(0) if self.probabilities else 0.0
        return np.array([[probability]], dtype=np.float32)

    def reset_states(self) -> None:
        self.reset_count += 1


class SileroVadAdapterTest(unittest.TestCase):
    def test_default_config_uses_onnx_runtime(self) -> None:
        self.assertTrue(SileroVadConfig().use_onnx)

    def test_buffers_until_silero_chunk_is_complete(self) -> None:
        model = FakeVadModel([0.8])
        vad = SileroVadAdapter(
            SileroVadConfig(threshold=0.5, min_speech_ms=32, min_silence_ms=64),
            model=model,
        )

        first = vad.accept(np.ones(256, dtype=np.float32))
        second = vad.accept(np.ones(256, dtype=np.float32))

        self.assertFalse(first.has_speech)
        self.assertEqual(model.calls, 1)
        self.assertTrue(second.speech_started)

    def test_requires_min_speech_and_min_silence(self) -> None:
        model = FakeVadModel([0.8, 0.8, 0.1, 0.1])
        vad = SileroVadAdapter(
            SileroVadConfig(threshold=0.5, min_speech_ms=64, min_silence_ms=64),
            model=model,
        )

        decision = vad.accept(np.ones(512 * 4, dtype=np.float32))

        self.assertTrue(decision.speech_started)
        self.assertTrue(decision.speech_ended)
        self.assertFalse(decision.speech_active)


if __name__ == "__main__":
    unittest.main()
