# coding=utf-8
from __future__ import annotations

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from realtime_server import (
    CUDA_GRAPH_CAPTURE_LOCK,
    WebSocketSendTimeout,
    _build_model_load,
    _build_session_config,
    _close_websocket,
    _parse_args,
    _prepare_cuda_graph_runtime,
    _prewarm_translation,
    _publish_finish_events,
    _publish_session_events,
    _receive_or_sender_failed,
    _send_queued_events,
    _session_translation_config,
    _translation_capture_lock,
)
from qwen3_asr_runtime.realtime_session import RealtimeASRConfig, RealtimeASRSession
from qwen3_asr_runtime.streaming import RecognitionFrame, TailSelector, TextStabilizer
from qwen3_asr_runtime.realtime_translation import RealtimeTranslationConfig, TranslationModelActor
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.vad import SileroVadAdapter, SileroVadConfig


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
) -> RealtimeASRSession:
    return RealtimeASRSession(
        model,
        config=RealtimeASRConfig(
            language="Chinese",
            live_stability_delay_ms=live_stability_delay_ms,
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
        self.assertTrue(args.translation_prewarm)
        self.assertEqual(args.translation_preview_debounce_ms, 700)

    def test_w8a16_can_be_disabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--no-w8a16"]):
            self.assertFalse(_parse_args().w8a16)

    def test_translation_prewarm_can_be_disabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--no-translation-prewarm"]):
            self.assertFalse(_parse_args().translation_prewarm)

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

    def test_translation_uses_capture_lock_when_prewarm_is_disabled(self) -> None:
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

    def test_translation_actor_needs_no_capture_lock_after_default_cuda_graph_prewarm(self) -> None:
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

    def test_translation_prewarm_runs_before_serving(self) -> None:
        class FakeTranslator:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def warmup(self, texts: object, **kwargs: object) -> list[object]:
                text_list = list(texts)  # type: ignore[arg-type]
                self.calls.append({"texts": text_list, **kwargs})
                return [object() for _ in text_list]

        translator = FakeTranslator()
        actor = TranslationModelActor(translator)
        config = RealtimeTranslationConfig(target_language="English", max_new_tokens=16)

        try:
            _prewarm_translation(actor, config)
        finally:
            actor.close(wait=True)

        self.assertEqual(len(translator.calls), 1)
        self.assertEqual(translator.calls[0]["target_language"], "English")
        self.assertEqual(translator.calls[0]["max_new_tokens"], 16)
        self.assertTrue(translator.calls[0]["sync_cuda"])
        texts = translator.calls[0]["texts"]  # type: ignore[assignment]
        self.assertEqual(len(texts), 3)
        self.assertLess(len(texts[0]), len(texts[1]))  # type: ignore[index]
        self.assertLess(len(texts[1]), len(texts[2]))  # type: ignore[index]

    def test_translation_prewarm_failure_is_startup_error(self) -> None:
        class FakeTranslator:
            def warmup(self, texts: object, **kwargs: object) -> list[object]:
                del texts, kwargs
                return []

        with self.assertRaises(RuntimeError):
            actor = TranslationModelActor(FakeTranslator())
            try:
                _prewarm_translation(actor, RealtimeTranslationConfig(target_language="English"))
            finally:
                actor.close(wait=True)

    def test_start_payload_can_disable_configured_translation(self) -> None:
        config = RealtimeTranslationConfig(target_language="English")

        self.assertIs(_session_translation_config({"type": "start"}, config), config)
        self.assertIsNone(_session_translation_config({"type": "start", "translation": False}, config))
        self.assertIsNone(
            _session_translation_config({"type": "start", "translation": {"enabled": False}}, config)
        )

    def test_start_payload_rejects_mismatched_translation_target(self) -> None:
        config = RealtimeTranslationConfig(target_language="English")

        with self.assertRaises(ValueError):
            _session_translation_config(
                {"type": "start", "translation": {"enabled": True, "target_language": "Japanese"}},
                config,
            )

    def test_service_session_config_uses_start_payload_context(self) -> None:
        config = _build_session_config({"type": "start", "context": "meeting", "language": "Chinese"})

        self.assertEqual(config.context, "meeting")
        self.assertEqual(config.language, "Chinese")


class RealtimeServerTranslationOrderingTest(unittest.IsolatedAsyncioTestCase):
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


class RealtimeASRSessionTest(unittest.TestCase):
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
