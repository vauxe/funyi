# coding=utf-8
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from realtime_server import _build_model_load, _parse_args
from qwen3_asr_runtime.realtime_session import EnergyVadConfig, RealtimeASRConfig, RealtimeASRSession
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.vad import SileroVadAdapter, SileroVadConfig


class FakeStreamingModel:
    def __init__(
        self,
        outputs: list[str] | None = None,
        finish_text: str | None = None,
        finish_outputs: list[str] | None = None,
    ) -> None:
        self.outputs = list(outputs or [])
        self.finish_text = finish_text
        self.finish_outputs = list(finish_outputs or [])
        self.init_count = 0
        self.stream_calls = 0
        self.finish_calls = 0
        self.init_kwargs: list[dict[str, object]] = []
        self.stream_audio_lengths: list[int] = []

    @classmethod
    def low_latency_preset_kwargs(cls) -> dict[str, object]:
        return {
            "chunk_size_sec": 1.0,
            "max_window_sec": 20.0,
            "spec_decode": True,
        }

    def init_streaming_state(self, **kwargs: object) -> SimpleNamespace:
        self.init_count += 1
        self.init_kwargs.append(dict(kwargs))
        return SimpleNamespace(text="", language=kwargs.get("language") or "Chinese")

    def streaming_transcribe(self, audio: np.ndarray, state: SimpleNamespace) -> SimpleNamespace:
        self.stream_calls += 1
        self.stream_audio_lengths.append(int(audio.shape[0]))
        if self.outputs:
            state.text = self.outputs.pop(0)
        return state

    def finish_streaming_transcribe(self, state: SimpleNamespace) -> SimpleNamespace:
        self.finish_calls += 1
        if self.finish_outputs:
            state.text = self.finish_outputs.pop(0)
        elif self.finish_text is not None:
            state.text = self.finish_text
        return state


def make_session(model: FakeStreamingModel, *, live_stability_delay_ms: int = 12_000) -> RealtimeASRSession:
    return RealtimeASRSession(
        model,
        config=RealtimeASRConfig(
            language="Chinese",
            live_stability_delay_ms=live_stability_delay_ms,
            vad=EnergyVadConfig(
                speech_threshold=0.01,
                min_speech_ms=60,
                min_silence_ms=300,
            ),
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

    def test_w8a16_can_be_disabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--no-w8a16"]):
            self.assertFalse(_parse_args().w8a16)

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


class RealtimeASRSessionTest(unittest.TestCase):
    def test_idle_silence_does_not_call_asr(self) -> None:
        model = FakeStreamingModel(outputs=["不会出现。"])
        session = make_session(model)

        events = session.ingest_audio(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(events, [])
        self.assertEqual(model.init_count, 0)
        self.assertEqual(model.stream_calls, 0)

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
        self.assertEqual(session.asr_epoch, 1)
        assert_transcript_update_invariants(self, events)

    def test_asr_runs_on_server_side_one_second_cadence(self) -> None:
        model = FakeStreamingModel(outputs=["一秒"])
        session = make_session(model)
        speech = np.ones(4_000, dtype=np.float32) * 0.2

        for _ in range(3):
            self.assertEqual(session.ingest_audio(speech), [])
            self.assertEqual(model.stream_calls, 0)

        events = session.ingest_audio(speech)

        self.assertEqual(partial_texts(events), ["一秒"])
        self.assertEqual(model.stream_calls, 1)
        self.assertEqual(model.stream_audio_lengths, [16_000])
        self.assertEqual(model.init_kwargs[0]["chunk_size_sec"], 1.0)
        self.assertEqual(model.init_kwargs[0]["max_window_sec"], 20.0)
        self.assertEqual(model.init_kwargs[0]["max_prefix_tokens"], 64)
        self.assertTrue(model.init_kwargs[0]["spec_decode"])
        self.assertEqual(model.init_kwargs[0]["language"], "Chinese")

    def test_input_audio_is_processed_in_server_side_200ms_chunks(self) -> None:
        model = FakeStreamingModel(outputs=["一秒"])
        session = make_session(model)
        speech = np.ones(1_600, dtype=np.float32) * 0.2

        for _ in range(9):
            self.assertEqual(session.ingest_audio(speech), [])
            self.assertEqual(model.stream_calls, 0)

        events = session.ingest_audio(speech)

        self.assertEqual(partial_texts(events), ["一秒"])
        self.assertEqual(model.stream_audio_lengths, [16_000])

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

    def test_continuous_speech_advances_stable_cursor_without_resetting_asr_or_vad(self) -> None:
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

    def test_vad_endpoint_after_stable_cursor_advance_stabilizes_only_tail_and_resets_asr(self) -> None:
        model = FakeStreamingModel(
            outputs=["第一秒", "前两秒", "前两秒第三秒"],
            finish_text="前两秒第三秒",
        )
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2
        silence = np.zeros(8_000, dtype=np.float32)

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(silence))

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

    def test_finish_with_unaligned_asr_does_not_stabilize_stale_partial(self) -> None:
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
        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["第一秒", "第二秒"])
        self.assertEqual(updates[-1]["stable_appends"], [])
        self.assertIsNone(updates[-1]["partial"])
        self.assertEqual(final_events[-1]["stable_count"], 2)
        self.assertEqual([segment["text"] for segment in final_events[-1]["segments"]], ["第一秒", "第二秒"])
        assert_transcript_update_invariants(self, events)

    def test_flush_stabilizes_tail_and_next_speech_starts_new_epoch(self) -> None:
        model = FakeStreamingModel(outputs=["尾句"], finish_text="尾句")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        flush_events = session.flush()
        session.ingest_audio(speech)

        self.assertEqual([segment["text"] for segment in stable_appends(flush_events)], ["尾句"])
        self.assertEqual(model.finish_calls, 1)
        self.assertEqual(model.init_count, 2)
        self.assertEqual(session.asr_epoch, 2)

    def test_forced_flush_stabilizes_one_speech_segment_without_punctuation_split(self) -> None:
        model = FakeStreamingModel(outputs=[""], finish_text="第一句。第二句。尾巴")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        events = session.flush()

        self.assertEqual([segment["text"] for segment in stable_appends(events)], ["第一句。第二句。尾巴"])

    def test_vad_endpoint_timestamp_excludes_trailing_silence(self) -> None:
        model = FakeStreamingModel(outputs=["语音段"], finish_text="语音段")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2
        silence = np.zeros(8_000, dtype=np.float32)

        session.ingest_audio(speech)
        events = session.ingest_audio(silence)

        stable = stable_appends(events)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0]["text"], "语音段")
        self.assertEqual(stable[0]["start_ms"], 0)
        self.assertEqual(stable[0]["end_ms"], 1_000)
        self.assertEqual(model.stream_audio_lengths, [16_000])

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

    def test_large_transport_frame_is_split_before_vad(self) -> None:
        model = FakeStreamingModel(outputs=["第一段", "第二段"])
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

        self.assertEqual([segment["text"] for segment in stable_appends(events + flush_events)], ["第一段", "第二段"])
        self.assertEqual(model.init_count, 2)
        self.assertEqual(model.stream_audio_lengths, [16_000, 16_000, 4_000])

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
