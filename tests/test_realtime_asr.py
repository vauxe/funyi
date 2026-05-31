# coding=utf-8
from __future__ import annotations

import ast
import asyncio
import json
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
import wave

import numpy as np
import pytest

from realtime_server import (
    CUDA_GRAPH_CAPTURE_LOCK,
    _DebugAudioRecorder,
    _PcmDebugSummary,
    TranslationServiceConfig,
    WebSocketSendTimeout,
    _build_realtime_session_config,
    _build_model_load,
    _build_translation,
    _close_websocket,
    _configure_logging,
    _format_event_log_summary,
    _format_pcm_debug_stats,
    _parse_args,
    _parse_language_config_update,
    _pcm_s16le_bytes,
    _prepare_cuda_graph_runtime,
    _prewarm_timestamp_runtime,
    _prewarm_translation_runtime,
    _publish_finish_events,
    _queue_event,
    _receive_start,
    _publish_session_events,
    _receive_or_sender_failed,
    _send_queued_events,
    _session_translation_config,
    _should_log_realtime_event,
    _streaming_ready_payload,
    _timestamp_prewarm_audio,
    _translation_capture_lock,
    _uvicorn_log_level,
)
from qwen3_asr_runtime.language_support import (
    HYMT_MODEL_CARD_LANGUAGES,
    QWEN3_ASR_MODEL_CARD_LANGUAGES,
    QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES,
)
from qwen3_asr_runtime.realtime_session import (
    RealtimeASRConfig,
    RealtimeASRSession,
    RealtimeConnectionSession,
    _SourceTimeline,
)
from qwen3_asr_runtime.speech_gate import SpeechGate, SpeechGateConfig
from qwen3_asr_runtime.streaming import RecognitionFrame, TailSelector, TextStabilizer
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.utils import SUPPORTED_LANGUAGES
from qwen3_asr_runtime.vad import SileroVadAdapter, SileroVadConfig, VadDecision


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
            "unfixed_token_num": 5,
            "max_window_sec": 20.0,
            "max_prefix_tokens": 64,
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


class FakeVadAdapter:
    def __init__(self, decisions: list[VadDecision]) -> None:
        self.decisions = list(decisions)
        self.accept_lengths: list[int] = []
        self._speech_active = False

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def reset(self) -> None:
        self._speech_active = False

    def accept(self, audio: np.ndarray) -> VadDecision:
        self.accept_lengths.append(int(audio.shape[0]))
        decision = self.decisions.pop(0) if self.decisions else VadDecision(speech_active=self._speech_active)
        self._speech_active = bool(decision.speech_active)
        return decision


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


def assert_transcript_update_invariants(events: list[dict[str, object]]) -> None:
    revision = 0
    stable_count = 0
    for event in transcript_updates(events):
        assert int(event["revision"]) > revision
        revision = int(event["revision"])
        assert event["stable_base"] == stable_count
        appends = event.get("stable_appends")
        assert isinstance(appends, list)
        stable_count += len(appends)
        assert event["stable_count"] == stable_count
        assert "partial" in event
        partial = event.get("partial")
        assert partial is None or isinstance(partial, dict)


class TestTranscriptStore:
    def test_segments_are_appended_with_monotonic_timestamps(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(text="第一句。", start_ms=0, end_ms=1200, language="Chinese")
        store.append_stable_segment(text="Second.", start_ms=1000, end_ms=2300, language="English")

        assert [segment.text for segment in store.stable_segments] == ['第一句。', 'Second.']
        assert store.stable_segments[1].start_ms == first.end_ms
        assert store.stable_count == 2

    def test_stable_segment_text_preserves_boundary_whitespace(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(text="hello ", start_ms=0, end_ms=1000, language="English")
        second = store.append_stable_segment(text="world", start_ms=1000, end_ms=2000, language="English")

        assert ''.join((segment.text for segment in store.stable_segments)) == 'hello world'
        event = store.update_event(stable_base=0, stable_appends=[first, second])
        assert ''.join((str(segment['text']) for segment in event['stable_appends'])) == 'hello world'

    def test_update_event_uses_stable_cursor_and_replaceable_partial(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(text="稳定", start_ms=0, end_ms=1000, language="Chinese")
        event = store.update_event(stable_base=0, stable_appends=[segment])

        assert event['type'] == 'transcript_update'
        assert event['stable_base'] == 0
        assert event['stable_count'] == 1
        assert event['stable_appends'][0]['text'] == '稳定'
        assert event['partial'] is None

    def test_update_event_rejects_stale_stable_base(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(text="稳定", start_ms=0, end_ms=1000, language="Chinese")

        with pytest.raises(ValueError):
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

        assert update['stable_appends'][0]['start_ms'] is None
        assert update['stable_appends'][0]['end_ms'] is None
        assert update['stable_appends'][0]['timing_status'] == 'pending'

        timing_update = store.update_segment_timing(
            source_segment_id="seg_000001",
            start_ms=120,
            end_ms=860,
            timing_status="aligned",
        )
        final = store.final_event()

        assert timing_update == {
            'type': 'transcript_timing_update',
            'source_segment_id': 'seg_000001',
            'start_ms': 120,
            'end_ms': 860,
            'timing_status': 'aligned',
        }
        assert final['segments'][0]['start_ms'] == 120
        assert final['segments'][0]['end_ms'] == 860
        assert final['segments'][0]['timing_status'] == 'aligned'

    def test_out_of_order_timing_patch_does_not_overlap_later_segment(self) -> None:
        store = TranscriptStore(transcript_id="t1")
        for _ in range(2):
            store.append_stable_segment(
                text="句", start_ms=None, end_ms=None, language="Chinese", timing_status="pending"
            )

        # Patch the SECOND segment first (out of index order).
        store.update_segment_timing(
            source_segment_id="seg_000002", start_ms=1000, end_ms=2000, timing_status="aligned"
        )
        # Now the first segment aligns with an end that would overlap into seg2's start.
        first = store.update_segment_timing(
            source_segment_id="seg_000001", start_ms=0, end_ms=1500, timing_status="aligned"
        )

        # seg1.end is clamped to seg2.start so the public timeline stays non-overlapping.
        assert first['start_ms'] == 0
        assert first['end_ms'] == 1000
        assert store.stable_segments[0].end_ms <= store.stable_segments[1].start_ms

    def test_unbounded_store_final_event_reports_revision_without_full_segments(self) -> None:
        store = TranscriptStore(transcript_id="t1", keep_segments=False)
        segment = store.append_stable_segment(text="稳定", start_ms=0, end_ms=1000, language="Chinese")
        update = store.update_event(stable_base=0, stable_appends=[segment])
        final = store.final_event()

        assert update["stable_appends"][0]["text"] == "稳定"
        assert store.stable_segments == []
        assert final["stable_count"] == 1
        assert final["final_revision"] == final["revision"]
        assert "segments" not in final


class TestTextStabilizer:
    def test_repeated_prefix_is_required_before_stabilizing(self) -> None:
        stabilizer = TextStabilizer()

        first = stabilizer.observe("第一秒", end_sample=16_000, can_commit=True)
        second = stabilizer.observe("第一秒第二秒", end_sample=32_000, can_commit=True)

        assert first.stable_text == ''
        assert first.partial_text == '第一秒'
        assert first.stable_end_sample is None
        assert second.stable_text == '第一秒'
        assert second.partial_text == '第二秒'
        assert second.stable_end_sample == 16000

    def test_stable_prefix_does_not_split_ascii_word(self) -> None:
        stabilizer = TextStabilizer("hello wor", 16_000)

        update = stabilizer.observe("hello world today", end_sample=32_000, can_commit=True)

        assert update.stable_text == 'hello'
        assert update.partial_text == 'world today'
        assert update.stable_end_sample == 16000


class TestTailSelector:
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

        assert tail.aligned
        assert tail.text == '后段'

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

        assert tail.aligned
        assert tail.text == '草稿后段'

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

        assert tail.aligned
        assert tail.text == '有轻微改写后段继续'

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

        assert tail.aligned
        assert tail.text == '出来创业'

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

        assert tail.aligned
        assert tail.text == '有一些广告'

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

        assert not tail.aligned

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

        assert tail.aligned
        assert tail.text == '谢谢大家下一句'

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

        assert tail.aligned
        assert tail.text == '后段'

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

        assert tail.aligned
        assert tail.text == '谢谢大家下一句'


class TestRealtimeServerCli:
    def test_gpu_runtime_is_default(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            args = _parse_args()

        assert args.device_map is None
        assert args.cuda_graph is None
        assert args.flashinfer is None
        assert args.fused_rmsnorm is None
        assert args.fused_linears is None
        assert args.w8a16 is None
        assert args.cuda_graph_prewarm
        assert args.cuda_graph_prewarm_language == 'Chinese'
        assert args.cuda_graph_prewarm_window_sec == 20.0
        assert args.cuda_graph_prewarm_prefix_tokens == 64
        assert args.timestamp_model is None
        assert args.timestamp_local_files_only
        assert args.timestamp_pad_ms == 500
        assert args.timestamp_finish_timeout_ms == 30000
        assert args.translation_model is None
        assert args.translation_preview_debounce_ms == 700
        assert args.translation_stable_batch_size == 1
        assert not args.translation_sample
        assert args.log_level == "info"
        assert not args.save_debug_audio
        assert args.debug_audio_dir == "local_data/realtime_debug_audio"

    def test_debug_log_level_can_be_configured(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--log-level", "debug"]):
            args = _parse_args()

        assert args.log_level == "debug"
        root = logging.getLogger()
        realtime_logger = logging.getLogger("realtime_server")
        runtime_logger = logging.getLogger("qwen3_asr_runtime")
        uvicorn_logger = logging.getLogger("uvicorn")
        websockets_logger = logging.getLogger("websockets")
        old_root_level = root.level
        old_realtime_level = realtime_logger.level
        old_runtime_level = runtime_logger.level
        old_uvicorn_level = uvicorn_logger.level
        old_websockets_level = websockets_logger.level
        try:
            with patch("realtime_server.logging.basicConfig") as basic_config:
                assert _configure_logging(args.log_level) == logging.DEBUG
            basic_config.assert_called_once()
            assert root.level == logging.INFO
            assert realtime_logger.level == logging.DEBUG
            assert runtime_logger.level == logging.DEBUG
            assert uvicorn_logger.level == logging.INFO
            assert websockets_logger.level == logging.INFO
        finally:
            root.setLevel(old_root_level)
            realtime_logger.setLevel(old_realtime_level)
            runtime_logger.setLevel(old_runtime_level)
            uvicorn_logger.setLevel(old_uvicorn_level)
            websockets_logger.setLevel(old_websockets_level)

    def test_uvicorn_stays_info_when_service_debug_is_enabled(self) -> None:
        assert _uvicorn_log_level(logging.DEBUG) == "info"
        assert _uvicorn_log_level(logging.WARNING) == "warning"

    def test_pcm_debug_stats_describe_frame_level_audio(self) -> None:
        audio = np.array([0, 32767, -32768], dtype=np.int16)

        stats = _format_pcm_debug_stats(audio)

        assert "samples=3" in stats
        assert "duration_ms=0" in stats
        assert "peak=1.0000" in stats
        assert "zero_pct=33.3" in stats

    def test_pcm_debug_summary_throttles_frame_logs(self) -> None:
        summary = _PcmDebugSummary(session_id="s1", sample_rate=16_000, interval_ms=1000)
        half_second = np.ones(8000, dtype=np.float32) * 0.5

        assert summary.accept(half_second, byte_count=16_000) is None
        message = summary.accept(half_second, byte_count=16_000)

        assert message is not None
        assert "PCM summary session_id=s1" in message
        assert "frames=2" in message
        assert "bytes=32000" in message
        assert "total_ms=1000" in message
        assert "samples=16000" in message

    def test_transcript_event_log_summary_includes_text_snapshot(self) -> None:
        summary = _format_event_log_summary(
            {
                "type": "transcript_update",
                "revision": 2,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [{"text": "稳定"}],
                "partial": {"text": "这这这"},
            }
        )

        assert "type=transcript_update" in summary
        assert "stable_texts=['稳定']" in summary
        assert "partial='这这这'" in summary

    def test_realtime_event_debug_log_filter_skips_partial_only_updates(self) -> None:
        assert not _should_log_realtime_event({"type": "transcript_update", "partial": {"text": "draft"}})
        assert _should_log_realtime_event({"type": "transcript_update", "stable_appends": [{"text": "done"}]})
        assert _should_log_realtime_event({"type": "error", "error": "bad"})
        assert not _should_log_realtime_event({"type": "translation_preview", "text": "draft"})

    def test_save_debug_audio_cli_uses_default_directory(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--save-debug-audio"]):
            args = _parse_args()

        assert args.save_debug_audio
        assert args.debug_audio_dir == "local_data/realtime_debug_audio"

    def test_debug_audio_directory_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["realtime_server.py", "--model", "model", "--save-debug-audio", "--debug-audio-dir", "/tmp/funyi-audio"],
        ):
            args = _parse_args()

        assert args.save_debug_audio
        assert args.debug_audio_dir == "/tmp/funyi-audio"

    def test_pcm_s16le_bytes_preserves_int16_samples(self) -> None:
        audio = np.array([0, 32767, -32768], dtype=np.int16)

        assert _pcm_s16le_bytes(audio) == b"\x00\x00\xff\x7f\x00\x80"

    def test_debug_audio_recorder_writes_wav_file(self, tmp_path: Path) -> None:
        recorder = _DebugAudioRecorder(tmp_path, session_id="bad/session id")
        recorder.write(np.array([0, 32767, -32768], dtype=np.int16))
        path = recorder.path
        recorder.close()

        assert path.parent == tmp_path
        assert path.name.startswith("bad_session_id-")
        with wave.open(str(path), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() == 3
            assert wav.readframes(3) == b"\x00\x00\xff\x7f\x00\x80"

    def test_translation_sampling_can_be_enabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-sample"]):
            assert _parse_args().translation_sample

    def test_translation_stable_batch_size_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["realtime_server.py", "--model", "model", "--translation-stable-batch-size", "4"],
        ):
            assert _parse_args().translation_stable_batch_size == 4

    def test_w8a16_can_be_disabled(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--no-w8a16"]):
            assert not _parse_args().w8a16

    def test_translation_model_flag_uses_default_model_when_value_is_omitted(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-model"]):
            args = _parse_args()

        assert args.translation_model == 'tencent/Hy-MT2-1.8B'
        assert not args.translation_trust_remote_code
        assert args.translation_decode_backend == 'fixed_mask'
        assert args.translation_w8a16
        assert args.translation_fused_rmsnorm

    def test_translation_model_can_be_configured(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--translation-model", "local/hymt"]):
            assert _parse_args().translation_model == 'local/hymt'

    def test_translation_model_enables_translation_without_default_target(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "realtime_server.py",
                "--model",
                "model",
                "--translation-model",
                "local/hymt",
                "--translation-model-revision",
                "abc123",
                "--no-translation-local-files-only",
                "--translation-trust-remote-code",
                "--no-translation-w8a16",
                "--no-translation-fused-rmsnorm",
            ],
        ):
            args = _parse_args()

        translator = object()
        with patch("realtime_server.HYMTTranslator", return_value=translator) as translator_class:
            built_translator, config = _build_translation(args)

        assert built_translator is translator
        assert config is not None
        assert translator_class.call_args.args[0] == 'local/hymt'
        kwargs = translator_class.call_args.kwargs
        assert kwargs['model_revision'] == 'abc123'
        assert not kwargs['local_files_only']
        assert kwargs['trust_remote_code']
        assert not kwargs['w8a16']
        assert not kwargs['fused_rmsnorm']

    def test_translation_prewarm_uses_actor_and_configured_target_buckets(self) -> None:
        class FakeTranslationActor:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def warmup(self, texts: list[str], **kwargs: object) -> list[object]:
                self.calls.append({"texts": list(texts), **kwargs})
                return [object() for _ in texts]

        actor = FakeTranslationActor()

        _prewarm_translation_runtime(
            actor,  # type: ignore[arg-type]
            TranslationServiceConfig(max_new_tokens=16, stable_batch_size=2),
        )

        assert [(call["target_language"], call["batch_size"]) for call in actor.calls] == [
            ("Chinese", 1),
            ("Chinese", 2),
        ]
        assert all(call["source_language"] == "" for call in actor.calls)
        assert all(call["max_new_tokens"] == 16 for call in actor.calls)
        assert all(call["sync_cuda"] is True for call in actor.calls)
        assert all(len(call["texts"]) == 3 for call in actor.calls)

    def test_timestamp_prewarm_uses_actor_before_service_ready(self) -> None:
        class FakeTimestampActor:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def warmup(self, audio: np.ndarray, *, text: str, language: str) -> None:
                self.calls.append({"audio": audio, "text": text, "language": language})

        actor = FakeTimestampActor()

        _prewarm_timestamp_runtime(actor)  # type: ignore[arg-type]

        assert len(actor.calls) == 1
        assert actor.calls[0]["text"] == "你好。"
        assert actor.calls[0]["language"] == "Chinese"
        audio = actor.calls[0]["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.shape == (16000,)

    def test_timestamp_prewarm_audio_is_synthetic_float32(self) -> None:
        audio = _timestamp_prewarm_audio(0.25)

        assert audio.dtype == np.float32
        assert audio.shape == (4000,)
        assert float(np.max(np.abs(audio))) <= 0.011

    def test_asr_supported_languages_follow_qwen_model_card(self) -> None:
        assert tuple(SUPPORTED_LANGUAGES) == QWEN3_ASR_MODEL_CARD_LANGUAGES

    def test_desktop_language_options_follow_backend_model_card_lists(self) -> None:
        assert _desktop_language_options('ASR_LANGUAGE_OPTIONS') == QWEN3_ASR_MODEL_CARD_LANGUAGES
        assert _desktop_language_options('TRANSLATION_TARGET_LANGUAGE_OPTIONS') == HYMT_MODEL_CARD_LANGUAGES

    def test_transformers_load_kwargs_are_default(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            backend, kwargs = _build_model_load(_parse_args())

        assert backend == 'transformers'
        assert kwargs['device_map'] == 'cuda:0'
        assert kwargs['cuda_graph']
        assert kwargs['cuda_graph_len_bucket'] == 64
        assert kwargs['flashinfer']
        assert kwargs['fused_rmsnorm']
        assert kwargs['fused_linears']
        # W8A16 is OFF by default for the streaming service: it slows the
        # prefill-bound streaming path ~3x at equal CER (recheck_w8a16_*).
        assert not kwargs['quantized_linears']

    def test_w8a16_flag_forces_quantized_linears_on(self) -> None:
        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model", "--w8a16"]):
            _, kwargs = _build_model_load(_parse_args())
        assert kwargs['quantized_linears']

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

        assert model.calls == [{'language': 'Chinese', 'max_window_sec': 20.0, 'max_prefix_tokens': 64}]

    def test_cuda_graph_prewarm_failure_is_startup_error(self) -> None:
        class FakeModel:
            def prewarm_realtime_cuda_graph(self, **kwargs: object) -> bool:
                del kwargs
                return False

        with patch.object(sys, "argv", ["realtime_server.py", "--model", "model"]):
            args = _parse_args()

        with pytest.raises(RuntimeError):
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

        assert lock is CUDA_GRAPH_CAPTURE_LOCK

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

        assert model.calls == 1
        assert lock is None

    def test_start_payload_without_target_disables_session_translation(self) -> None:
        config = TranslationServiceConfig()

        assert _session_translation_config({'type': 'start'}, config) is None

    def test_start_payload_rejects_empty_translation_target(self) -> None:
        config = TranslationServiceConfig()

        with pytest.raises(ValueError, match='target_language must not be empty'):
            _session_translation_config({"type": "start", "target_language": ""}, config)

    def test_start_payload_rejects_translation_target_when_translation_is_not_configured(self) -> None:
        assert _session_translation_config({'type': 'start'}, None) is None

        with pytest.raises(ValueError, match='translation model to be configured'):
            _session_translation_config({"type": "start", "target_language": "English"}, None)

    def test_start_payload_can_enable_session_translation_target(self) -> None:
        config = TranslationServiceConfig(
            max_new_tokens=16,
        )

        session_config = _session_translation_config(
            {"type": "start", "target_language": "Japanese"},
            config,
        )
        assert session_config is not None
        assert session_config.target_language == 'Japanese'
        assert session_config.max_new_tokens == 16

    def test_start_payload_rejects_translation_target_outside_hymt_model_card(self) -> None:
        config = TranslationServiceConfig()

        with pytest.raises(ValueError, match='Unsupported target_language'):
            _session_translation_config({"type": "start", "target_language": "Swedish"}, config)

    def test_start_payload_normalizes_translation_target(self) -> None:
        config = TranslationServiceConfig()

        session_config = _session_translation_config(
            {"type": "start", "target_language": "traditional chinese"},
            config,
        )

        assert session_config is not None
        assert session_config.target_language == 'Traditional Chinese'

    def test_service_session_config_uses_start_payload_context(self) -> None:
        config = _build_realtime_session_config(
            {"type": "start", "context": "meeting", "language": "Chinese"},
        )

        assert config.context == 'meeting'
        assert config.language == 'Chinese'
        assert config.force_align_timestamps is True

    def test_set_language_command_normalizes_language_choices(self) -> None:
        update = _parse_language_config_update(
            {"type": "set_language", "language": "japanese", "target_language": "traditional chinese"},
            TranslationServiceConfig(),
        )

        assert update == {'language': 'Japanese', 'target_language': 'Traditional Chinese'}

    def test_set_language_command_allows_auto_asr_and_translation_off(self) -> None:
        update = _parse_language_config_update(
            {"type": "set_language", "language": "", "target_language": None},
            None,
        )

        assert update == {'language': None, 'target_language': None}

    def test_set_language_command_rejects_target_without_translation_model(self) -> None:
        with pytest.raises(ValueError, match='translation model to be configured'):
            _parse_language_config_update({"type": "set_language", "target_language": "English"}, None)

    def test_set_language_command_rejects_unknown_field(self) -> None:
        with pytest.raises(ValueError, match='Unsupported set_language command field'):
            _parse_language_config_update(
                {"type": "set_language", "language": "English", "extra": True},
                TranslationServiceConfig(),
            )

    def test_aligned_session_config_rejects_languages_without_forced_aligner_support(self) -> None:
        with pytest.raises(ValueError, match='Forced aligner does not support source language'):
            _build_realtime_session_config(
                {"type": "start", "language": "Arabic"},
            )

        assert 'Japanese' in QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES
        assert 'Arabic' not in QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES

    def test_realtime_session_config_accepts_only_aligned_realtime_mode(self) -> None:
        config = _build_realtime_session_config(
            {"type": "start", "language": "Chinese", "context": "meeting"},
        )

        assert config.language == "Chinese"
        assert config.context == "meeting"
        assert config.force_align_timestamps is True

        with pytest.raises(ValueError, match="aligned_windowed"):
            _build_realtime_session_config(
                {"type": "start", "realtime_commit_mode": "asr_only"},
            )

    def test_streaming_ready_payload_declares_timing_patch_event(self) -> None:
        ready = _streaming_ready_payload(RealtimeASRConfig(live_stability_delay_ms=12_000))

        assert ready["mode"] == "aligned_windowed"
        assert ready["stable"]["source"] == "asr_streaming_text_and_forced_aligner"  # type: ignore[index]
        assert ready["stable"]["patch_event"] == "transcript_timing_update"  # type: ignore[index]
        assert ready["stable"]["live_stability_delay_ms"] == 12_000  # type: ignore[index]

    def test_set_language_command_accepts_only_forced_aligner_source_languages(self) -> None:
        japanese = _parse_language_config_update(
            {"type": "set_language", "language": "Japanese"},
            TranslationServiceConfig(),
        )

        assert japanese == {'language': 'Japanese'}
        with pytest.raises(ValueError, match='Forced aligner does not support source language'):
            _parse_language_config_update(
                {"type": "set_language", "language": "Arabic"},
                TranslationServiceConfig(),
            )


class TestRealtimeServerTranslationOrdering:
    async def test_receive_start_normalizes_supported_language(self) -> None:
        class FakeWebSocket:
            async def receive(self) -> dict[str, object]:
                return {"text": '{"type":"start","language":"japanese"}'}

        payload = await _receive_start(FakeWebSocket())

        assert payload is not None
        assert payload['language'] == 'Japanese'  # type: ignore[index]

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

        assert payload is None
        assert websocket.closed_code == 1003
        assert 'Unsupported language' in websocket.sent[0]
        assert json.loads(websocket.sent[0])['fatal'] is True

    async def test_receive_start_rejects_language_without_forced_aligner_support(self) -> None:
        class FakeWebSocket:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.closed_code: int | None = None

            async def receive(self) -> dict[str, object]:
                return {"text": '{"type":"start","language":"Arabic"}'}

            async def send_text(self, text: str) -> None:
                self.sent.append(text)

            async def close(self, code: int) -> None:
                self.closed_code = code

        websocket = FakeWebSocket()

        payload = await _receive_start(websocket)

        assert payload is None
        assert websocket.closed_code == 1003
        assert 'Forced aligner does not support source language' in websocket.sent[0]
        assert json.loads(websocket.sent[0])['fatal'] is True

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

        assert payload is None
        assert websocket.closed_code == 1003
        assert 'Unsupported start command field' in websocket.sent[0]
        assert 'unsupported' in websocket.sent[0]
        assert json.loads(websocket.sent[0])['fatal'] is True

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

        with pytest.raises(WebSocketSendTimeout):
            await _send_queued_events(websocket, queue, send_timeout_sec=0.01)

        assert websocket.send_started.is_set()
        assert queue.qsize() == 0

    async def test_queue_event_applies_backpressure_until_live_sender_drains(self) -> None:
        # A full queue must not be fatal: ASR already consumed the audio, so dropping the
        # event would skip published transcript text. The producer blocks until the live
        # sender frees a slot.
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=1)
        await queue.put({"type": "ready"})

        async def drain_then_idle() -> None:
            await asyncio.sleep(0.02)
            await queue.get()
            await asyncio.Future()

        sender_task = asyncio.create_task(drain_then_idle())
        try:
            await asyncio.wait_for(
                _queue_event(queue, {"type": "transcript_update"}, sender_task=sender_task),
                timeout=1.0,
            )
            assert queue.qsize() == 1
            assert (await queue.get())["type"] == "transcript_update"
        finally:
            sender_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sender_task

    async def test_queue_event_raises_when_sender_already_stopped(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=1)
        await queue.put({"type": "ready"})

        async def fail_sender() -> None:
            raise WebSocketSendTimeout("client did not consume output")

        sender_task = asyncio.create_task(fail_sender())
        await asyncio.sleep(0.01)
        assert sender_task.done()

        with pytest.raises(WebSocketSendTimeout):
            await _queue_event(queue, {"type": "transcript_update"}, sender_task=sender_task)

    async def test_queue_event_raises_when_sender_dies_while_waiting(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=1)
        await queue.put({"type": "ready"})

        async def fail_later() -> None:
            await asyncio.sleep(0.02)
            raise WebSocketSendTimeout("client stalled while queue stayed full")

        sender_task = asyncio.create_task(fail_later())
        with pytest.raises(WebSocketSendTimeout):
            await asyncio.wait_for(
                _queue_event(queue, {"type": "transcript_update"}, sender_task=sender_task),
                timeout=1.0,
            )

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

        with pytest.raises(WebSocketSendTimeout):
            await _receive_or_sender_failed(websocket, sender_task)

        assert websocket.receive_cancelled

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

        assert websocket.close_started.is_set()

    async def test_pending_old_preview_is_not_queued_after_new_source_revision(self) -> None:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        class FakeTranslation:
            async def accept_source_event(self, event: dict[str, object]) -> None:
                self.check_source_event(event)
                await queue.put({"type": "translation_preview", "source_revision": 1, "text": "old"})

            def check_source_event(self, event: dict[str, object]) -> None:
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
        assert first['type'] == 'translation_preview'
        assert first['source_revision'] == 1
        assert second['type'] == 'transcript_update'
        assert second['revision'] == 2

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
        assert [event['type'] for event in events] == [
            'translation_preview',
            'transcript_update',
            'translation_stable',
            'transcript_final',
        ]
        assert events[0]['source_revision'] == 1
        assert events[1]['revision'] == 2

    async def test_publish_finish_emits_timing_patch_before_terminal_final(self) -> None:
        # transcript_final is terminal; any transcript_timing_update for a stable segment
        # must reach the client before it.
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        await _publish_finish_events(
            queue,
            None,
            [
                {
                    "type": "transcript_timing_update",
                    "source_segment_id": "seg_000001",
                    "start_ms": 0,
                    "end_ms": 900,
                    "timing_status": "aligned",
                },
                {"type": "transcript_final", "revision": 3, "stable_count": 1},
            ],
        )

        types = []
        while not queue.empty():
            types.append((await queue.get())["type"])
        assert types == ["transcript_timing_update", "transcript_final"]


class TestSpeechGate:
    def test_initial_silence_produces_no_speech_events(self) -> None:
        gate = SpeechGate(
            vad=FakeVadAdapter([VadDecision(speech_active=False)]),
            config=SpeechGateConfig(pre_roll_ms=400),
        )

        events = gate.accept(np.zeros(16_000, dtype=np.float32))

        assert events == []

    def test_speech_start_includes_bounded_preroll(self) -> None:
        gate = SpeechGate(
            vad=FakeVadAdapter(
                [
                    VadDecision(speech_active=False),
                    VadDecision(speech_started=True, speech_active=True, speech_start_sample=16_000),
                ]
            ),
            config=SpeechGateConfig(pre_roll_ms=400),
        )
        gate.accept(np.zeros(16_000, dtype=np.float32))

        events = gate.accept(np.ones(16_000, dtype=np.float32) * 0.2)

        assert len(events) == 1
        assert events[0].type == 'speech_start'
        assert events[0].start_sample == 9_600
        assert events[0].end_sample == 32_000
        assert events[0].audio.shape[0] == 22_400

    def test_short_speech_in_one_chunk_emits_start_and_end(self) -> None:
        gate = SpeechGate(
            vad=FakeVadAdapter(
                [
                    VadDecision(
                        speech_started=True,
                        speech_ended=True,
                        speech_active=False,
                        speech_start_sample=2_000,
                        speech_end_sample=10_000,
                    )
                ]
            ),
            config=SpeechGateConfig(pre_roll_ms=400),
        )

        events = gate.accept(np.ones(16_000, dtype=np.float32) * 0.2)

        assert [event.type for event in events] == ['speech_start', 'speech_end']
        assert events[0].start_sample == 0
        assert events[0].end_sample == 10_000
        assert events[1].start_sample == 10_000
        assert not gate.speech_active

    def test_speech_restart_in_same_chunk_does_not_duplicate_previous_turn_audio(self) -> None:
        gate = SpeechGate(
            vad=FakeVadAdapter(
                [
                    VadDecision(speech_started=True, speech_active=True, speech_start_sample=0),
                    VadDecision(
                        speech_started=True,
                        speech_ended=True,
                        speech_active=True,
                        speech_start_sample=28_000,
                        speech_end_sample=20_000,
                    ),
                ]
            ),
            config=SpeechGateConfig(pre_roll_ms=400),
        )
        gate.accept(np.ones(16_000, dtype=np.float32) * 0.2)

        events = gate.accept(np.ones(16_000, dtype=np.float32) * 0.2)

        assert [event.type for event in events] == ['speech_audio', 'speech_end', 'speech_start']
        assert events[0].end_sample == 20_000
        assert events[2].start_sample == 21_600
        assert events[2].start_sample >= events[1].end_sample


class TestRealtimeASRSession:
    def test_source_timeline_coalesces_contiguous_source_clock_spans(self) -> None:
        timeline = _SourceTimeline()

        timeline.append(16_000, source_start_sample=0)
        timeline.append(8_000, source_start_sample=16_000)
        timeline.append(4_000)
        timeline.append(8_000, source_start_sample=40_000)

        assert len(timeline._spans) == 2
        assert timeline.source_start_sample(20_000) == 20_000
        assert timeline.source_end_sample(28_000) == 28_000
        assert timeline.source_start_sample(30_000) == 42_000

    def test_source_timeline_maps_gap_boundary_start_and_end_distinctly(self) -> None:
        timeline = _SourceTimeline()

        timeline.append(16_000, source_start_sample=0)
        timeline.append(8_000, source_start_sample=40_000)  # 24_000-sample source-clock gap

        # Inside the pre-gap span both edges map straight through.
        assert timeline.source_start_sample(8_000) == 8_000
        assert timeline.source_end_sample(8_000) == 8_000

        # At the local boundary between spans: a segment that *ends* here keeps the pre-gap
        # source end, while one that *starts* here jumps to the post-gap source start. This
        # discontinuity is what keeps timestamp crops on the right side of skipped silence.
        assert timeline.source_end_sample(16_000) == 16_000
        assert timeline.source_start_sample(16_000) == 40_000

        # Inside the post-gap span both edges map onto post-gap source samples.
        assert timeline.source_start_sample(20_000) == 44_000
        assert timeline.source_end_sample(20_000) == 44_000

    def test_service_default_keeps_stable_history_conservative(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "前两秒", "前两秒第三秒"])
        config = RealtimeASRConfig(language="Chinese")
        session = RealtimeASRSession(model, config=config)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        assert stable_appends(events) == []
        assert partial_texts(events)[-1] == '前两秒第三秒'
        assert_transcript_update_invariants(events)

    def test_force_align_timestamp_mode_emits_pending_stable_segment_and_hidden_timing_job(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = make_session(model, live_stability_delay_ms=0, force_align_timestamps=True)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events = session.ingest_audio(speech) + session.ingest_audio(speech)
        stable = stable_appends(events)
        jobs = session.consume_stable_timing_jobs_for_events(events)

        assert len(stable) == 1
        assert stable[0]['start_ms'] is None
        assert stable[0]['end_ms'] is None
        assert stable[0]['timing_status'] == 'pending'
        assert len(jobs) == 1
        assert jobs[0].source_segment_id == stable[0]['id']
        assert jobs[0].source_text == '第一秒'
        assert (jobs[0].start_sample, jobs[0].end_sample) == (0, 16000)
        assert session.consume_stable_timing_jobs_for_events(events) == []

    def test_silence_is_not_an_asr_input_gate(self) -> None:
        model = FakeStreamingModel(outputs=["低能量语音。"])
        session = make_session(model)

        events = session.ingest_audio(np.zeros(16_000, dtype=np.float32))

        assert partial_texts(events) == ['低能量语音。']
        assert model.init_count == 1
        assert model.stream_calls == 1
        assert model.stream_audio_lengths == [16000]
        assert_transcript_update_invariants(events)

    def test_turn_time_origin_keeps_absolute_transcript_timestamps(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = RealtimeASRSession(
            model,
            config=RealtimeASRConfig(
                language="Chinese",
                live_stability_delay_ms=0,
            ),
            time_origin_sample=32_000,
        )
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events = session.ingest_audio(speech) + session.ingest_audio(speech)

        stable = stable_appends(events)
        assert [segment['text'] for segment in stable] == ['第一秒']
        assert stable[0]['start_ms'] == 2000
        assert stable[0]['end_ms'] == 3000
        assert_transcript_update_invariants(events)

    def test_noncontiguous_source_audio_maps_timestamps_without_model_silence(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = RealtimeASRSession(
            model,
            config=RealtimeASRConfig(
                language="Chinese",
                live_stability_delay_ms=0,
            ),
        )
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        first_events = session.ingest_audio(speech, source_start_sample=0)
        resume_events = session.ingest_audio(speech, source_start_sample=32_000)

        stable = stable_appends(first_events + resume_events)
        updates = transcript_updates(resume_events)
        assert [segment['text'] for segment in stable] == ['第一秒']
        assert stable[0]['start_ms'] == 0
        assert stable[0]['end_ms'] == 1000
        assert updates[-1]['partial']['text'] == '第二秒'  # type: ignore[index]
        assert updates[-1]['partial']['start_ms'] == 2000  # type: ignore[index]
        assert updates[-1]['partial']['end_ms'] == 3000  # type: ignore[index]
        assert model.stream_audio_lengths == [16_000, 16_000]
        assert_transcript_update_invariants(first_events + resume_events)


class TestRealtimeConnectionSession:
    def test_connection_timing_jobs_are_consumed_after_runtime_handoff(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = RealtimeConnectionSession(
            model,
            config=RealtimeASRConfig(
                language="Chinese",
                live_stability_delay_ms=0,
                force_align_timestamps=True,
            ),
            speech_gate=SpeechGate(
                vad=FakeVadAdapter(
                    [
                        VadDecision(
                            speech_started=True,
                            speech_active=True,
                            speech_start_sample=0,
                        ),
                        VadDecision(speech_active=True),
                    ]
                ),
                config=SpeechGateConfig(pre_roll_ms=0),
            ),
        )
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events = session.ingest_audio(speech) + session.ingest_audio(speech)
        stable = stable_appends(events)
        jobs = session.consume_stable_timing_jobs_for_events(events)

        assert len(stable) == 1
        assert len(jobs) == 1
        assert jobs[0].source_segment_id == stable[0]["id"]
        assert session.consume_stable_timing_jobs_for_events(events) == []

    def test_initial_silence_does_not_start_asr_or_emit_partial(self) -> None:
        model = FakeStreamingModel(outputs=["静音幻觉"])
        session = RealtimeConnectionSession(
            model,
            config=RealtimeASRConfig(language="Chinese"),
            speech_gate=SpeechGate(
                vad=FakeVadAdapter([VadDecision(speech_active=False)]),
            ),
        )

        events = session.ingest_audio(np.zeros(16_000, dtype=np.float32))

        assert events == []
        assert model.init_count == 0
        assert model.stream_calls == 0

    def test_speech_end_closes_model_epoch_without_closing_transcript(self) -> None:
        model = FakeStreamingModel(outputs=["开头", "后续"], finish_text="开头")
        session = RealtimeConnectionSession(
            model,
            config=RealtimeASRConfig(language="Chinese"),
            speech_gate=SpeechGate(
                vad=FakeVadAdapter(
                    [
                        VadDecision(speech_active=False),
                        VadDecision(speech_started=True, speech_active=True, speech_start_sample=16_000),
                        VadDecision(speech_ended=True, speech_active=False, speech_end_sample=32_000),
                        VadDecision(speech_started=True, speech_active=True, speech_start_sample=48_000),
                    ]
                ),
                config=SpeechGateConfig(pre_roll_ms=0),
            ),
        )

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(np.zeros(16_000, dtype=np.float32)))
        events.extend(session.ingest_audio(np.ones(16_000, dtype=np.float32) * 0.2))
        events.extend(session.ingest_audio(np.zeros(16_000, dtype=np.float32)))

        assert [segment['text'] for segment in stable_appends(events)] == ['开头']
        assert model.finish_calls == 1
        assert model.init_count == 1
        assert session.active_asr is None
        assert_transcript_update_invariants(events)
        stream_calls_after_flush = model.stream_calls
        first_turn_events = events

        resume_events = session.ingest_audio(np.ones(16_000, dtype=np.float32) * 0.2)

        assert session.store.stable_count == 1
        assert model.init_count == 2
        assert model.stream_calls > stream_calls_after_flush
        assert session.active_asr is not None
        assert_transcript_update_invariants(first_turn_events + resume_events)

    def test_short_vad_pause_advances_timeline_without_publishing_silence(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第二秒"], finish_text="第一秒")
        session = RealtimeConnectionSession(
            model,
            config=RealtimeASRConfig(language="Chinese", live_stability_delay_ms=0),
            speech_gate=SpeechGate(
                vad=FakeVadAdapter(
                    [
                        VadDecision(speech_started=True, speech_active=True, speech_start_sample=0),
                        VadDecision(speech_ended=True, speech_active=False, speech_end_sample=16_000),
                        VadDecision(speech_started=True, speech_active=True, speech_start_sample=32_000),
                    ]
                ),
                config=SpeechGateConfig(pre_roll_ms=0),
            ),
        )

        first_events = session.ingest_audio(np.ones(16_000, dtype=np.float32) * 0.2)
        pause_events = session.ingest_audio(np.zeros(16_000, dtype=np.float32))
        resume_events = session.ingest_audio(np.ones(16_000, dtype=np.float32) * 0.2)

        assert stable_appends(first_events) == []
        stable = stable_appends(pause_events)
        updates = transcript_updates(resume_events)
        assert [segment['text'] for segment in stable] == ['第一秒']
        assert stable[0]['start_ms'] == 0
        assert stable[0]['end_ms'] == 1000
        assert updates[-1]['partial']['text'] == '第二秒'  # type: ignore[index]
        assert updates[-1]['partial']['start_ms'] == 2000  # type: ignore[index]
        assert updates[-1]['partial']['end_ms'] == 3000  # type: ignore[index]
        assert model.finish_calls == 1
        assert model.init_count == 2
        assert model.stream_audio_lengths == [16000, 16000]
        assert_transcript_update_invariants(first_events + pause_events + resume_events)

    def test_speech_end_flushes_and_closes_active_epoch(self) -> None:
        model = FakeStreamingModel(outputs=["开头"], finish_text="开头")
        session = RealtimeConnectionSession(
            model,
            config=RealtimeASRConfig(language="Chinese"),
            speech_gate=SpeechGate(
                vad=FakeVadAdapter(
                    [
                        VadDecision(speech_started=True, speech_active=True, speech_start_sample=0),
                        VadDecision(speech_ended=True, speech_active=False, speech_end_sample=16_000),
                    ]
                ),
            ),
        )

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(np.ones(16_000, dtype=np.float32) * 0.2))
        events.extend(session.ingest_audio(np.zeros(16_000, dtype=np.float32)))

        assert [segment['text'] for segment in stable_appends(events)] == ['开头']
        assert model.finish_calls == 1
        assert model.init_count == 1
        assert model.stream_audio_lengths == [16000]
        assert session.active_asr is None
        assert_transcript_update_invariants(events)

    def test_punctuation_does_not_stabilize_while_speech_continues(self) -> None:
        model = FakeStreamingModel(outputs=["第一句。第二", "第一句。第二句。第三", "第一句。第二句。第三句"])
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events1 = session.ingest_audio(speech)
        events2 = session.ingest_audio(speech)
        events3 = session.ingest_audio(speech)

        events = events1 + events2 + events3
        assert stable_appends(events) == []
        assert partial_texts(events)[-1] == '第一句。第二句。第三句'
        assert model.init_count == 1
        assert_transcript_update_invariants(events)

    def test_asr_runs_on_server_side_half_second_cadence(self) -> None:
        model = FakeStreamingModel(outputs=["半秒"], chunk_size_sec=0.5)
        session = make_session(model)
        speech = np.ones(1_600, dtype=np.float32) * 0.2

        for _ in range(4):
            assert session.ingest_audio(speech) == []
            assert model.stream_calls == 0

        events = session.ingest_audio(speech)

        assert partial_texts(events) == ['半秒']
        assert model.stream_calls == 1
        assert model.stream_audio_lengths == [8000]
        assert model.init_kwargs[0]['chunk_size_sec'] == 0.5
        assert model.init_kwargs[0]['unfixed_chunk_num'] == 4
        assert model.init_kwargs[0]['unfixed_token_num'] == 5
        assert model.init_kwargs[0]['max_window_sec'] == 20.0
        assert model.init_kwargs[0]['max_prefix_tokens'] == 64
        assert model.init_kwargs[0]['spec_decode']
        assert model.init_kwargs[0]['language'] == 'Chinese'

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

        assert stable_appends(events) == []
        assert partial_texts(events)[-1] == '第一句话，有补充。下一段，'
        assert_transcript_update_invariants(events)

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
        assert len(stable) == 1
        assert stable[0]['text'] == '前两秒'
        assert stable[0]['start_ms'] == 0
        assert stable[0]['end_ms'] == 2000
        assert non_empty_partials[-1]['text'] == '第三秒'
        assert non_empty_partials[-1]['start_ms'] == 2000
        assert non_empty_partials[-1]['end_ms'] == 3000
        assert model.init_count == 1
        assert model.finish_calls == 0
        assert model.stream_audio_lengths == [16000, 16000, 16000]
        assert_transcript_update_invariants(events)

    def test_live_stability_delay_waits_for_repeated_prefix_before_stabilizing(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "前两秒"])
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        assert stable_appends(events) == []
        assert partial_texts(events)[-1] == '前两秒'
        assert_transcript_update_invariants(events)

    def test_zero_live_stability_delay_still_requires_repeated_prefix(self) -> None:
        model = FakeStreamingModel(outputs=["第一秒", "第一秒第二秒"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        assert [segment['text'] for segment in stable] == ['第一秒']
        assert partial_texts(events)[-1] == '第二秒'
        assert_transcript_update_invariants(events)

    def test_repeated_tail_text_after_stable_prefix_is_not_dropped(self) -> None:
        model = FakeStreamingModel(outputs=["哈哈", "哈哈哈哈", "哈哈哈哈哈"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        assert [segment['text'] for segment in stable] == ['哈哈', '哈哈']
        assert partial_texts(events)[-1] == '哈'
        assert_transcript_update_invariants(events)

    def test_unaligned_live_window_still_updates_partial_without_stabilizing_it(self) -> None:
        model = FakeStreamingModel(outputs=["旧段", "旧段", "新内容", "新内容继续"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        assert [segment['text'] for segment in stable_appends(events)] == ['旧段']
        assert partial_texts(events)[-1] == '新内容继续'
        assert_transcript_update_invariants(events)

    def test_tail_only_window_keeps_updating_current_partial(self) -> None:
        model = FakeStreamingModel(outputs=["旧段", "旧段", "新", "新内容"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        assert [segment['text'] for segment in stable_appends(events)] == ['旧段']
        assert partial_texts(events)[-1] == '新内容'
        assert_transcript_update_invariants(events)

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
        assert final_text == '旧段新内容'
        assert_transcript_update_invariants(events)

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
        assert [segment['text'] for segment in final_events[-1]['segments']] == ['前段', '后段']
        assert_transcript_update_invariants(events)

    def test_window_roll_preserves_text_across_multiple_boundaries(self) -> None:
        # Drive the session through two window rolls (window_start 0 -> 16_000 -> 48_000),
        # simulating the model trimming already-stabilized text out of its bounded window.
        # The published stable transcript must reconstruct the full utterance with no text
        # dropped, duplicated, or reordered at the roll boundaries.
        spoken = "旧段内容新段内容末段内容"
        model = FakeStreamingModel(
            outputs=[
                "旧段内容",
                "旧段内容",
                ("新段内容", 16_000),
                ("新段内容", 16_000),
                ("末段内容", 48_000),
                ("末段内容", 48_000),
            ],
        )
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        for _ in range(6):
            events.extend(session.ingest_audio(speech))

        # While streaming, the stable prefix is append-only and never runs ahead of the
        # spoken utterance — no duplication or reordering at a roll boundary.
        running = ""
        for segment in stable_appends(events):
            running += str(segment["text"])
            assert spoken.startswith(running), f"stable text diverged from utterance: {running!r}"

        events.extend(session.finish())

        final = [event for event in events if event.get("type") == "transcript_final"][-1]
        final_text = "".join(segment["text"] for segment in final["segments"])
        assert final_text == spoken
        assert len(final["segments"]) >= 2  # text genuinely survived across the roll boundaries
        assert_transcript_update_invariants(events)

    def test_stable_prefix_does_not_split_ascii_word(self) -> None:
        model = FakeStreamingModel(outputs=["hello wor", "hello world today"])
        session = make_session(model, live_stability_delay_ms=2_000)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        assert [segment['text'] for segment in stable] == ['hello']
        assert partial_texts(events)[-1] == 'world today'
        assert_transcript_update_invariants(events)

    def test_long_stable_text_is_committed_as_one_transcript_segment(self) -> None:
        stable_text = "一二三四五六七八九十甲乙丙丁戊己庚辛。后续文本"
        model = FakeStreamingModel(outputs=[stable_text, stable_text + "后续"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        assert len(stable) == 1
        assert stable[0]['text'] == stable_text
        assert stable[0]['start_ms'] == 0
        assert stable[-1]['end_ms'] == 1000
        assert partial_texts(events)[-1] == '后续'
        assert_transcript_update_invariants(events)

    def test_long_ascii_stable_text_preserves_spaces(self) -> None:
        stable_text = "hello world today again tomorrow"
        model = FakeStreamingModel(outputs=[stable_text, stable_text + " next"])
        session = make_session(model, live_stability_delay_ms=0)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        events: list[dict[str, object]] = []
        events.extend(session.ingest_audio(speech))
        events.extend(session.ingest_audio(speech))

        stable = stable_appends(events)
        assert len(stable) == 1
        assert stable[0]['text'] == stable_text
        assert partial_texts(events)[-1] == 'next'
        assert_transcript_update_invariants(events)

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
        assert [segment['text'] for segment in stable] == ['前两秒', '第三秒']
        assert [(segment['start_ms'], segment['end_ms']) for segment in stable] == [(0, 2000), (2000, 3000)]
        assert model.init_count == 1
        assert model.finish_calls == 1
        assert_transcript_update_invariants(events)

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
        assert [segment['text'] for segment in stable] == ['第一秒', '第二秒']
        assert partial_texts(events)[-1] == '第三秒'
        assert_transcript_update_invariants(events)

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
        assert [segment['text'] for segment in stable_appends(events)] == ['第一秒', '第二秒', '第三秒']
        assert updates[-1]['stable_appends'][0]['text'] == '第三秒'
        assert updates[-1]['partial'] is None
        assert final_events[-1]['stable_count'] == 3
        assert [segment['text'] for segment in final_events[-1]['segments']] == ['第一秒', '第二秒', '第三秒']
        assert_transcript_update_invariants(events)

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
        assert [segment['text'] for segment in stable_appends(events)] == ['第一秒', '第二秒', '第三秒尾巴']
        assert [segment['text'] for segment in final_events[-1]['segments']] == ['第一秒', '第二秒', '第三秒尾巴']
        assert_transcript_update_invariants(events)

    def test_flush_stabilizes_tail_without_resetting_streaming_state(self) -> None:
        model = FakeStreamingModel(outputs=["尾句", "尾句后续"], finish_text="尾句")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        flush_events = session.flush()
        resume_events = session.ingest_audio(speech)

        assert [segment['text'] for segment in stable_appends(flush_events)] == ['尾句']
        assert partial_texts(resume_events) == ['后续']
        assert model.finish_calls == 1
        assert model.init_count == 1

    def test_set_language_flushes_tail_and_restarts_future_asr_state(self) -> None:
        model = FakeStreamingModel(outputs=["hello", "world"], finish_text="hello")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        switch_events = session.set_language("English")
        resume_events = session.ingest_audio(speech)

        stable = stable_appends(switch_events)
        assert [segment['text'] for segment in stable] == ['hello']
        assert stable[0]['language'] == 'Chinese'
        assert session.config.language == 'English'
        assert partial_texts(resume_events) == ['world']
        assert model.finish_calls == 1
        assert model.init_count == 2
        assert model.init_kwargs[0]['language'] == 'Chinese'
        assert model.init_kwargs[1]['language'] == 'English'
        assert_transcript_update_invariants(switch_events + resume_events)

    def test_set_language_none_returns_future_asr_to_auto_language(self) -> None:
        model = FakeStreamingModel(outputs=["hello", "world"], finish_text="hello")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        session.set_language(None)
        session.ingest_audio(speech)

        assert session.config.language is None
        assert model.init_kwargs[1]['language'] is None

    def test_forced_flush_stabilizes_one_speech_segment_without_punctuation_split(self) -> None:
        model = FakeStreamingModel(outputs=[""], finish_text="第一句。第二句。尾巴")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        events = session.flush()

        assert [segment['text'] for segment in stable_appends(events)] == ['第一句。第二句。尾巴']

    def test_finish_feeds_received_tail_even_below_asr_cadence(self) -> None:
        model = FakeStreamingModel(outputs=["前段", "前段后段"], finish_text="前段后段")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2
        quiet_tail = np.ones(3_840, dtype=np.float32) * 0.005

        session.ingest_audio(speech)
        session.ingest_audio(quiet_tail)
        events = session.finish()

        assert model.stream_audio_lengths == [16000, 3840]
        assert [segment['text'] for segment in stable_appends(events)] == ['前段后段']

    def test_low_energy_audio_between_speech_is_not_dropped(self) -> None:
        model = FakeStreamingModel(outputs=["前半", "前半低能量后半"], finish_text="前半低能量后半")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2
        low_energy = np.zeros(16_000, dtype=np.float32)

        session.ingest_audio(speech)
        events = session.ingest_audio(low_energy)

        assert partial_texts(events) == ['前半低能量后半']
        assert model.stream_audio_lengths == [16000, 16000]

    def test_short_pause_is_promoted_when_speech_resumes(self) -> None:
        model = FakeStreamingModel(outputs=["前半", "前半后半"])
        session = make_session(model)
        speech_one = np.ones(16_000, dtype=np.float32) * 0.2
        short_pause = np.zeros(3_200, dtype=np.float32)
        speech_two = np.ones(12_800, dtype=np.float32) * 0.2

        session.ingest_audio(speech_one)
        pause_events = session.ingest_audio(short_pause)
        resume_events = session.ingest_audio(speech_two)

        assert partial_texts(pause_events + resume_events) == ['前半后半']
        assert model.stream_audio_lengths == [16000, 16000]

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

        assert [segment['text'] for segment in stable_appends(events + flush_events)] == ['第一段第二段尾段']
        assert model.init_count == 1
        assert model.stream_audio_lengths == [16000, 16000, 8000]

    def test_finish_emits_transcript_final_snapshot(self) -> None:
        model = FakeStreamingModel(outputs=["尾句"], finish_text="尾句")
        session = make_session(model)
        speech = np.ones(16_000, dtype=np.float32) * 0.2

        session.ingest_audio(speech)
        events = session.finish()

        final_events = [event for event in events if event["type"] == "transcript_final"]
        assert len(final_events) == 1
        assert final_events[0]['stable_count'] == 1
        assert final_events[0]['segments'][0]['text'] == '尾句'
        assert 'final' not in {event['type'] for event in events}
        assert not {'partial', 'committed'} & {str(event['type']) for event in events}
        assert_transcript_update_invariants(events)


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


class TestSileroVadAdapter:
    def test_default_config_uses_onnx_runtime(self) -> None:
        assert SileroVadConfig().use_onnx

    def test_config_rejects_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            SileroVadConfig(threshold=0.0)
        with pytest.raises(ValueError):
            SileroVadConfig(threshold=1.5)

    def test_config_rejects_negative_threshold_not_below_threshold(self) -> None:
        # negative_threshold must stay within (0, threshold) to preserve hysteresis.
        with pytest.raises(ValueError):
            SileroVadConfig(threshold=0.5, negative_threshold=0.5)
        with pytest.raises(ValueError):
            SileroVadConfig(threshold=0.5, negative_threshold=0.0)
        # A valid hysteresis gap is accepted.
        assert SileroVadConfig(threshold=0.5, negative_threshold=0.3).negative_threshold == 0.3

    def test_buffers_until_silero_chunk_is_complete(self) -> None:
        model = FakeVadModel([0.8])
        vad = SileroVadAdapter(
            SileroVadConfig(threshold=0.5, min_speech_ms=32, min_silence_ms=64),
            model=model,
        )

        first = vad.accept(np.ones(256, dtype=np.float32))
        second = vad.accept(np.ones(256, dtype=np.float32))

        assert not first.has_speech
        assert model.calls == 1
        assert second.speech_started

    def test_requires_min_speech_and_min_silence(self) -> None:
        model = FakeVadModel([0.8, 0.8, 0.1, 0.1])
        vad = SileroVadAdapter(
            SileroVadConfig(threshold=0.5, min_speech_ms=64, min_silence_ms=64),
            model=model,
        )

        decision = vad.accept(np.ones(512 * 4, dtype=np.float32))

        assert decision.speech_started
        assert decision.speech_ended
        assert not decision.speech_active
