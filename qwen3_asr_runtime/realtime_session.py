# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Optional

import numpy as np

from .audio_utils import normalize_pcm
from .streaming import RecognitionFrame, RecognitionTail, TailSelector, TextStabilizer
from .realtime_timestamps import StableTimingJob
from .speech_gate import SpeechGate, SpeechGateEvent
from .transcript_store import PartialSegment, StableSegment, TranscriptStore
from .utils import SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)


def _empty_audio() -> np.ndarray:
    return np.zeros((0,), dtype=np.float32)


def _truncate_log_text(text: Any, *, limit: int = 80) -> str:
    value = str(text or "").replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


_LOW_LATENCY_STREAMING_KWARGS: dict[str, Any] = {
    "chunk_size_sec": 0.5,
    "unfixed_chunk_num": 4,
    "unfixed_token_num": 5,
    "max_window_sec": 20.0,
    "max_prefix_tokens": 64,
    "spec_decode": True,
}


@dataclass
class RealtimeASRConfig:
    sample_rate: int = SAMPLE_RATE
    context: str = ""
    language: Optional[str] = None
    live_stability_delay_ms: int = 12_000
    force_align_timestamps: bool = False


@dataclass
class _SampleBuffer:
    start_sample: int = 0
    audio: np.ndarray = field(default_factory=_empty_audio)

    @property
    def samples(self) -> int:
        return int(self.audio.shape[0])

    def append(self, audio: np.ndarray) -> None:
        if audio.shape[0] == 0:
            return
        chunk = audio.astype(np.float32, copy=True)
        if self.samples == 0:
            self.audio = chunk
            return
        self.audio = np.concatenate([self.audio, chunk], axis=0)

    def pop(self, samples: int) -> tuple[np.ndarray, int]:
        samples = min(max(0, int(samples)), self.samples)
        if samples == 0:
            return _empty_audio(), int(self.start_sample)

        start_sample = int(self.start_sample)
        chunk = self.audio[:samples].copy()
        self.audio = self.audio[samples:].copy()
        self.start_sample = start_sample + samples
        return chunk, start_sample


@dataclass(frozen=True)
class _TimelineSpan:
    local_start: int
    local_end: int
    source_start: int
    source_end: int


class _SourceTimeline:
    """Map ASR-local speech samples back to the connection source clock."""

    def __init__(self, origin_sample: int = 0) -> None:
        self._origin_sample = int(origin_sample)
        self._spans: list[_TimelineSpan] = []

    def append(self, samples: int, *, source_start_sample: int | None = None) -> None:
        sample_count = max(0, int(samples))
        if sample_count == 0:
            return

        local_start = self._spans[-1].local_end if self._spans else 0
        local_end = local_start + sample_count
        if source_start_sample is None:
            source_start = self._spans[-1].source_end if self._spans else int(self._origin_sample)
        else:
            source_start = int(source_start_sample)
        if self._spans and source_start < self._spans[-1].source_end:
            raise ValueError("source_start_sample must not overlap previous source audio")
        source_end = source_start + sample_count
        if self._spans:
            previous = self._spans[-1]
            if previous.local_end == local_start and previous.source_end == source_start:
                self._spans[-1] = _TimelineSpan(
                    local_start=previous.local_start,
                    local_end=local_end,
                    source_start=previous.source_start,
                    source_end=source_end,
                )
                return
        self._spans.append(
            _TimelineSpan(
                local_start=local_start,
                local_end=local_end,
                source_start=source_start,
                source_end=source_end,
            )
        )

    def source_start_sample(self, local_sample: int) -> int:
        sample = int(local_sample)
        if not self._spans:
            return int(self._origin_sample)
        return self._source_sample_start(sample)

    def source_end_sample(self, local_sample: int) -> int:
        sample = int(local_sample)
        if not self._spans:
            return int(self._origin_sample)
        return self._source_sample_end(sample)

    def _source_sample_start(self, sample: int) -> int:
        for span in self._spans:
            if sample < span.local_start:
                return span.source_start
            if span.local_start <= sample < span.local_end:
                return span.source_start + (sample - span.local_start)
        last = self._spans[-1]
        return last.source_end + max(0, sample - last.local_end)

    def _source_sample_end(self, sample: int) -> int:
        previous: _TimelineSpan | None = None
        for span in self._spans:
            if sample < span.local_start:
                return span.source_start
            if sample == span.local_start:
                return previous.source_end if previous is not None else span.source_start
            if span.local_start < sample <= span.local_end:
                return span.source_start + (sample - span.local_start)
            previous = span
        last = self._spans[-1]
        return last.source_end + max(0, sample - last.local_end)


@dataclass
class _TranscriptCursor:
    sample: int = 0
    stable_text_prefix: str = ""
    stabilizer: TextStabilizer = field(default_factory=TextStabilizer)


class RealtimeASRSession:
    """Realtime ASR state for one contiguous audio epoch."""

    def __init__(
        self,
        model: Any,
        *,
        transcript_store: TranscriptStore | None = None,
        config: RealtimeASRConfig | None = None,
        time_origin_sample: int = 0,
    ) -> None:
        self.model = model
        self.store = transcript_store or TranscriptStore()
        self.config = config or RealtimeASRConfig()
        self._time_origin_sample = int(time_origin_sample)

        self._asr_state: Any = None
        self._transcript = _TranscriptCursor()
        self._source_timeline = _SourceTimeline(time_origin_sample)
        self._samples_received = 0
        self._asr_audio = _SampleBuffer()
        self._streaming_kwargs = self._low_latency_streaming_kwargs()
        self._asr_cadence_samples = self._streaming_chunk_samples(self._streaming_kwargs)
        self._live_stability_delay_samples = max(
            0,
            int(round(self.config.sample_rate * self.config.live_stability_delay_ms / 1000)),
        )
        self._timing_hints: dict[str, tuple[int, int]] = {}
        self._partial_sample_range: tuple[int, int] | None = None

        self._last_asr_end_sample = 0

    def ingest_audio(
        self,
        pcm16k: np.ndarray,
        *,
        source_start_sample: int | None = None,
    ) -> list[dict[str, Any]]:
        audio = normalize_pcm(pcm16k)
        if audio.shape[0] == 0:
            return []
        self._append_asr_audio(audio, source_start_sample=source_start_sample)
        return self._drain_asr_audio(force=False, emit_live=True)

    def flush(self) -> list[dict[str, Any]]:
        events = self._drain_asr_audio(force=True, emit_live=False)
        if self._asr_state is None:
            if self._set_partial(None, sample_range=None):
                events.extend(self._emit_transcript_update(stable_base=self.store.stable_count, stable_appends=[]))
            return events

        self._asr_state = self.model.finish_streaming_transcribe(self._asr_state)
        self._last_asr_end_sample = max(self._last_asr_end_sample, self._samples_received)
        events.extend(self._handle_decoded_text(finalize=True))
        return events

    def set_language(self, language: Optional[str]) -> list[dict[str, Any]]:
        events = self.flush()
        self.config.language = (str(language).strip() or None) if language is not None else None
        cursor_sample = max(int(self._transcript.sample), int(self._last_asr_end_sample))
        self._transcript = _TranscriptCursor(sample=cursor_sample)
        self._asr_state = None
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self.flush()
        events.append(self.store.final_event())
        return events

    def consume_stable_timing_jobs(self, event: dict[str, Any]) -> list[StableTimingJob]:
        if not self.config.force_align_timestamps or event.get("type") != "transcript_update":
            return []

        jobs: list[StableTimingJob] = []
        for segment in event.get("stable_appends") or []:
            if not isinstance(segment, dict):
                continue
            segment_id = str(segment.get("id") or "")
            hint = self._timing_hints.get(segment_id)
            source_text = str(segment.get("text") or "").strip()
            if hint is None or not segment_id or not source_text:
                continue
            self._timing_hints.pop(segment_id, None)
            jobs.append(
                StableTimingJob(
                    source_segment_id=segment_id,
                    source_text=source_text,
                    source_language=str(segment.get("language") or self.config.language or ""),
                    start_sample=int(hint[0]),
                    end_sample=int(hint[1]),
                )
            )
        return jobs

    def consume_stable_timing_jobs_for_events(self, events: list[dict[str, Any]]) -> list[StableTimingJob]:
        jobs: list[StableTimingJob] = []
        for event in events:
            jobs.extend(self.consume_stable_timing_jobs(event))
        return jobs

    def _append_asr_audio(self, audio: np.ndarray, *, source_start_sample: int | None = None) -> None:
        self._source_timeline.append(int(audio.shape[0]), source_start_sample=source_start_sample)
        self._samples_received += int(audio.shape[0])
        self._asr_audio.append(audio)

    def _run_asr(
        self,
        audio: np.ndarray,
        chunk_end_sample: int,
        *,
        emit_live: bool,
    ) -> list[dict[str, Any]]:
        state = self._ensure_asr_state()
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "ASR chunk samples=%d start_sample=%d end_sample=%d emit_live=%s language=%s",
                int(audio.shape[0]),
                int(chunk_end_sample) - int(audio.shape[0]),
                int(chunk_end_sample),
                bool(emit_live),
                self.config.language or "auto",
            )
        self._asr_state = self.model.streaming_transcribe(audio, state)
        self._last_asr_end_sample = int(chunk_end_sample)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            frame = self._current_recognition_frame()
            if frame is not None:
                _LOGGER.debug(
                    "ASR frame window_start=%d audio_end=%d language=%s decoded=%r generated=%r full=%r",
                    int(frame.window_start_sample),
                    int(frame.audio_end_sample),
                    frame.language,
                    _truncate_log_text(frame.decoded_text),
                    _truncate_log_text(frame.generated_text),
                    _truncate_log_text(frame.full_text),
                )
        if not emit_live:
            return []
        return self._handle_decoded_text(finalize=False)

    def _ensure_asr_state(self) -> Any:
        if self._asr_state is not None:
            return self._asr_state

        kwargs = dict(self._streaming_kwargs)
        kwargs.setdefault("context", self.config.context)
        kwargs.setdefault("language", self.config.language)
        self._asr_state = self.model.init_streaming_state(**kwargs)
        return self._asr_state

    def _handle_decoded_text(self, *, finalize: bool) -> list[dict[str, Any]]:
        if self._asr_state is None:
            return []

        tail = self._current_tail()
        tail_text = tail.text
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "ASR tail finalize=%s aligned=%s text=%r",
                bool(finalize),
                bool(tail.aligned),
                _truncate_log_text(tail_text),
            )
        if not tail.aligned:
            if finalize:
                return self._append_final_unaligned_tail(tail_text)
            current_partial = ""
            if self.store.partial is not None:
                current_partial = str(getattr(self.store.partial, "text", "") or "")
            if current_partial and not TextStabilizer.is_tail_update(current_partial, tail_text):
                return []
            return self._replace_partial_text(tail_text)

        if finalize:
            update = self._transcript.stabilizer.finalize(tail_text, end_sample=self._last_asr_end_sample)
            return self._append_stable_text(
                stable_text=update.stable_text,
                partial_text=update.partial_text,
                end_sample=update.stable_end_sample or self._last_asr_end_sample,
            )

        update = self._transcript.stabilizer.observe(
            tail_text,
            end_sample=self._last_asr_end_sample,
            can_commit=self._live_stability_delay_elapsed(),
        )
        if update.stable_text and update.stable_end_sample is not None:
            return self._append_stable_text(
                stable_text=update.stable_text,
                partial_text=update.partial_text,
                end_sample=update.stable_end_sample,
            )

        return self._replace_partial_text(update.partial_text)

    def _current_tail(self) -> RecognitionTail:
        frame = self._current_recognition_frame()
        if frame is None:
            raise RuntimeError("streaming ASR state did not produce a RecognitionFrame")
        previous_partial = ""
        if self.store.partial is not None:
            previous_partial = str(self.store.partial.text or "")
        return TailSelector.select(
            frame,
            stable_text_prefix=self._transcript.stable_text_prefix,
            stable_end_sample=self._transcript.sample,
            previous_partial_text=previous_partial,
        )

    def _current_recognition_frame(self) -> RecognitionFrame | None:
        frame = getattr(self._asr_state, "recognition_frame", None)
        if isinstance(frame, RecognitionFrame):
            return frame
        return None

    def _current_asr_language(self) -> str:
        frame = self._current_recognition_frame()
        if frame is not None:
            language = str(frame.language or "").strip()
            if language:
                return language
        return str(getattr(self._asr_state, "language", "") or self.config.language or "")

    def _live_stability_delay_elapsed(self) -> bool:
        return self._last_asr_end_sample - self._transcript.sample >= self._live_stability_delay_samples

    def _append_stable_text(
        self,
        *,
        stable_text: str,
        partial_text: str,
        end_sample: int,
    ) -> list[dict[str, Any]]:
        normalized = TextStabilizer.clean_tail_text(stable_text)
        partial = TextStabilizer.clean_tail_text(partial_text)
        if not normalized:
            self._transcript.stabilizer.set_tail(partial, end_sample=self._last_asr_end_sample if partial else None)
            return self._replace_partial_text(partial)

        stable_base = self.store.stable_count
        end_sample = max(int(self._transcript.sample), int(end_sample))
        start_sample = int(self._transcript.sample)
        language = self._current_asr_language()

        start_ms = self._start_ms(start_sample)
        end_ms = max(start_ms, self._end_ms(end_sample))
        segment = self.store.append_stable_segment(
            text=normalized,
            start_ms=None if self.config.force_align_timestamps else start_ms,
            end_ms=None if self.config.force_align_timestamps else end_ms,
            language=language,
            timing_status="pending" if self.config.force_align_timestamps else None,
        )
        self._remember_timing_hint(segment, start_sample=start_sample, end_sample=end_sample)
        self._transcript.sample = end_sample
        self._transcript.stable_text_prefix += normalized
        self._transcript.stabilizer.set_tail(partial, end_sample=self._last_asr_end_sample if partial else None)
        partial_segment, partial_range = self._partial_segment(partial)
        self._set_partial(partial_segment, sample_range=partial_range)
        return self._emit_transcript_update(stable_base=stable_base, stable_appends=[segment])

    def _append_final_unaligned_tail(self, tail_text: str) -> list[dict[str, Any]]:
        partial = self.store.partial
        if partial is None:
            return []

        current = str(partial.text or "")
        tail = TextStabilizer.clean_tail_text(tail_text)
        if tail and TextStabilizer.is_tail_update(current, tail) and len(tail) >= len(current.strip()):
            return self._append_stable_text(
                stable_text=tail,
                partial_text="",
                end_sample=self._last_asr_end_sample,
            )

        return self._append_existing_partial()

    def _append_existing_partial(self) -> list[dict[str, Any]]:
        partial = self.store.partial
        if partial is None or not str(partial.text or "").strip():
            return []

        stable_base = self.store.stable_count
        start_sample, end_sample = self._partial_samples(partial)
        segment = self.store.append_stable_segment(
            text=partial.text,
            start_ms=None if self.config.force_align_timestamps else partial.start_ms,
            end_ms=None if self.config.force_align_timestamps else partial.end_ms,
            language=partial.language,
            timing_status="pending" if self.config.force_align_timestamps else None,
        )
        self._remember_timing_hint(segment, start_sample=start_sample, end_sample=end_sample)
        self._transcript.sample = max(
            int(self._transcript.sample),
            end_sample,
        )
        self._transcript.stable_text_prefix += TextStabilizer.clean_tail_text(partial.text)
        self._transcript.stabilizer.set_tail("", end_sample=None)
        self._set_partial(None, sample_range=None)
        return self._emit_transcript_update(stable_base=stable_base, stable_appends=[segment])

    def _replace_partial_text(self, text: str) -> list[dict[str, Any]]:
        partial, sample_range = self._partial_segment(text)
        return self._replace_partial(partial, sample_range=sample_range)

    def _partial_segment(self, text: str) -> tuple[PartialSegment | None, tuple[int, int] | None]:
        normalized = str(text or "").strip()
        if not normalized:
            return None, None

        language = self._current_asr_language()
        start_sample = int(self._transcript.sample)
        end_sample = max(start_sample, int(self._last_asr_end_sample))
        start_ms = self._start_ms(start_sample)
        end_ms = max(start_ms, self._end_ms(end_sample))
        return (
            PartialSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=normalized,
                language=language,
            ),
            (start_sample, end_sample),
        )

    def _set_partial(
        self,
        partial: PartialSegment | None,
        *,
        sample_range: tuple[int, int] | None,
    ) -> bool:
        changed = self.store.replace_partial(partial)
        self._partial_sample_range = sample_range if partial is not None else None
        return changed

    def _replace_partial(
        self,
        partial: PartialSegment | None,
        *,
        sample_range: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        changed = self._set_partial(partial, sample_range=sample_range)
        if not changed:
            return []
        return self._emit_transcript_update(stable_base=self.store.stable_count, stable_appends=[])

    def _partial_samples(self, partial: PartialSegment) -> tuple[int, int]:
        if self._partial_sample_range is not None:
            return self._partial_sample_range
        start_sample = self._source_ms_to_local_sample(partial.start_ms)
        end_sample = max(start_sample, self._source_ms_to_local_sample(partial.end_ms))
        return start_sample, end_sample

    def _emit_transcript_update(
        self,
        *,
        stable_base: int,
        stable_appends: list[StableSegment],
    ) -> list[dict[str, Any]]:
        event = self.store.update_event(stable_base=stable_base, stable_appends=stable_appends)
        return [event]

    def _remember_timing_hint(self, segment: StableSegment, *, start_sample: int, end_sample: int) -> None:
        if not self.config.force_align_timestamps:
            return
        source_start = self._source_timeline.source_start_sample(start_sample)
        source_end = self._source_timeline.source_end_sample(end_sample)
        self._timing_hints[segment.id] = (
            source_start,
            max(source_start, source_end),
        )

    def _drain_asr_audio(self, *, force: bool, emit_live: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self._asr_audio.samples >= self._asr_cadence_samples or (force and self._asr_audio.samples > 0):
            chunk_samples = min(self._asr_cadence_samples, self._asr_audio.samples)
            chunk, start_sample = self._asr_audio.pop(chunk_samples)
            chunk_end_sample = int(start_sample) + chunk_samples
            events.extend(self._run_asr(chunk, chunk_end_sample, emit_live=emit_live))
        return events

    def _start_ms(self, sample_index: int) -> int:
        return self._source_sample_to_ms(self._source_timeline.source_start_sample(sample_index))

    def _end_ms(self, sample_index: int) -> int:
        return self._source_sample_to_ms(self._source_timeline.source_end_sample(sample_index))

    def _source_sample_to_ms(self, source_sample: int) -> int:
        return int(round(1000 * int(source_sample) / int(self.config.sample_rate)))

    def _source_ms_to_local_sample(self, source_ms: int) -> int:
        source_sample = int(round(int(source_ms) * int(self.config.sample_rate) / 1000))
        return max(0, source_sample - int(self._time_origin_sample))

    def _low_latency_streaming_kwargs(self) -> dict[str, Any]:
        kwargs = dict(_LOW_LATENCY_STREAMING_KWARGS)
        if hasattr(self.model, "low_latency_preset_kwargs"):
            kwargs.update(self.model.low_latency_preset_kwargs())
        return kwargs

    def _streaming_chunk_samples(self, kwargs: dict[str, Any]) -> int:
        chunk_size_sec = float(kwargs.get("chunk_size_sec", _LOW_LATENCY_STREAMING_KWARGS["chunk_size_sec"]))
        if chunk_size_sec <= 0:
            raise ValueError(f"low-latency chunk_size_sec must be > 0, got: {chunk_size_sec}")
        return max(1, int(round(self.config.sample_rate * chunk_size_sec)))


class RealtimeConnectionSession:
    """Long-lived realtime connection that owns VAD epochs and transcript history."""

    def __init__(
        self,
        model: Any,
        *,
        transcript_store: TranscriptStore | None = None,
        config: RealtimeASRConfig | None = None,
        speech_gate: SpeechGate | None = None,
    ) -> None:
        self.model = model
        self.store = transcript_store or TranscriptStore()
        self.config = config or RealtimeASRConfig()
        self.speech_gate = speech_gate or SpeechGate()
        self._active_asr: RealtimeASRSession | None = None
        self._active_asr_end_sample: int | None = None
        self._active_asr_flushed = False
        self._timing_jobs: dict[str, StableTimingJob] = {}

    @property
    def active_asr(self) -> RealtimeASRSession | None:
        return self._active_asr

    def ingest_audio(self, pcm16k: np.ndarray) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        speech_events = self.speech_gate.accept(pcm16k)
        if speech_events and _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "VAD events samples=%d active=%s events=%s",
                int(np.asarray(pcm16k).reshape(-1).shape[0]),
                bool(self.speech_gate.speech_active),
                [
                    {
                        "type": event.type,
                        "start_sample": event.start_sample,
                        "end_sample": event.end_sample,
                        "samples": int(event.audio.shape[0]),
                    }
                    for event in speech_events
                ],
            )
        for speech_event in speech_events:
            if speech_event.type in ("speech_start", "speech_audio"):
                events.extend(self._accept_speech_audio(speech_event))
            elif speech_event.type == "speech_end":
                events.extend(self._flush_active_asr(close=True))
        return events

    def flush(self) -> list[dict[str, Any]]:
        return self._flush_active_asr(close=False)

    def set_language(self, language: Optional[str]) -> list[dict[str, Any]]:
        events = self._flush_active_asr(close=True)
        self.config.language = (str(language).strip() or None) if language is not None else None
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self._flush_active_asr(close=True)
        events.append(self.store.final_event())
        return events

    def consume_stable_timing_jobs(self, event: dict[str, Any]) -> list[StableTimingJob]:
        if event.get("type") != "transcript_update":
            return []
        jobs: list[StableTimingJob] = []
        for segment in event.get("stable_appends") or []:
            if not isinstance(segment, dict):
                continue
            job = self._timing_jobs.get(str(segment.get("id") or ""))
            if job is not None:
                self._timing_jobs.pop(job.source_segment_id, None)
                jobs.append(job)
        return jobs

    def consume_stable_timing_jobs_for_events(self, events: list[dict[str, Any]]) -> list[StableTimingJob]:
        jobs: list[StableTimingJob] = []
        for event in events:
            jobs.extend(self.consume_stable_timing_jobs(event))
        return jobs

    def _new_asr_epoch(self, origin_sample: int) -> RealtimeASRSession:
        return RealtimeASRSession(
            self.model,
            transcript_store=self.store,
            config=self.config,
            time_origin_sample=int(origin_sample),
        )

    def _ensure_active_asr(self, origin_sample: int) -> None:
        if self._active_asr is None:
            self._active_asr = self._new_asr_epoch(origin_sample)
            self._active_asr_end_sample = int(origin_sample)
            self._active_asr_flushed = False

    def _accept_speech_audio(self, speech_event: SpeechGateEvent) -> list[dict[str, Any]]:
        self._ensure_active_asr(speech_event.start_sample)
        return self._ingest_speech_audio(speech_event)

    def _ingest_speech_audio(self, speech_event: SpeechGateEvent) -> list[dict[str, Any]]:
        if self._active_asr is None or speech_event.audio.shape[0] == 0:
            return []
        audio = speech_event.audio
        start_sample = int(speech_event.start_sample)
        represented_end = (
            start_sample if self._active_asr_end_sample is None else int(self._active_asr_end_sample)
        )

        if start_sample < represented_end:
            drop = min(represented_end - start_sample, int(audio.shape[0]))
            audio = audio[drop:]
            start_sample += drop
            if audio.shape[0] == 0:
                return []

        if start_sample > represented_end and _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "ASR source gap skipped start_sample=%d represented_end=%d gap_samples=%d",
                start_sample,
                represented_end,
                start_sample - represented_end,
            )

        self._active_asr_flushed = False
        events = self._active_asr.ingest_audio(audio, source_start_sample=start_sample)
        self._active_asr_end_sample = start_sample + int(audio.shape[0])
        self._remember_timing_jobs(self._active_asr, events)
        return events

    def _flush_active_asr(self, *, close: bool) -> list[dict[str, Any]]:
        if self._active_asr is None:
            return []
        session = self._active_asr
        events: list[dict[str, Any]] = []
        if not self._active_asr_flushed:
            events = session.flush()
            self._remember_timing_jobs(session, events)
            self._active_asr_flushed = True
        if close:
            self._drop_active_asr()
        return events

    def _drop_active_asr(self) -> None:
        self._active_asr = None
        self._active_asr_end_sample = None
        self._active_asr_flushed = False

    def _remember_timing_jobs(self, turn: RealtimeASRSession, events: list[dict[str, Any]]) -> None:
        for job in turn.consume_stable_timing_jobs_for_events(events):
            self._timing_jobs[job.source_segment_id] = job


__all__ = [
    "RealtimeASRConfig",
    "RealtimeASRSession",
    "RealtimeConnectionSession",
]
