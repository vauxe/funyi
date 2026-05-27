# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .streaming import RecognitionFrame, RecognitionTail, TailSelector, TextStabilizer
from .realtime_timestamps import StableTimingJob
from .transcript_store import PartialSegment, StableSegment, TranscriptStore
from .utils import SAMPLE_RATE
from .vad import normalize_pcm


def _empty_audio() -> np.ndarray:
    return np.zeros((0,), dtype=np.float32)


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


@dataclass
class _TranscriptCursor:
    sample: int = 0
    stable_text_prefix: str = ""
    stabilizer: TextStabilizer = field(default_factory=TextStabilizer)


class RealtimeASRSession:
    """Single-user realtime ASR session.

    The session has one lossless audio path: every accepted PCM sample is
    eventually fed to one continuous streaming ASR state.
    """

    def __init__(
        self,
        model: Any,
        *,
        transcript_store: TranscriptStore | None = None,
        config: RealtimeASRConfig | None = None,
    ) -> None:
        self.model = model
        self.store = transcript_store or TranscriptStore()
        self.config = config or RealtimeASRConfig()

        self._asr_state: Any = None
        self._transcript = _TranscriptCursor()
        self._samples_received = 0
        self._asr_audio = _SampleBuffer()
        self._streaming_kwargs = self._low_latency_streaming_kwargs()
        self._asr_cadence_samples = self._streaming_chunk_samples(self._streaming_kwargs)
        self._live_stability_delay_samples = max(
            0,
            int(round(self.config.sample_rate * self.config.live_stability_delay_ms / 1000)),
        )
        self._timing_hints: dict[str, tuple[int, int]] = {}

        self._last_asr_end_sample = 0

    def ingest_audio(self, pcm16k: np.ndarray) -> list[dict[str, Any]]:
        audio = normalize_pcm(pcm16k)
        if audio.shape[0] == 0:
            return []
        self._append_asr_audio(audio)
        return self._drain_asr_audio(force=False, emit_live=True)

    def flush(self) -> list[dict[str, Any]]:
        events = self._drain_asr_audio(force=True, emit_live=False)
        if self._asr_state is None:
            if self.store.clear_partial():
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

    def stable_timing_jobs(self, event: dict[str, Any]) -> list[StableTimingJob]:
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

    def stable_timing_jobs_for_events(self, events: list[dict[str, Any]]) -> list[StableTimingJob]:
        jobs: list[StableTimingJob] = []
        for event in events:
            jobs.extend(self.stable_timing_jobs(event))
        return jobs

    def _append_asr_audio(self, audio: np.ndarray) -> None:
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
        self._asr_state = self.model.streaming_transcribe(audio, state)
        self._last_asr_end_sample = int(chunk_end_sample)
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

        start_ms = self._sample_to_ms(start_sample)
        end_ms = max(start_ms, self._sample_to_ms(end_sample))
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
        self.store.replace_partial(self._partial_segment(partial))
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
        start_sample = int(round(int(partial.start_ms) * int(self.config.sample_rate) / 1000))
        end_sample = int(round(int(partial.end_ms) * int(self.config.sample_rate) / 1000))
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
        self.store.replace_partial(None)
        return self._emit_transcript_update(stable_base=stable_base, stable_appends=[segment])

    def _replace_partial_text(self, text: str) -> list[dict[str, Any]]:
        return self._replace_partial(self._partial_segment(text))

    def _partial_segment(self, text: str) -> PartialSegment | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None

        language = self._current_asr_language()
        start_ms = self._sample_to_ms(self._transcript.sample)
        end_ms = max(start_ms, self._sample_to_ms(self._last_asr_end_sample))
        return PartialSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            text=normalized,
            language=language,
        )

    def _replace_partial(self, partial: PartialSegment | None) -> list[dict[str, Any]]:
        changed = self.store.replace_partial(partial)
        if not changed:
            return []
        return self._emit_transcript_update(stable_base=self.store.stable_count, stable_appends=[])

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
        self._timing_hints[segment.id] = (int(start_sample), max(int(start_sample), int(end_sample)))

    def _drain_asr_audio(self, *, force: bool, emit_live: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self._asr_audio.samples >= self._asr_cadence_samples or (force and self._asr_audio.samples > 0):
            chunk_samples = min(self._asr_cadence_samples, self._asr_audio.samples)
            chunk, start_sample = self._asr_audio.pop(chunk_samples)
            chunk_end_sample = int(start_sample) + chunk_samples
            events.extend(self._run_asr(chunk, chunk_end_sample, emit_live=emit_live))
        return events

    def _sample_to_ms(self, sample_index: int) -> int:
        return int(round(1000 * int(sample_index) / int(self.config.sample_rate)))

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


__all__ = [
    "RealtimeASRConfig",
    "RealtimeASRSession",
]
