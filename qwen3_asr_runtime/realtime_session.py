# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .transcript_store import PartialSegment, StableSegment, TranscriptStore
from .utils import SAMPLE_RATE
from .vad import (
    EnergyVadAdapter,
    EnergyVadConfig,
    SileroVadAdapter,
    SileroVadConfig,
    VadConfig,
    VadDecision,
    create_vad_adapter,
    normalize_pcm,
)


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
    pre_roll_ms: int = 240
    input_chunk_ms: int = 100
    live_stability_delay_ms: int = 12_000
    vad: VadConfig = field(default_factory=SileroVadConfig)


@dataclass
class _SampleBuffer:
    start_sample: Optional[int] = None
    audio: np.ndarray = field(default_factory=_empty_audio)

    @property
    def samples(self) -> int:
        return int(self.audio.shape[0])

    def clear(self) -> None:
        self.start_sample = None
        self.audio = _empty_audio()

    def append(self, audio: np.ndarray, *, start_sample: int) -> None:
        if audio.shape[0] == 0:
            return
        chunk = audio.astype(np.float32, copy=True)
        if self.samples == 0 or self.start_sample is None:
            self.start_sample = int(start_sample)
            self.audio = chunk
            return

        current_end = int(self.start_sample) + self.samples
        offset = max(0, current_end - int(start_sample))
        if offset < chunk.shape[0]:
            self.audio = np.concatenate([self.audio, chunk[offset:]], axis=0)

    def pop(self, samples: int) -> tuple[np.ndarray, Optional[int]]:
        samples = min(max(0, int(samples)), self.samples)
        if samples == 0 or self.start_sample is None:
            return _empty_audio(), None

        start_sample = int(self.start_sample)
        chunk = self.audio[:samples].copy()
        self.audio = self.audio[samples:].copy()
        self.start_sample = start_sample + samples if self.samples > 0 else None
        return chunk, start_sample

    def pop_until(self, sample: int) -> tuple[np.ndarray, Optional[int]]:
        if self.start_sample is None:
            return _empty_audio(), None
        samples = max(0, int(sample) - int(self.start_sample))
        return self.pop(samples)

    def samples_until(self, sample: int | None) -> int:
        if sample is None or self.start_sample is None:
            return self.samples
        return min(self.samples, max(0, int(sample) - int(self.start_sample)))


@dataclass
class _ActiveSpeech:
    tail_start_sample: int
    last_speech_end_sample: Optional[int] = None
    stable_text_anchor: str = ""
    previous_tail_text: str = ""
    previous_tail_end_sample: Optional[int] = None
    asr_state: Any = None
    awaiting_speech_start: bool = False
    confirmed: _SampleBuffer = field(default_factory=_SampleBuffer)


class RealtimeASRSession:
    """Single-user realtime ASR session.

    The state machine has two external states:
    - idle: no active speech segment;
    - speaking: one _ActiveSpeech owns ASR/VAD audio and transcript cursor state.
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
        self.vad = create_vad_adapter(self.config.vad)

        self.revision = 0
        self.asr_epoch = 0

        self._active: _ActiveSpeech | None = None
        self._samples_received = 0
        self._vad_base_sample = 0
        self._input_audio = _empty_audio()
        self._input_chunk_samples = max(1, int(round(self.config.sample_rate * self.config.input_chunk_ms / 1000)))
        self._pre_roll_audio = _empty_audio()
        self._pre_roll_samples = max(0, int(round(self.config.sample_rate * self.config.pre_roll_ms / 1000)))
        self._streaming_kwargs = self._low_latency_streaming_kwargs()
        self._asr_cadence_samples = self._streaming_chunk_samples(self._streaming_kwargs)
        self._live_stability_delay_samples = max(
            0,
            int(round(self.config.sample_rate * self.config.live_stability_delay_ms / 1000)),
        )

        self._last_asr_end_sample = 0

    def ingest_audio(self, pcm16k: np.ndarray) -> list[dict[str, Any]]:
        audio = normalize_pcm(pcm16k)
        if audio.shape[0] == 0:
            return []
        self._input_audio = np.concatenate([self._input_audio, audio], axis=0)
        return self._drain_input_audio(force=False)

    def flush(self) -> list[dict[str, Any]]:
        events = self._drain_input_audio(force=True)
        events.extend(self._close_speech_segment())
        self.vad.reset()
        self._vad_base_sample = self._samples_received
        self._pre_roll_audio = _empty_audio()
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self.flush()
        events.append(self.store.final_event())
        self.revision = self.store.revision
        return events

    def _ingest_audio_chunk(self, audio: np.ndarray, *, force: bool = False) -> list[dict[str, Any]]:
        chunk_start_sample = self._samples_received
        chunk_end_sample = chunk_start_sample + int(audio.shape[0])
        self._samples_received = chunk_end_sample

        decision = self.vad.accept(audio)
        active = self._active

        if active is None:
            if not decision.speech_started:
                self._remember_pre_roll(audio)
                return []
            speech_start_sample = self._absolute_vad_sample(decision.speech_start_sample, chunk_start_sample)
            active = self._open_speech(
                speech_start_sample,
                chunk_start_sample=chunk_start_sample,
                audio=audio,
            )
        else:
            active.confirmed.append(audio, start_sample=chunk_start_sample)
            if active.awaiting_speech_start and decision.speech_started:
                speech_start_sample = self._absolute_vad_sample(decision.speech_start_sample, chunk_start_sample)
                context_start = max(0, int(speech_start_sample) - self._pre_roll_samples)
                self._drop_confirmed_before(active, context_start)
                active.tail_start_sample = int(speech_start_sample)
                active.awaiting_speech_start = False

        if decision.last_speech_end_sample is not None:
            last_speech_end = self._absolute_vad_sample(decision.last_speech_end_sample, chunk_end_sample)
            active.last_speech_end_sample = (
                last_speech_end
                if active.last_speech_end_sample is None
                else max(active.last_speech_end_sample, last_speech_end)
            )

        events: list[dict[str, Any]] = []

        if decision.speech_ended and self._active is not None:
            active = self._active
            speech_end_sample = self._absolute_vad_sample(decision.speech_end_sample, chunk_end_sample)
            active.last_speech_end_sample = speech_end_sample
            if force:
                events.extend(self._close_speech_segment())
            elif self._last_asr_end_sample - speech_end_sample > self._asr_cadence_samples:
                active.awaiting_speech_start = True
            else:
                events.extend(self._finalize_endpoint(active, end_sample=speech_end_sample))
                active.awaiting_speech_start = True
            self.vad.reset()
            self._vad_base_sample = self._samples_received
            return events

        active = self._active
        if active is not None:
            if active.awaiting_speech_start:
                return events
            events.extend(self._drain_confirmed(active, force=False, emit_live=True))

        return events

    def _open_speech(self, speech_start_sample: int, *, chunk_start_sample: int, audio: np.ndarray) -> _ActiveSpeech:
        active = _ActiveSpeech(
            tail_start_sample=int(speech_start_sample),
        )

        context_start = max(0, int(speech_start_sample) - self._pre_roll_samples)
        pre_roll_start = int(chunk_start_sample) - int(self._pre_roll_audio.shape[0])
        if self._pre_roll_audio.shape[0] > 0 and context_start < chunk_start_sample:
            pre_roll_offset = max(0, context_start - pre_roll_start)
            if pre_roll_offset < self._pre_roll_audio.shape[0]:
                active.confirmed.append(
                    self._pre_roll_audio[pre_roll_offset:],
                    start_sample=pre_roll_start + pre_roll_offset,
                )

        current_start = max(int(chunk_start_sample), context_start)
        current_offset = max(0, current_start - int(chunk_start_sample))
        if current_offset < audio.shape[0]:
            active.confirmed.append(audio[current_offset:], start_sample=current_start)

        self._pre_roll_audio = _empty_audio()
        self._active = active
        return active

    def _close_speech_segment(self, *, end_sample: int | None = None) -> list[dict[str, Any]]:
        active = self._active
        if active is None:
            return []

        close_sample = self._samples_received if end_sample is None else int(end_sample)
        events = self._drain_confirmed(active, force=True, emit_live=False, until_sample=close_sample)
        if active.asr_state is None:
            if self.store.clear_partial():
                events.extend(self._emit_transcript_update(stable_base=self.store.stable_count, stable_appends=[]))
            self._active = None
            return events

        active.asr_state = self.model.finish_streaming_transcribe(active.asr_state)
        self._last_asr_end_sample = close_sample
        events.extend(self._handle_decoded_text(active, finalize=True))
        self._active = None
        return events

    def _finalize_endpoint(self, active: _ActiveSpeech, *, end_sample: int) -> list[dict[str, Any]]:
        end_sample = int(end_sample)
        events = self._drain_confirmed(active, force=True, emit_live=False, until_sample=end_sample)
        if active.asr_state is None:
            if self.store.clear_partial():
                events.extend(self._emit_transcript_update(stable_base=self.store.stable_count, stable_appends=[]))
            return events

        active.asr_state = self.model.finish_streaming_transcribe(active.asr_state)
        self._last_asr_end_sample = end_sample
        events.extend(self._handle_decoded_text(active, finalize=True))
        return events

    def _run_asr(
        self,
        active: _ActiveSpeech,
        audio: np.ndarray,
        chunk_end_sample: int,
        *,
        emit_live: bool,
    ) -> list[dict[str, Any]]:
        state = self._ensure_asr_state(active)
        active.asr_state = self.model.streaming_transcribe(audio, state)
        self._last_asr_end_sample = int(chunk_end_sample)
        if not emit_live:
            return []
        return self._handle_decoded_text(active, finalize=False)

    def _ensure_asr_state(self, active: _ActiveSpeech) -> Any:
        if active.asr_state is not None:
            return active.asr_state

        kwargs = dict(self._streaming_kwargs)
        kwargs.setdefault("context", self.config.context)
        kwargs.setdefault("language", self.config.language)
        active.asr_state = self.model.init_streaming_state(**kwargs)
        self.asr_epoch += 1
        return active.asr_state

    def _handle_decoded_text(self, active: _ActiveSpeech, *, finalize: bool) -> list[dict[str, Any]]:
        if active.asr_state is None:
            return []

        asr_text = str(getattr(active.asr_state, "text", "") or "").strip()
        tail_text = self._tail_after_stable_anchor(active.stable_text_anchor, asr_text)
        if tail_text is None:
            if finalize:
                return self._replace_partial(None)
            return []

        if finalize:
            return self._append_stable_text(
                active,
                stable_text=tail_text,
                tail_text=tail_text,
                full_asr_text=asr_text,
                end_sample=self._last_asr_end_sample,
            )

        stable_prefix = ""
        stable_end_sample: int | None = None
        if self._live_stability_delay_elapsed(active):
            stable_prefix = self._repeated_tail_prefix(active.previous_tail_text, tail_text)
            stable_end_sample = active.previous_tail_end_sample

        if stable_prefix and stable_end_sample is not None:
            return self._append_stable_text(
                active,
                stable_text=stable_prefix,
                tail_text=tail_text,
                full_asr_text=asr_text,
                end_sample=stable_end_sample,
            )

        active.previous_tail_text = tail_text
        active.previous_tail_end_sample = self._last_asr_end_sample if tail_text else None
        return self._replace_partial_text(active, tail_text)

    def _live_stability_delay_elapsed(self, active: _ActiveSpeech) -> bool:
        return self._last_asr_end_sample - active.tail_start_sample >= self._live_stability_delay_samples

    def _append_stable_text(
        self,
        active: _ActiveSpeech,
        *,
        stable_text: str,
        tail_text: str,
        full_asr_text: str,
        end_sample: int,
    ) -> list[dict[str, Any]]:
        normalized = str(stable_text or "").strip()
        if not normalized:
            active.previous_tail_text = str(tail_text or "").strip()
            active.previous_tail_end_sample = self._last_asr_end_sample if active.previous_tail_text else None
            return self._replace_partial_text(active, active.previous_tail_text)

        stable_base = self.store.stable_count
        end_sample = max(int(active.tail_start_sample), int(end_sample))
        start_ms = self._sample_to_ms(active.tail_start_sample)
        end_ms = max(start_ms, self._sample_to_ms(end_sample))
        language = str(getattr(active.asr_state, "language", "") or self.config.language or "")

        segment = self.store.append_stable_segment(
            text=normalized,
            start_ms=start_ms,
            end_ms=end_ms,
            language=language,
        )
        active.stable_text_anchor = self._advance_stable_anchor(active.stable_text_anchor, full_asr_text, normalized)
        active.tail_start_sample = end_sample

        remaining_text = self._remove_text_prefix(tail_text, normalized)
        active.previous_tail_text = remaining_text
        active.previous_tail_end_sample = self._last_asr_end_sample if remaining_text else None
        self.store.replace_partial(self._partial_segment(active, remaining_text))
        return self._emit_transcript_update(stable_base=stable_base, stable_appends=[segment])

    @staticmethod
    def _tail_after_stable_anchor(anchor: str, text: str) -> str | None:
        full_text = str(text or "").strip()
        stable_prefix = str(anchor or "").strip()
        if not full_text or not stable_prefix:
            return full_text
        if full_text.startswith(stable_prefix):
            return full_text[len(stable_prefix) :].strip()

        max_overlap = min(len(stable_prefix), len(full_text))
        for overlap in range(max_overlap, 0, -1):
            if stable_prefix[-overlap:] == full_text[:overlap]:
                return full_text[overlap:].strip()
        return None

    @staticmethod
    def _repeated_tail_prefix(previous: str, current: str) -> str:
        previous_text = str(previous or "").strip()
        current_text = str(current or "").strip()
        max_len = min(len(previous_text), len(current_text))
        index = 0
        while index < max_len and previous_text[index] == current_text[index]:
            index += 1
        next_char = current_text[index : index + 1]
        return RealtimeASRSession._trim_stable_prefix_to_boundary(current_text[:index], next_char)

    @staticmethod
    def _trim_stable_prefix_to_boundary(prefix: str, next_char: str) -> str:
        raw_prefix = str(prefix or "")
        next_text = str(next_char or "")
        right_stripped = raw_prefix.rstrip()
        stable_prefix = right_stripped.strip()
        if not stable_prefix or not next_text:
            return stable_prefix
        if len(right_stripped) < len(raw_prefix):
            return stable_prefix
        last_char = stable_prefix[-1]
        if not (
            last_char.isascii()
            and next_text[0].isascii()
            and last_char.isalnum()
            and next_text[0].isalnum()
        ):
            return stable_prefix
        for index in range(len(stable_prefix) - 1, -1, -1):
            if not stable_prefix[index].isalnum():
                return stable_prefix[: index + 1].strip()
        return ""

    @staticmethod
    def _remove_text_prefix(text: str, prefix: str) -> str:
        full_text = str(text or "").strip()
        prefix_text = str(prefix or "").strip()
        if not prefix_text:
            return full_text
        if full_text.startswith(prefix_text):
            return full_text[len(prefix_text) :].strip()
        return full_text

    @staticmethod
    def _advance_stable_anchor(anchor: str, full_asr_text: str, stable_text: str) -> str:
        current_anchor = str(anchor or "").strip()
        full_text = str(full_asr_text or "").strip()
        stable_prefix = str(stable_text or "").strip()
        if not stable_prefix:
            return current_anchor
        if current_anchor and full_text.startswith(current_anchor):
            suffix = full_text[len(current_anchor) :]
            leading = len(suffix) - len(suffix.lstrip())
            suffix_text = suffix.lstrip()
            if suffix_text.startswith(stable_prefix):
                return full_text[: len(current_anchor) + leading + len(stable_prefix)].strip()
        if not current_anchor and full_text.startswith(stable_prefix):
            return full_text[: len(stable_prefix)].strip()
        return f"{current_anchor}{stable_prefix}".strip()

    def _replace_partial_text(self, active: _ActiveSpeech, text: str) -> list[dict[str, Any]]:
        return self._replace_partial(self._partial_segment(active, text))

    def _partial_segment(self, active: _ActiveSpeech, text: str) -> PartialSegment | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None

        language = str(getattr(active.asr_state, "language", "") or self.config.language or "")
        start_ms = self._sample_to_ms(active.tail_start_sample)
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
        self.revision = self.store.revision
        return [event]

    def _drain_input_audio(self, *, force: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self._input_audio.shape[0] >= self._input_chunk_samples or (
            force and self._input_audio.shape[0] > 0
        ):
            chunk_samples = min(self._input_chunk_samples, int(self._input_audio.shape[0]))
            chunk = self._input_audio[:chunk_samples].copy()
            self._input_audio = self._input_audio[chunk_samples:].copy()
            events.extend(self._ingest_audio_chunk(chunk, force=force))
        return events

    def _drain_confirmed(
        self,
        active: _ActiveSpeech,
        *,
        force: bool,
        emit_live: bool,
        until_sample: int | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            available = active.confirmed.samples_until(until_sample)
            if available < self._asr_cadence_samples and not (force and available > 0):
                break

            chunk_samples = min(self._asr_cadence_samples, available)
            chunk, start_sample = active.confirmed.pop(chunk_samples)
            if start_sample is None:
                break
            chunk_end_sample = int(start_sample) + chunk_samples
            events.extend(self._run_asr(active, chunk, chunk_end_sample, emit_live=emit_live))
        return events

    def _drop_confirmed_before(self, active: _ActiveSpeech, sample: int) -> None:
        active.confirmed.pop_until(int(sample))

    def _remember_pre_roll(self, audio: np.ndarray) -> None:
        if self._pre_roll_samples <= 0:
            self._pre_roll_audio = _empty_audio()
            return
        if self._pre_roll_audio.shape[0] == 0:
            combined = audio
        else:
            combined = np.concatenate([self._pre_roll_audio, audio], axis=0)
        self._pre_roll_audio = combined[-self._pre_roll_samples :].copy()

    def _sample_to_ms(self, sample_index: int) -> int:
        return int(round(1000 * int(sample_index) / int(self.config.sample_rate)))

    def _absolute_vad_sample(self, vad_sample: int | None, fallback_sample: int) -> int:
        if vad_sample is None:
            return max(0, int(fallback_sample))
        return max(0, int(self._vad_base_sample) + int(vad_sample))

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
    "EnergyVadAdapter",
    "EnergyVadConfig",
    "SileroVadAdapter",
    "SileroVadConfig",
    "RealtimeASRConfig",
    "RealtimeASRSession",
    "VadDecision",
]
