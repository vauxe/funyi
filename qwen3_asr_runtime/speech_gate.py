# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .audio_utils import normalize_pcm
from .utils import SAMPLE_RATE
from .vad import VadAdapter, VadBoundary


@dataclass(frozen=True)
class SpeechGateConfig:
    sample_rate: int = SAMPLE_RATE
    pre_roll_ms: int = 400


@dataclass(frozen=True)
class SpeechGateEvent:
    type: Literal["speech_start", "speech_audio", "speech_end"]
    start_sample: int
    end_sample: int
    audio: np.ndarray


class SpeechGate:
    """Convert a continuous PCM stream into speech-turn events."""

    def __init__(
        self,
        *,
        vad: VadAdapter,
        config: SpeechGateConfig | None = None,
    ) -> None:
        self.config = config or SpeechGateConfig()
        if int(self.config.sample_rate) != SAMPLE_RATE:
            raise ValueError(
                f"SpeechGate expects {SAMPLE_RATE} Hz audio, got {self.config.sample_rate}."
            )
        self.vad = vad
        self._pre_roll_samples = max(
            0, int(round(SAMPLE_RATE * int(self.config.pre_roll_ms) / 1000))
        )
        self._samples_seen = 0
        self._speech_active = False
        self._idle_floor_sample = 0
        self._speech_audio_cursor = 0
        self._buffer_start_sample = 0
        self._buffer_audio = np.zeros((0,), dtype=np.float32)

    @property
    def samples_seen(self) -> int:
        return int(self._samples_seen)

    @property
    def speech_active(self) -> bool:
        return bool(self._speech_active)

    def reset(self) -> None:
        self.vad.reset()
        self._samples_seen = 0
        self._speech_active = False
        self._idle_floor_sample = 0
        self._speech_audio_cursor = 0
        self._buffer_start_sample = 0
        self._buffer_audio = np.zeros((0,), dtype=np.float32)

    def accept(self, pcm16k: np.ndarray) -> list[SpeechGateEvent]:
        audio = normalize_pcm(pcm16k)
        if audio.shape[0] == 0:
            return []

        chunk_start = int(self._samples_seen)
        chunk_end = chunk_start + int(audio.shape[0])
        self._append_buffer(audio, chunk_start)
        events = self._consume_boundaries(self.vad.accept(audio), chunk_end)
        self._samples_seen = chunk_end
        self._trim_buffer()
        return events

    def flush(self, *, force_end: bool = False) -> list[SpeechGateEvent]:
        if not force_end:
            self._trim_buffer()
            return []

        events = self._consume_boundaries(self.vad.flush(), self._samples_seen)
        if self._speech_active:
            # End of stream: force-close a turn the VAD left open (FireRed emits no
            # trailing speech_end without trailing silence). Its audio was already
            # flushed by _consume_boundaries' trailing speech_audio.
            self._append_speech_end(events, self._samples_seen)
        self._idle_floor_sample = max(
            int(self._idle_floor_sample), int(self._samples_seen)
        )
        self._trim_buffer()
        return events

    def _consume_boundaries(
        self, boundaries: list[VadBoundary], chunk_end: int
    ) -> list[SpeechGateEvent]:
        """Fold an ordered VAD boundary stream into speech-turn events.

        Linear over the boundaries, so any number of turns in one chunk is handled
        uniformly; the gate's active state is derived from the stream itself.
        """
        events: list[SpeechGateEvent] = []
        for index, boundary in enumerate(boundaries):
            if boundary.kind == "speech_start":
                # The speech_start event carries pre-roll plus audio up to the next
                # boundary (or chunk end if this turn keeps running).
                span_end = (
                    boundaries[index + 1].sample
                    if index + 1 < len(boundaries)
                    else chunk_end
                )
                if self._append_speech_start(events, boundary.sample, span_end):
                    self._speech_active = True
                    self._speech_audio_cursor = span_end
            else:
                speech_end = min(
                    max(int(boundary.sample), self._speech_audio_cursor), chunk_end
                )
                self._append_speech_audio(events, speech_end)
                self._append_speech_end(events, speech_end)
        if self._speech_active:
            self._append_speech_audio(events, chunk_end)
        return events

    def _append_speech_start(
        self, events: list[SpeechGateEvent], start_sample: int, end_sample: int
    ) -> bool:
        event_start = max(
            int(self._idle_floor_sample), int(start_sample) - self._pre_roll_samples
        )
        return self._append_event(events, "speech_start", event_start, end_sample)

    def _append_speech_end(
        self, events: list[SpeechGateEvent], end_sample: int
    ) -> None:
        end_sample = int(end_sample)
        self._speech_active = False
        self._idle_floor_sample = end_sample
        events.append(
            SpeechGateEvent(
                type="speech_end",
                start_sample=end_sample,
                end_sample=end_sample,
                audio=np.zeros((0,), dtype=np.float32),
            )
        )

    def _append_speech_audio(
        self, events: list[SpeechGateEvent], end_sample: int
    ) -> None:
        end_sample = int(end_sample)
        self._append_event(
            events,
            "speech_audio",
            self._speech_audio_cursor,
            end_sample,
        )
        self._speech_audio_cursor = end_sample

    def _append_event(
        self,
        events: list[SpeechGateEvent],
        event_type: Literal["speech_start", "speech_audio"],
        start_sample: int,
        end_sample: int,
    ) -> bool:
        start_sample = max(int(start_sample), int(self._buffer_start_sample))
        end_sample = min(int(end_sample), self._buffer_end_sample())
        if end_sample <= start_sample:
            return False
        start = start_sample - int(self._buffer_start_sample)
        end = end_sample - int(self._buffer_start_sample)
        events.append(
            SpeechGateEvent(
                type=event_type,
                start_sample=start_sample,
                end_sample=end_sample,
                audio=self._buffer_audio[start:end].copy(),
            )
        )
        return True

    def _append_buffer(self, audio: np.ndarray, start_sample: int) -> None:
        chunk = audio.astype(np.float32, copy=True)
        if (
            self._buffer_audio.shape[0] == 0
            or int(start_sample) != self._buffer_end_sample()
        ):
            self._buffer_start_sample = int(start_sample)
            self._buffer_audio = chunk
            return
        self._buffer_audio = np.concatenate([self._buffer_audio, chunk], axis=0)

    def _trim_buffer(self) -> None:
        if self._speech_active:
            keep_from = self._speech_audio_cursor
        else:
            keep_from = max(
                int(self._idle_floor_sample),
                int(self._samples_seen) - self._pre_roll_samples,
            )
        self._drop_before(keep_from)

    def _drop_before(self, sample: int) -> None:
        sample = max(int(sample), int(self._buffer_start_sample))
        drop = min(
            sample - int(self._buffer_start_sample), int(self._buffer_audio.shape[0])
        )
        if drop <= 0:
            return
        self._buffer_audio = self._buffer_audio[drop:].copy()
        self._buffer_start_sample += drop

    def _buffer_end_sample(self) -> int:
        return int(self._buffer_start_sample) + int(self._buffer_audio.shape[0])


__all__ = [
    "SpeechGate",
    "SpeechGateConfig",
    "SpeechGateEvent",
]
