# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import SAMPLE_RATE
from .vad import VadAdapter, create_vad_adapter, normalize_pcm


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
        vad: VadAdapter | None = None,
        config: SpeechGateConfig | None = None,
    ) -> None:
        self.config = config or SpeechGateConfig()
        if int(self.config.sample_rate) != SAMPLE_RATE:
            raise ValueError(f"SpeechGate expects {SAMPLE_RATE} Hz audio, got {self.config.sample_rate}.")
        self.vad = vad or create_vad_adapter()
        self._pre_roll_samples = max(0, int(round(SAMPLE_RATE * int(self.config.pre_roll_ms) / 1000)))
        self._samples_seen = 0
        self._speech_active = False
        self._idle_floor_sample = 0
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
        self._buffer_start_sample = 0
        self._buffer_audio = np.zeros((0,), dtype=np.float32)

    def accept(self, pcm16k: np.ndarray) -> list[SpeechGateEvent]:
        audio = normalize_pcm(pcm16k)
        if audio.shape[0] == 0:
            return []

        chunk_start = int(self._samples_seen)
        chunk_end = chunk_start + int(audio.shape[0])
        self._append_buffer(audio, chunk_start)

        decision = self.vad.accept(audio)
        start_sample = int(decision.speech_start_sample) if decision.speech_start_sample is not None else None
        end_sample = int(decision.speech_end_sample) if decision.speech_end_sample is not None else None

        events: list[SpeechGateEvent] = []
        if self._speech_active:
            self._accept_active_chunk(events, chunk_start, chunk_end, start_sample, end_sample)
        else:
            self._accept_idle_chunk(events, chunk_end, start_sample, end_sample)

        self._samples_seen = chunk_end
        self._trim_buffer()
        return events

    def _accept_idle_chunk(
        self,
        events: list[SpeechGateEvent],
        chunk_end: int,
        start_sample: int | None,
        end_sample: int | None,
    ) -> None:
        if start_sample is None:
            return

        speech_end = chunk_end if end_sample is None or end_sample < start_sample else min(end_sample, chunk_end)
        self._append_speech_start(events, start_sample, speech_end)
        self._speech_active = True

        if end_sample is not None and end_sample >= start_sample:
            self._append_speech_end(events, end_sample)

    def _accept_active_chunk(
        self,
        events: list[SpeechGateEvent],
        chunk_start: int,
        chunk_end: int,
        start_sample: int | None,
        end_sample: int | None,
    ) -> None:
        if end_sample is None:
            self._append_event(events, "speech_audio", chunk_start, chunk_end)
            return

        self._append_event(events, "speech_audio", chunk_start, min(end_sample, chunk_end))
        self._append_speech_end(events, end_sample)
        if start_sample is not None and start_sample > end_sample:
            self._append_speech_start(events, start_sample, chunk_end)
            self._speech_active = True

    def _append_speech_start(self, events: list[SpeechGateEvent], start_sample: int, end_sample: int) -> None:
        event_start = max(int(self._idle_floor_sample), int(start_sample) - self._pre_roll_samples)
        self._append_event(events, "speech_start", event_start, end_sample)

    def _append_speech_end(self, events: list[SpeechGateEvent], end_sample: int) -> None:
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

    def _append_event(
        self,
        events: list[SpeechGateEvent],
        event_type: Literal["speech_start", "speech_audio"],
        start_sample: int,
        end_sample: int,
    ) -> None:
        start_sample = max(int(start_sample), int(self._buffer_start_sample))
        end_sample = min(int(end_sample), self._buffer_end_sample())
        if end_sample <= start_sample:
            return
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

    def _append_buffer(self, audio: np.ndarray, start_sample: int) -> None:
        chunk = audio.astype(np.float32, copy=True)
        if self._buffer_audio.shape[0] == 0 or int(start_sample) != self._buffer_end_sample():
            self._buffer_start_sample = int(start_sample)
            self._buffer_audio = chunk
            return
        self._buffer_audio = np.concatenate([self._buffer_audio, chunk], axis=0)

    def _trim_buffer(self) -> None:
        if self._speech_active:
            keep_from = self._samples_seen
        else:
            keep_from = max(int(self._idle_floor_sample), int(self._samples_seen) - self._pre_roll_samples)
        self._drop_before(keep_from)

    def _drop_before(self, sample: int) -> None:
        sample = max(int(sample), int(self._buffer_start_sample))
        drop = min(sample - int(self._buffer_start_sample), int(self._buffer_audio.shape[0]))
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
