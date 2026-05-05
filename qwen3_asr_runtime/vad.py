# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, TypeAlias

import numpy as np

from .utils import SAMPLE_RATE


@dataclass
class VadDecision:
    speech_started: bool = False
    speech_ended: bool = False
    has_speech: bool = False
    speech_active: bool = False
    speech_start_sample: Optional[int] = None
    last_speech_end_sample: Optional[int] = None
    speech_end_sample: Optional[int] = None


class VadAdapter(Protocol):
    @property
    def speech_active(self) -> bool: ...

    def reset(self) -> None: ...

    def accept(self, audio: np.ndarray) -> VadDecision: ...


@dataclass
class EnergyVadConfig:
    sample_rate: int = SAMPLE_RATE
    frame_ms: int = 30
    speech_threshold: float = 0.01
    min_speech_ms: int = 120
    min_silence_ms: int = 600


class EnergyVadAdapter:
    """Small dependency-free VAD for fallback and smoke tests."""

    def __init__(self, config: EnergyVadConfig | None = None) -> None:
        self.config = config or EnergyVadConfig()
        self._frame_samples = max(1, int(round(self.config.sample_rate * self.config.frame_ms / 1000)))
        self._active = False
        self._candidate_speech_ms = 0
        self._silence_ms = 0
        self._processed_samples = 0
        self._candidate_speech_start_sample: int | None = None
        self._silence_start_sample: int | None = None

    @property
    def speech_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._candidate_speech_ms = 0
        self._silence_ms = 0
        self._processed_samples = 0
        self._candidate_speech_start_sample = None
        self._silence_start_sample = None

    def accept(self, audio: np.ndarray) -> VadDecision:
        x = normalize_pcm(audio)
        if x.shape[0] == 0:
            return VadDecision(speech_active=self._active)

        decision = VadDecision(speech_active=self._active)
        for start in range(0, x.shape[0], self._frame_samples):
            frame = x[start : start + self._frame_samples]
            if frame.shape[0] == 0:
                continue
            frame_start = self._processed_samples
            frame_end = frame_start + int(frame.shape[0])
            rms = float(np.sqrt(np.mean(frame * frame)))
            is_speech = rms >= float(self.config.speech_threshold)
            self._apply_frame(
                decision,
                is_speech=is_speech,
                frame_ms=self.config.frame_ms,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            self._processed_samples = frame_end

        decision.speech_active = self._active
        return decision

    def _apply_frame(
        self,
        decision: VadDecision,
        *,
        is_speech: bool,
        frame_ms: int,
        frame_start: int,
        frame_end: int,
    ) -> None:
        decision.has_speech = decision.has_speech or is_speech
        if is_speech:
            decision.last_speech_end_sample = frame_end

        if self._active:
            if is_speech:
                self._silence_ms = 0
                self._silence_start_sample = None
                return
            if self._silence_start_sample is None:
                self._silence_start_sample = frame_start
            self._silence_ms += frame_ms
            if self._silence_ms >= self.config.min_silence_ms:
                self._active = False
                self._silence_ms = 0
                self._candidate_speech_ms = 0
                decision.speech_end_sample = self._silence_start_sample
                self._candidate_speech_start_sample = None
                self._silence_start_sample = None
                decision.speech_ended = True
            return

        if is_speech:
            if self._candidate_speech_ms == 0:
                self._candidate_speech_start_sample = frame_start
            self._candidate_speech_ms += frame_ms
            if self._candidate_speech_ms >= self.config.min_speech_ms:
                self._active = True
                self._silence_ms = 0
                decision.speech_start_sample = self._candidate_speech_start_sample
                decision.speech_started = True
        else:
            self._candidate_speech_ms = 0
            self._candidate_speech_start_sample = None


@dataclass
class SileroVadConfig:
    sample_rate: int = SAMPLE_RATE
    threshold: float = 0.5
    negative_threshold: Optional[float] = None
    min_speech_ms: int = 160
    min_silence_ms: int = 700
    use_onnx: bool = False


class SileroVadAdapter:
    """Streaming Silero VAD wrapper with the same endpoint semantics as EnergyVadAdapter."""

    def __init__(self, config: SileroVadConfig | None = None, *, model: Any | None = None) -> None:
        self.config = config or SileroVadConfig()
        if self.config.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Silero VAD expects {SAMPLE_RATE} Hz audio, got {self.config.sample_rate}.")

        self._chunk_samples = 512
        self._chunk_ms = int(round(1000 * self._chunk_samples / self.config.sample_rate))
        self._negative_threshold = (
            float(self.config.negative_threshold)
            if self.config.negative_threshold is not None
            else max(0.0, float(self.config.threshold) - 0.15)
        )
        self._model = model if model is not None else self._load_model()
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._active = False
        self._candidate_speech_ms = 0
        self._silence_ms = 0
        self._processed_samples = 0
        self._candidate_speech_start_sample: int | None = None
        self._silence_start_sample: int | None = None

    @property
    def speech_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._active = False
        self._candidate_speech_ms = 0
        self._silence_ms = 0
        self._processed_samples = 0
        self._candidate_speech_start_sample = None
        self._silence_start_sample = None
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def accept(self, audio: np.ndarray) -> VadDecision:
        x = normalize_pcm(audio)
        if x.shape[0] == 0:
            return VadDecision(speech_active=self._active)

        self._buffer = np.concatenate([self._buffer, x], axis=0)
        decision = VadDecision(speech_active=self._active)

        offset = 0
        while self._buffer.shape[0] - offset >= self._chunk_samples:
            frame = self._buffer[offset : offset + self._chunk_samples]
            offset += self._chunk_samples
            frame_start = self._processed_samples
            frame_end = frame_start + self._chunk_samples
            speech_prob = self._predict(frame)
            if self._active:
                is_speech = speech_prob >= self._negative_threshold
            else:
                is_speech = speech_prob >= float(self.config.threshold)
            self._apply_frame(decision, is_speech=is_speech, frame_start=frame_start, frame_end=frame_end)
            self._processed_samples = frame_end

        if offset:
            self._buffer = self._buffer[offset:].copy()
        decision.speech_active = self._active
        return decision

    def _apply_frame(self, decision: VadDecision, *, is_speech: bool, frame_start: int, frame_end: int) -> None:
        decision.has_speech = decision.has_speech or is_speech
        if is_speech:
            decision.last_speech_end_sample = frame_end

        if self._active:
            if is_speech:
                self._silence_ms = 0
                self._silence_start_sample = None
                return
            if self._silence_start_sample is None:
                self._silence_start_sample = frame_start
            self._silence_ms += self._chunk_ms
            if self._silence_ms >= int(self.config.min_silence_ms):
                self._active = False
                self._silence_ms = 0
                self._candidate_speech_ms = 0
                decision.speech_end_sample = self._silence_start_sample
                self._candidate_speech_start_sample = None
                self._silence_start_sample = None
                decision.speech_ended = True
            return

        if is_speech:
            if self._candidate_speech_ms == 0:
                self._candidate_speech_start_sample = frame_start
            self._candidate_speech_ms += self._chunk_ms
            if self._candidate_speech_ms >= int(self.config.min_speech_ms):
                self._active = True
                self._silence_ms = 0
                decision.speech_start_sample = self._candidate_speech_start_sample
                decision.speech_started = True
        else:
            self._candidate_speech_ms = 0
            self._candidate_speech_start_sample = None

    def _predict(self, frame: np.ndarray) -> float:
        import torch

        tensor = torch.from_numpy(np.asarray(frame, dtype=np.float32))
        with torch.inference_mode():
            return float(self._model(tensor, self.config.sample_rate).item())

    def _load_model(self) -> Any:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD requires service dependencies. Install with `uv sync --python 3.12`."
            ) from exc

        try:
            return load_silero_vad(onnx=bool(self.config.use_onnx))
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD model loading failed because an optional runtime dependency is missing. "
                "Install with `uv sync --python 3.12`, or set `use_onnx=false`."
            ) from exc


VadConfig: TypeAlias = EnergyVadConfig | SileroVadConfig


def create_vad_adapter(config: VadConfig | None = None) -> VadAdapter:
    if config is None:
        config = SileroVadConfig()
    if isinstance(config, EnergyVadConfig):
        return EnergyVadAdapter(config)
    if isinstance(config, SileroVadConfig):
        return SileroVadAdapter(config)
    raise TypeError(f"Unsupported VAD config: {type(config).__name__}")


def normalize_pcm(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32, copy=False)


__all__ = [
    "EnergyVadAdapter",
    "EnergyVadConfig",
    "SileroVadAdapter",
    "SileroVadConfig",
    "VadAdapter",
    "VadConfig",
    "VadDecision",
    "create_vad_adapter",
    "normalize_pcm",
]
