# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from .audio_utils import normalize_pcm
from .firered_stream_vad_postprocessor import StreamVadPostprocessor, VadState
from .utils import SAMPLE_RATE

DEFAULT_FIRERED_STREAM_VAD_MODEL_DIR = Path("local_data/models/firered-stream-vad-onnx")
VadMode = Literal["firered-stream-vad", "none"]
FIRERED_STREAM_VAD_MODE: VadMode = "firered-stream-vad"
PASSTHROUGH_VAD_MODE: VadMode = "none"
DEFAULT_VAD_MODE: VadMode = FIRERED_STREAM_VAD_MODE
VAD_MODES: tuple[VadMode, ...] = (FIRERED_STREAM_VAD_MODE, PASSTHROUGH_VAD_MODE)
_FIRERED_FRAME_SHIFT_MS = 10
_FIRERED_FRAME_SHIFT_SAMPLES = int(SAMPLE_RATE * _FIRERED_FRAME_SHIFT_MS / 1000)
_FIRERED_FRAME_LENGTH_MS = 25
# Sentinel "never force-split" cap: realtime ASR owns long-window handling, so
# the default profile intentionally does not split continuous speech at 20s.
_NO_MAX_SPEECH_FRAMES = 1 << 62


@dataclass(frozen=True)
class VadBoundary:
    kind: Literal["speech_start", "speech_end"]
    sample: int


class VadAdapter(Protocol):
    """Audio-chunk -> ordered speech-turn boundary stream.

    ``accept``/``flush`` return the boundaries detected in that chunk (zero, one,
    or several, in timeline order); SpeechGate folds them into turn events. A turn
    left open at chunk end has a trailing speech_start with no matching speech_end
    yet. ``speech_active`` is the adapter's own live turn state (used internally),
    not part of the per-chunk return.
    """

    @property
    def speech_active(self) -> bool: ...

    def reset(self) -> None: ...

    def warmup(self) -> None: ...

    def accept(self, audio: np.ndarray) -> list[VadBoundary]: ...

    def flush(self) -> list[VadBoundary]: ...


class PassthroughVadAdapter:
    """VadAdapter that treats every non-empty chunk as speech."""

    def __init__(self) -> None:
        self._active = False
        self._samples_seen = 0

    @property
    def speech_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._samples_seen = 0

    def warmup(self) -> None:
        pass

    def accept(self, audio: np.ndarray) -> list[VadBoundary]:
        sample_count = int(np.asarray(audio).size)
        if sample_count == 0:
            return []

        start_sample = self._samples_seen
        self._samples_seen += sample_count
        starts_turn = not self._active
        self._active = True
        return [VadBoundary("speech_start", int(start_sample))] if starts_turn else []

    def flush(self) -> list[VadBoundary]:
        was_active = self._active
        self._active = False
        return [VadBoundary("speech_end", int(self._samples_seen))] if was_active else []


@dataclass
class FireRedStreamVadConfig:
    model_dir: str | Path = DEFAULT_FIRERED_STREAM_VAD_MODEL_DIR
    smooth_window_size: int = 5
    speech_threshold: float = 0.5
    pad_start_ms: int = 50
    min_speech_ms: int = 80
    min_silence_ms: int = 200
    onnx_intra_op_num_threads: int = 4
    onnx_inter_op_num_threads: int = 1

    def __post_init__(self) -> None:
        if not (0.0 < float(self.speech_threshold) <= 1.0):
            raise ValueError(
                f"speech_threshold must be in (0, 1], got: {self.speech_threshold}"
            )
        for name in (
            "smooth_window_size",
            "min_speech_ms",
            "min_silence_ms",
            "onnx_intra_op_num_threads",
            "onnx_inter_op_num_threads",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got: {value}")
        if int(self.pad_start_ms) < 0:
            raise ValueError(
                f"pad_start_ms must be non-negative, got: {self.pad_start_ms}"
            )


class _FireRedStreamRunner(Protocol):
    def predict_speech_probabilities(self, audio: np.ndarray) -> list[float]: ...

    def reset(self) -> None: ...


class FireRedStreamVadAdapter:
    """FireRed Stream-VAD speech gate using the cached streaming model."""

    def __init__(
        self,
        config: FireRedStreamVadConfig | None = None,
        *,
        runner: _FireRedStreamRunner | None = None,
    ) -> None:
        self.config = config or FireRedStreamVadConfig()
        self._runner: _FireRedStreamRunner | None = runner
        self._postprocessor = _build_stream_postprocessor(self.config)
        self._samples_seen = 0
        self._frame_sample_offset = 0

    @property
    def speech_active(self) -> bool:
        return self._postprocessor.state in (VadState.SPEECH, VadState.POSSIBLE_SILENCE)

    def reset(self) -> None:
        self._postprocessor.reset()
        self._samples_seen = 0
        self._frame_sample_offset = 0
        if self._runner is not None:
            self._runner.reset()

    def accept(self, audio: np.ndarray) -> list[VadBoundary]:
        x = normalize_pcm(audio)
        if x.shape[0] == 0:
            return []

        self._samples_seen += int(x.shape[0])
        boundaries: list[VadBoundary] = []
        for raw_prob in self._get_runner().predict_speech_probabilities(x):
            result = self._postprocessor.process_one_frame(float(raw_prob))
            if result.is_speech_start:
                boundaries.append(
                    VadBoundary(
                        "speech_start", self._event_sample(result.speech_start_frame)
                    )
                )
            if result.is_speech_end:
                boundaries.append(
                    VadBoundary(
                        "speech_end", self._event_sample(result.speech_end_frame)
                    )
                )
        return boundaries

    def flush(self) -> list[VadBoundary]:
        was_active = self.speech_active
        boundaries = (
            [VadBoundary("speech_end", int(self._samples_seen))] if was_active else []
        )
        self._postprocessor.reset()
        self._frame_sample_offset = int(self._samples_seen)
        if self._runner is not None:
            self._runner.reset()
        return boundaries

    def warmup(self) -> None:
        warmup_audio = np.full((SAMPLE_RATE // 2,), 1.0e-4, dtype=np.float32)
        self._get_runner().predict_speech_probabilities(warmup_audio)
        self.reset()

    def _get_runner(self) -> _FireRedStreamRunner:
        if self._runner is None:
            self._runner = _FireRedStreamVadOnnxRunner(self.config)
        return self._runner

    def _event_sample(self, frame_1_based: int) -> int:
        return int(self._frame_sample_offset) + _firered_frame_to_sample(
            int(frame_1_based)
        )


def _build_stream_postprocessor(
    config: FireRedStreamVadConfig,
) -> StreamVadPostprocessor:
    """Build the upstream FireRed Stream-VAD state machine from ms-based config.

    ``smooth_window_size`` is already a frame count; the ``*_ms`` knobs convert to
    10ms frames. The upstream 20s force-split is disabled (``_NO_MAX_SPEECH_FRAMES``):
    the realtime ASR owns long-window handling, and a mid-speech force-split would
    defer its restart boundary to the next frame, which the gate fold should not
    have to special-case.
    """
    return StreamVadPostprocessor(
        smooth_window_size=int(config.smooth_window_size),
        speech_threshold=float(config.speech_threshold),
        pad_start_frame=_ms_to_firered_frames(int(config.pad_start_ms)),
        min_speech_frame=_ms_to_firered_frames(int(config.min_speech_ms)),
        max_speech_frame=_NO_MAX_SPEECH_FRAMES,
        min_silence_frame=_ms_to_firered_frames(int(config.min_silence_ms)),
    )


class _FireRedStreamVadOnnxRunner:
    def __init__(self, config: FireRedStreamVadConfig) -> None:
        self.config = config
        model_dir = Path(config.model_dir)
        model_path = model_dir / "fireredvad_stream_vad_with_cache.onnx"
        cmvn_path = model_dir / "cmvn.ark"
        missing = [str(path) for path in (model_path, cmvn_path) if not path.exists()]
        if missing:
            raise RuntimeError(
                "FireRed Stream-VAD model files are missing: "
                f"{', '.join(missing)}. "
                "Set --firered-vad-model-dir or FUNYI_FIRERED_VAD_MODEL_DIR to a "
                "directory containing fireredvad_stream_vad_with_cache.onnx and cmvn.ark."
            )
        try:
            import kaldi_native_fbank as knf
            import kaldiio
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "FireRed Stream-VAD requires onnxruntime, kaldiio, and "
                "kaldi-native-fbank. Install service dependencies with "
                "`uv sync --python 3.12`."
            ) from exc

        self._knf = knf
        self._cmvn_mean, self._cmvn_inv_std = self._load_cmvn(kaldiio, cmvn_path)
        self._fbank_options = self._build_fbank_options()
        self._session = self._build_session(ort, model_path)
        self._feat_input_name, self._cache_input_name, cache_shape = (
            self._resolve_session_inputs()
        )
        self._cache = np.zeros(cache_shape, dtype=np.float32)
        self._stream_fbank: Any | None = None
        self._stream_frames_ready = 0

    def reset(self) -> None:
        self._cache.fill(0.0)
        self._stream_fbank = None
        self._stream_frames_ready = 0

    def predict_speech_probabilities(self, audio: np.ndarray) -> list[float]:
        normalized = normalize_pcm(audio)
        features = self._extract_new_features(normalized)
        if features.shape[0] == 0:
            return []
        outputs = self._session.run(
            None,
            {
                self._feat_input_name: features[None, :, :],
                self._cache_input_name: self._cache,
            },
        )
        if len(outputs) < 2:
            raise RuntimeError(
                "FireRed Stream-VAD ONNX expected probabilities and cache outputs."
            )
        self._cache = np.asarray(outputs[1], dtype=np.float32)
        probs = np.asarray(outputs[0], dtype=np.float32).squeeze()
        if probs.ndim == 0:
            probs = probs.reshape(1)
        return np.clip(probs.reshape(-1), 0.0, 1.0).astype(np.float32).tolist()

    def _build_session(self, ort: Any, model_path: Path) -> Any:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = int(
            self.config.onnx_intra_op_num_threads
        )
        session_options.inter_op_num_threads = int(
            self.config.onnx_inter_op_num_threads
        )
        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def _resolve_session_inputs(self) -> tuple[str, str, tuple[int, ...]]:
        inputs = list(self._session.get_inputs())
        if len(inputs) < 2:
            raise RuntimeError(
                "FireRed Stream-VAD ONNX must expose feat and cache inputs."
            )
        feat_input = inputs[0]
        cache_input = inputs[1]
        cache_shape = list(cache_input.shape)
        if len(cache_shape) != 4 or not all(
            isinstance(value, int) and value > 0 for value in cache_shape
        ):
            raise RuntimeError(
                f"Unsupported FireRed Stream-VAD cache shape: {cache_shape}"
            )
        return (
            str(feat_input.name),
            str(cache_input.name),
            tuple(int(value) for value in cache_shape),
        )

    def _extract_new_features(self, normalized: np.ndarray) -> np.ndarray:
        if normalized.shape[0] == 0:
            return np.zeros((0, self._cmvn_mean.shape[0]), dtype=np.float32)
        if self._stream_fbank is None:
            self._stream_fbank = self._knf.OnlineFbank(self._fbank_options)
            self._stream_frames_ready = 0
        self._stream_fbank.accept_waveform(
            SAMPLE_RATE, self._normalized_to_pcm16(normalized).tolist()
        )
        frames_ready = int(self._stream_fbank.num_frames_ready)
        if frames_ready <= self._stream_frames_ready:
            return np.zeros((0, self._cmvn_mean.shape[0]), dtype=np.float32)

        frames = [
            self._stream_fbank.get_frame(index)
            for index in range(self._stream_frames_ready, frames_ready)
        ]
        self._stream_frames_ready = frames_ready
        features = np.vstack(frames).astype(np.float32, copy=False)
        if features.shape[-1] != self._cmvn_mean.shape[0]:
            raise RuntimeError(
                "FireRed Stream-VAD feature dimension does not match CMVN stats: "
                f"features={features.shape[-1]}, cmvn={self._cmvn_mean.shape[0]}"
            )
        return (features - self._cmvn_mean) * self._cmvn_inv_std

    def _load_cmvn(
        self, kaldiio: Any, cmvn_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        stats = np.asarray(kaldiio.load_mat(str(cmvn_path)), dtype=np.float32)
        if stats.ndim != 2 or stats.shape[0] != 2 or stats.shape[1] < 2:
            raise RuntimeError(
                f"Invalid FireRed Stream-VAD CMVN stats shape: {stats.shape}"
            )
        dim = int(stats.shape[1] - 1)
        count = float(stats[0, dim])
        if count < 1.0:
            raise RuntimeError(
                "Invalid FireRed Stream-VAD CMVN stats: count must be >= 1."
            )
        mean = stats[0, :dim] / count
        variance = (stats[1, :dim] / count) - (mean * mean)
        inv_std = 1.0 / np.sqrt(np.maximum(variance, 1e-20))
        return mean.astype(np.float32), inv_std.astype(np.float32)

    def _build_fbank_options(self) -> Any:
        opts = self._knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = _FIRERED_FRAME_LENGTH_MS
        opts.frame_opts.frame_shift_ms = _FIRERED_FRAME_SHIFT_MS
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False
        return opts

    def _normalized_to_pcm16(self, normalized: np.ndarray) -> np.ndarray:
        return np.clip(normalized * 32768.0, -32768, 32767).astype(np.int16)


def _ms_to_firered_frames(ms: int) -> int:
    return max(1, int(round(int(ms) / _FIRERED_FRAME_SHIFT_MS)))


def _firered_frame_to_sample(frame_1_based: int) -> int:
    return max(0, (int(frame_1_based) - 1) * _FIRERED_FRAME_SHIFT_SAMPLES)


__all__ = [
    "DEFAULT_FIRERED_STREAM_VAD_MODEL_DIR",
    "DEFAULT_VAD_MODE",
    "FIRERED_STREAM_VAD_MODE",
    "FireRedStreamVadAdapter",
    "FireRedStreamVadConfig",
    "PASSTHROUGH_VAD_MODE",
    "PassthroughVadAdapter",
    "VadAdapter",
    "VadBoundary",
    "VadMode",
    "VAD_MODES",
]
