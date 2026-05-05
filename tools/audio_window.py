# coding=utf-8
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def slice_audio(wav: np.ndarray, sample_rate: int, start_sec: float, duration_sec: float) -> tuple[np.ndarray, int]:
    start_sample = int(round(float(start_sec) * sample_rate))
    end_sample = start_sample + int(round(float(duration_sec) * sample_rate))
    return wav[start_sample:end_sample], sample_rate


def load_audio_window(
    audio_source: Any,
    *,
    sample_rate: int,
    start_sec: float,
    duration_sec: float,
    normalize_audios: Callable[[Any], list[np.ndarray]],
) -> tuple[np.ndarray, int]:
    wav = normalize_audios(audio_source)[0]
    return slice_audio(wav, sample_rate, start_sec, duration_sec)
