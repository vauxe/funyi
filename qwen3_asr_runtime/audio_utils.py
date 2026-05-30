# coding=utf-8
from __future__ import annotations

import numpy as np


def normalize_pcm(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32, copy=False)


__all__ = ["normalize_pcm"]
