# coding=utf-8
from __future__ import annotations

import gc
import random
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return table[name]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dispose_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _default_attn_implementation(device_map: Any = None) -> Optional[str]:
    if find_spec("flash_attn") is None and find_spec("flash_attn_2_cuda") is None:
        return None
    if torch.cuda.is_available():
        return "flash_attention_2"
    if device_map is not None and "cuda" in str(device_map).lower():
        return "flash_attention_2"
    return None


def _serialize_transcriptions(items: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "language": item.language,
            "text": item.text,
            "time_stamps": item.time_stamps,
        }
        for item in items
    ]


def _cer(hyp: str, ref: str) -> float:
    from tools.sweep_cer_vs_srt import _cer as compute_cer

    return compute_cer(hyp, ref)


def _extract_srt_text_range(path: str | Path, start_sec: float, duration_sec: float) -> str:
    from tools.sweep_cer_vs_srt import load_srt, srt_text_in_window

    return srt_text_in_window(load_srt(str(path)), start_sec, duration_sec)
