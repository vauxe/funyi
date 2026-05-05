# coding=utf-8
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class StreamingCaseSpec:
    name: str
    start_sec: float
    duration_sec: float
    context: str
    language: Optional[str]


DEFAULT_STREAMING_CASES: tuple[StreamingCaseSpec, ...] = (
    StreamingCaseSpec(name="short_default_15s", start_sec=0.0, duration_sec=15.0, context="", language=None),
    StreamingCaseSpec(name="short_context_15s", start_sec=0.0, duration_sec=15.0, context="交易 停滞", language=None),
    StreamingCaseSpec(name="short_forced_language_15s", start_sec=0.0, duration_sec=15.0, context="", language="English"),
)

DEFAULT_STEP_MS: tuple[int, ...] = (500, 2000)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_int_list(value: str | None, *, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer value.")
    return items


def parse_name_filter(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    names = {item.strip() for item in str(value).split(",") if item.strip()}
    return names or None


def selected_default_cases(names: set[str] | None) -> list[StreamingCaseSpec]:
    cases = list(DEFAULT_STREAMING_CASES)
    if names is None:
        return cases
    available = {case.name for case in cases}
    unknown = sorted(names.difference(available))
    if unknown:
        raise ValueError(f"Unknown streaming cases: {unknown}. Available: {sorted(available)}")
    return [case for case in cases if case.name in names]


def summarize_seconds(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min_sec": 0.0,
            "median_sec": 0.0,
            "max_sec": 0.0,
            "mean_sec": 0.0,
        }
    ordered = sorted(values)
    return {
        "min_sec": round(ordered[0], 4),
        "median_sec": round(float(median(ordered)), 4),
        "max_sec": round(ordered[-1], 4),
        "mean_sec": round(sum(ordered) / len(ordered), 4),
    }


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _state_snapshot(
    *,
    state: Any,
    event: str,
    audio_samples_seen: int,
    sample_rate: int,
    push_index: Optional[int],
    segment_samples: int,
    decode_steps: int,
) -> dict[str, Any]:
    text = str(getattr(state, "text", "") or "")
    raw = str(getattr(state, "_raw_decoded", "") or "")
    buffer_samples = int(getattr(state, "buffer").shape[0])
    audio_accum_samples = int(getattr(state, "audio_accum").shape[0])
    snapshot: dict[str, Any] = {
        "event": event,
        "push_index": push_index,
        "audio_ms_seen": round(audio_samples_seen * 1000.0 / float(sample_rate), 3),
        "segment_samples": int(segment_samples),
        "decode_steps": int(decode_steps),
        "chunk_id": int(getattr(state, "chunk_id")),
        "buffer_samples": buffer_samples,
        "audio_accum_samples": audio_accum_samples,
        "language": str(getattr(state, "language", "") or ""),
        "text_chars": len(text),
        "text_sha256": text_sha256(text),
        "raw_chars": len(raw),
        "raw_sha256": text_sha256(raw),
    }
    if getattr(state, "max_window_samples", None) is not None:
        committed = str(getattr(state, "committed_text", "") or "")
        partial = str(getattr(state, "partial_text", "") or "")
        snapshot.update(
            {
                "max_window_samples": int(getattr(state, "max_window_samples")),
                "max_prefix_tokens": int(getattr(state, "max_prefix_tokens")),
                "audio_seen_samples": int(getattr(state, "audio_seen_samples")),
                "audio_dropped_samples": int(getattr(state, "audio_dropped_samples")),
                "committed_text_chars": len(committed),
                "committed_text_sha256": text_sha256(committed),
                "partial_text_chars": len(partial),
                "partial_text_sha256": text_sha256(partial),
            }
        )
    return snapshot


def final_state_payload(state: Any) -> dict[str, Any]:
    text = str(getattr(state, "text", "") or "")
    raw = str(getattr(state, "_raw_decoded", "") or "")
    payload = {
        "language": str(getattr(state, "language", "") or ""),
        "text": text,
        "text_sha256": text_sha256(text),
        "raw_sha256": text_sha256(raw),
        "text_chars": len(text),
        "raw_chars": len(raw),
        "chunk_id": int(getattr(state, "chunk_id")),
    }
    if getattr(state, "max_window_samples", None) is not None:
        committed = str(getattr(state, "committed_text", "") or "")
        partial = str(getattr(state, "partial_text", "") or "")
        payload.update(
            {
                "max_window_samples": int(getattr(state, "max_window_samples")),
                "max_prefix_tokens": int(getattr(state, "max_prefix_tokens")),
                "audio_seen_samples": int(getattr(state, "audio_seen_samples")),
                "audio_dropped_samples": int(getattr(state, "audio_dropped_samples")),
                "committed_text_chars": len(committed),
                "committed_text_sha256": text_sha256(committed),
                "partial_text_chars": len(partial),
                "partial_text_sha256": text_sha256(partial),
            }
        )
    return payload


def run_streaming_case(
    *,
    model: Any,
    wav16k: np.ndarray,
    sample_rate: int,
    case: StreamingCaseSpec,
    step_ms: int,
    chunk_size_sec: float,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    max_window_sec: float | None = None,
    max_prefix_tokens: int | None = None,
    timed: bool = False,
    spec_decode: bool = False,
    include_internal_stats: bool = False,
) -> dict[str, Any]:
    step_samples = max(1, int(round(float(step_ms) / 1000.0 * sample_rate)))
    if isinstance(wav16k, tuple) and len(wav16k) == 2:
        wav, sr = wav16k
        if int(sr) != int(sample_rate):
            raise ValueError(f"Streaming audio must be {sample_rate} Hz, got {sr}.")
        audio = np.asarray(wav, dtype=np.float32)
    else:
        audio = np.asarray(wav16k, dtype=np.float32)

    state = model.init_streaming_state(
        context=case.context,
        language=case.language,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
        max_window_sec=max_window_sec,
        max_prefix_tokens=max_prefix_tokens,
        spec_decode=spec_decode,
    )

    snapshots: list[dict[str, Any]] = [
        _state_snapshot(
            state=state,
            event="init",
            audio_samples_seen=0,
            sample_rate=sample_rate,
            push_index=None,
            segment_samples=0,
            decode_steps=0,
        )
    ]
    event_timings: list[dict[str, Any]] = []
    first_text_audio_ms: float | None = None
    model_decode_updates = 0

    pos = 0
    push_index = 0
    if timed:
        _sync_cuda()
    total_t0 = time.perf_counter()
    while pos < audio.shape[0]:
        seg = audio[pos : pos + step_samples]
        pos += int(seg.shape[0])
        push_index += 1

        before_chunk = int(state.chunk_id)
        t0 = time.perf_counter()
        if timed:
            _sync_cuda()
            t0 = time.perf_counter()
        model.streaming_transcribe(seg, state)
        if timed:
            _sync_cuda()
        elapsed = time.perf_counter() - t0

        decode_steps = int(state.chunk_id) - before_chunk
        model_decode_updates += decode_steps
        if first_text_audio_ms is None and str(state.text or ""):
            first_text_audio_ms = pos * 1000.0 / float(sample_rate)

        snapshots.append(
            _state_snapshot(
                state=state,
                event="push",
                audio_samples_seen=pos,
                sample_rate=sample_rate,
                push_index=push_index,
                segment_samples=int(seg.shape[0]),
                decode_steps=decode_steps,
            )
        )
        if timed:
            event_timings.append(
                {
                    "event": "push",
                    "push_index": push_index,
                    "decode_steps": decode_steps,
                    "audio_ms_seen": round(pos * 1000.0 / float(sample_rate), 3),
                    "wall_sec": elapsed,
                }
            )

    before_chunk = int(state.chunk_id)
    t0 = time.perf_counter()
    if timed:
        _sync_cuda()
        t0 = time.perf_counter()
    model.finish_streaming_transcribe(state)
    if timed:
        _sync_cuda()
    finish_elapsed = time.perf_counter() - t0
    total_wall = time.perf_counter() - total_t0

    finish_decode_steps = int(state.chunk_id) - before_chunk
    model_decode_updates += finish_decode_steps
    if first_text_audio_ms is None and str(state.text or ""):
        first_text_audio_ms = audio.shape[0] * 1000.0 / float(sample_rate)

    snapshots.append(
        _state_snapshot(
            state=state,
            event="finish",
            audio_samples_seen=int(audio.shape[0]),
            sample_rate=sample_rate,
            push_index=None,
            segment_samples=0,
            decode_steps=finish_decode_steps,
        )
    )
    if timed:
        event_timings.append(
            {
                "event": "finish",
                "push_index": None,
                "decode_steps": finish_decode_steps,
                "audio_ms_seen": round(audio.shape[0] * 1000.0 / float(sample_rate), 3),
                "wall_sec": finish_elapsed,
            }
        )

    payload: dict[str, Any] = {
        "step_ms": int(step_ms),
        "snapshots": snapshots,
        "final": final_state_payload(state),
        "metrics": {
            "push_count": push_index,
            "model_decode_updates": model_decode_updates,
            "finish_decode_steps": finish_decode_steps,
            "audio_duration_sec": round(audio.shape[0] / float(sample_rate), 6),
            "first_text_audio_ms": round(first_text_audio_ms, 3) if first_text_audio_ms is not None else None,
        },
    }
    if include_internal_stats:
        payload["metrics"]["spec_decode_stats"] = dict(getattr(state, "spec_decode_stats", {}) or {})
    if timed:
        active_update_walls = [item["wall_sec"] for item in event_timings if item["decode_steps"] > 0]
        all_push_walls = [item["wall_sec"] for item in event_timings if item["event"] == "push"]
        payload["timing"] = {
            "total_wall_sec": round(total_wall, 4),
            "active_update_wall": summarize_seconds(active_update_walls),
            "all_push_wall": summarize_seconds(all_push_walls),
            "events": [
                {
                    **item,
                    "wall_sec": round(float(item["wall_sec"]), 4),
                }
                for item in event_timings
            ],
        }
    return payload


def first_diff_index(expected: list[Any], actual: list[Any]) -> int | None:
    limit = min(len(expected), len(actual))
    for idx in range(limit):
        if expected[idx] != actual[idx]:
            return idx
    if len(expected) != len(actual):
        return limit
    return None
