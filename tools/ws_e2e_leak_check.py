# coding=utf-8
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import websockets
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class ResourceSample:
    elapsed_sec: float
    audio_sent_sec: float
    rss_mb: float | None
    gpu_used_mb: int | None
    gpu_total_mb: int | None
    gpu_util_pct: int | None
    gpu_temp_c: int | None
    transcript_update_count: int
    transcript_final_count: int


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _summarize_event_timings(
    event_times: list[dict[str, Any]],
    *,
    finish_sent_wall_sec: float | None,
    elapsed_sec: float,
    audio_sent_sec: float,
) -> dict[str, Any]:
    def times_for(event_type: str) -> list[float]:
        return [
            float(item["wall_sec"])
            for item in event_times
            if item.get("type") == event_type
        ]

    update_times = times_for("transcript_update")
    timing_update_times = times_for("transcript_timing_update")
    final_times = times_for("transcript_final")
    update_gaps = [right - left for left, right in zip(update_times, update_times[1:])]
    stable_emit_wall_by_id: dict[str, float] = {}
    timing_update_lags: list[float] = []
    for item in event_times:
        if item.get("type") == "transcript_update":
            for segment_id in item.get("stable_segment_ids") or []:
                stable_emit_wall_by_id[str(segment_id)] = float(item["wall_sec"])
        elif item.get("type") == "transcript_timing_update":
            segment_id = str(item.get("source_segment_id") or "")
            if segment_id in stable_emit_wall_by_id:
                timing_update_lags.append(
                    float(item["wall_sec"]) - stable_emit_wall_by_id[segment_id]
                )

    final_wall = final_times[0] if final_times else None
    return {
        "first_transcript_update_wall_sec": round(update_times[0], 3)
        if update_times
        else None,
        "final_wall_sec": round(final_wall, 3) if final_wall is not None else None,
        "finish_to_final_sec": (
            round(final_wall - finish_sent_wall_sec, 3)
            if final_wall is not None and finish_sent_wall_sec is not None
            else None
        ),
        "update_gap_p95_sec": (
            round(_percentile(update_gaps, 0.95), 3)
            if _percentile(update_gaps, 0.95) is not None
            else None
        ),
        "first_timing_update_wall_sec": round(timing_update_times[0], 3)
        if timing_update_times
        else None,
        "timing_update_lag_p50_sec": (
            round(_percentile(timing_update_lags, 0.50), 3)
            if _percentile(timing_update_lags, 0.50) is not None
            else None
        ),
        "timing_update_lag_p95_sec": (
            round(_percentile(timing_update_lags, 0.95), 3)
            if _percentile(timing_update_lags, 0.95) is not None
            else None
        ),
        "processing_speed_x": round(audio_sent_sec / elapsed_sec, 3)
        if elapsed_sec > 0
        else None,
    }


def _translation_validation_issues(
    *,
    ready_event: dict[str, Any] | None,
    counters: dict[str, int],
    expect_translation: bool,
    min_translation_stable: int,
    min_translation_preview: int,
    max_translation_status: int | None,
) -> list[str]:
    issues: list[str] = []
    if expect_translation:
        translation_ready = (ready_event or {}).get("translation")
        if not isinstance(translation_ready, dict) or not translation_ready.get(
            "enabled"
        ):
            issues.append("ready.translation is missing or disabled")
    stable_count = counters.get("translation_stable", 0)
    if stable_count < min_translation_stable:
        issues.append(
            f"translation_stable count {stable_count} < {min_translation_stable}"
        )
    preview_count = counters.get("translation_preview", 0)
    if preview_count < min_translation_preview:
        issues.append(
            f"translation_preview count {preview_count} < {min_translation_preview}"
        )
    status_count = counters.get("translation_status", 0)
    if max_translation_status is not None and status_count > max_translation_status:
        issues.append(
            f"translation_status count {status_count} > {max_translation_status}"
        )
    return issues


def _record_event_contract(state: dict[str, Any], event: dict[str, Any]) -> list[str]:
    event_type = str(event.get("type") or "")
    issues: list[str] = []
    if event_type == "transcript_update":
        revision = int(event.get("revision") or 0)
        latest_revision = int(state.get("latest_transcript_revision") or 0)
        if revision <= latest_revision:
            issues.append(
                f"transcript_update revision {revision} is not greater than previous revision {latest_revision}"
            )
        state["latest_transcript_revision"] = max(latest_revision, revision)
        source_ids = state.setdefault("source_stable_segment_ids", [])
        source_segments = state.setdefault("source_stable_segments", [])
        seen_source_ids = state.setdefault("seen_source_stable_segment_ids", set())
        for segment in event.get("stable_appends") or []:
            if not isinstance(segment, dict):
                continue
            segment_id = str(segment.get("id") or "")
            if not segment_id:
                issues.append(
                    "transcript_update stable_appends contains a segment without id"
                )
                continue
            if segment_id in seen_source_ids:
                issues.append(f"duplicate source stable segment id: {segment_id}")
                continue
            source_ids.append(segment_id)
            source_segments.append(dict(segment))
            seen_source_ids.add(segment_id)
    elif event_type == "translation_preview":
        source_revision = int(event.get("source_revision") or 0)
        latest_revision = int(state.get("latest_transcript_revision") or 0)
        if source_revision < latest_revision:
            issues.append(
                "translation_preview source_revision "
                f"{source_revision} is older than latest transcript_update revision {latest_revision}"
            )
    elif event_type == "translation_stable":
        segment_id = str(event.get("source_segment_id") or "")
        covered_ids = _translation_source_segment_ids(event)
        issues.extend(_translation_coverage_issues(state, event, event_type=event_type))
        translated_ids = state.setdefault("translation_stable_segment_ids", [])
        seen_translated_ids = state.setdefault(
            "seen_translation_stable_segment_ids", set()
        )
        if not segment_id:
            issues.append("translation_stable is missing source_segment_id")
        else:
            if segment_id not in covered_ids:
                issues.append(
                    "translation_stable source_segment_ids does not include source_segment_id"
                )
            for covered_id in covered_ids:
                if covered_id in seen_translated_ids:
                    issues.append(
                        f"duplicate translation_stable for source segment: {covered_id}"
                    )
                    continue
                translated_ids.append(covered_id)
                seen_translated_ids.add(covered_id)
    elif (
        event_type == "translation_status" and str(event.get("scope") or "") == "stable"
    ):
        issues.extend(_translation_coverage_issues(state, event, event_type=event_type))
        status_ids = state.setdefault("translation_status_segment_ids", [])
        status_ids.extend(_translation_source_segment_ids(event))
    elif event_type == "transcript_timing_update":
        segment_id = str(event.get("source_segment_id") or "")
        if not segment_id:
            issues.append("transcript_timing_update is missing source_segment_id")
        elif segment_id not in state.setdefault(
            "seen_source_stable_segment_ids", set()
        ):
            issues.append(
                f"transcript_timing_update references unknown source segment: {segment_id}"
            )
        status = str(event.get("timing_status") or "")
        if status not in {"aligned", "failed"}:
            issues.append(
                f"transcript_timing_update has invalid timing_status: {status}"
            )
    return issues


def _translation_source_segment_ids(event: dict[str, Any]) -> list[str]:
    raw_ids = event.get("source_segment_ids")
    if isinstance(raw_ids, list):
        ids = [str(item or "") for item in raw_ids]
        ids = [item for item in ids if item]
        if ids:
            return ids
    segment_id = str(event.get("source_segment_id") or "")
    return [segment_id] if segment_id else []


def _translation_coverage_issues(
    state: dict[str, Any],
    event: dict[str, Any],
    *,
    event_type: str,
) -> list[str]:
    raw_ids = event.get("source_segment_ids")
    raw_indices = event.get("source_segment_indices")
    if not isinstance(raw_ids, list) or not isinstance(raw_indices, list):
        return []

    segment_ids = [str(item or "") for item in raw_ids]
    segment_indices = [_optional_int(item) for item in raw_indices]
    if len(segment_ids) != len(segment_indices):
        return [
            f"{event_type} source_segment_ids/source_segment_indices length mismatch"
        ]

    source_indices_by_id: dict[str, int] = {}
    for segment in state.get("source_stable_segments") or []:
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("id") or "")
        segment_index = _optional_int(segment.get("index"))
        if segment_id and segment_index is not None:
            source_indices_by_id[segment_id] = segment_index

    issues: list[str] = []
    for position, (segment_id, segment_index) in enumerate(
        zip(segment_ids, segment_indices, strict=True)
    ):
        source_index = source_indices_by_id.get(segment_id)
        if (
            source_index is None
            or segment_index is None
            or segment_index == source_index
        ):
            continue
        issues.append(
            f"{event_type} source_segment_indices[{position}] {segment_index} "
            f"does not match source segment {segment_id} index {source_index}"
        )
    return issues


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _final_event_contract_issues(
    state: dict[str, Any],
    final_event: dict[str, Any] | None,
    *,
    expect_translation: bool,
) -> list[str]:
    issues: list[str] = []
    if final_event is None:
        return issues

    source_ids = list(state.get("source_stable_segment_ids") or [])
    if "segments" in final_event:
        final_ids = [
            str(segment.get("id") or "")
            for segment in final_event.get("segments", [])
            if isinstance(segment, dict) and str(segment.get("id") or "")
        ]
        if source_ids and source_ids != final_ids:
            issues.append(
                "transcript_update stable history does not match transcript_final segments"
            )
    else:
        final_ids = source_ids
        final_count = int(final_event.get("stable_count") or 0)
        if final_count != len(source_ids):
            issues.append(
                f"transcript_final stable_count {final_count} does not match replayed stable history {len(source_ids)}"
            )

    if not expect_translation:
        return issues

    translated_ids = set(state.get("translation_stable_segment_ids") or [])
    missing = [
        segment_id for segment_id in final_ids if segment_id not in translated_ids
    ]
    if missing:
        issues.append(
            f"missing translation_stable for source segments: {', '.join(missing)}"
        )

    final_id_set = set(final_ids)
    unknown = [
        segment_id
        for segment_id in state.get("translation_stable_segment_ids") or []
        if segment_id not in final_id_set
    ]
    if unknown:
        issues.append(
            f"translation_stable references unknown source segments: {', '.join(unknown)}"
        )
    status_ids = [
        segment_id
        for segment_id in state.get("translation_status_segment_ids") or []
        if segment_id in final_id_set
    ]
    if status_ids:
        issues.append(
            f"translation_status emitted for stable source segments: {', '.join(status_ids)}"
        )
    return issues


def _compute_reference_cer(
    *,
    reference_srt: str | None,
    final_event: dict[str, Any] | None,
    stable_segments: list[dict[str, Any]] | None = None,
    start_sec: float,
    duration_sec: float,
    strip_ruby: bool,
) -> dict[str, Any] | None:
    if reference_srt is None:
        return None
    from tools.sweep_cer_vs_srt import (
        _cer,
        _normalize_for_cer,
        load_srt,
        srt_text_in_window,
    )

    segments = (final_event or {}).get("segments")
    if not isinstance(segments, list):
        segments = stable_segments or []
    hyp_text = "".join(
        str(segment.get("text") or "")
        for segment in segments
        if isinstance(segment, dict)
    )
    ref_text = srt_text_in_window(
        load_srt(reference_srt, strip_ruby=strip_ruby), start_sec, duration_sec
    )
    return {
        "reference_srt": reference_srt,
        "cer": round(_cer(hyp_text, ref_text), 6),
        "hyp_chars": len(_normalize_for_cer(hyp_text)),
        "ref_chars": len(_normalize_for_cer(ref_text)),
    }


def _normalized_chars(text: str) -> list[str]:
    import unicodedata

    chars: list[str] = []
    for ch in str(text or ""):
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            continue
        chars.append(ch.lower())
    return chars


def _detect_repetition_loop(text: str) -> dict[str, Any] | None:
    normalized = "".join(_normalized_chars(text))
    if len(normalized) < 80:
        return None
    max_unit = min(120, len(normalized) // 4)
    for unit_len in range(8, max_unit + 1):
        required_repeats = max(4, (80 + unit_len - 1) // unit_len)
        limit = len(normalized) - unit_len * required_repeats + 1
        for start in range(0, max(0, limit)):
            unit = normalized[start : start + unit_len]
            repeats = 1
            while (
                start + (repeats + 1) * unit_len <= len(normalized)
                and normalized[
                    start + repeats * unit_len : start + (repeats + 1) * unit_len
                ]
                == unit
            ):
                repeats += 1
            if repeats >= required_repeats:
                return {
                    "start_char": start,
                    "unit_chars": unit_len,
                    "repeat_count": repeats,
                    "repeated_chars": repeats * unit_len,
                    "preview": unit[:40],
                }
    return None


def _repetition_validation_issues(segments: list[dict[str, Any]]) -> list[str]:
    text = "".join(
        str(segment.get("text") or "")
        for segment in segments
        if isinstance(segment, dict)
    )
    loop = _detect_repetition_loop(text)
    if loop is None:
        return []
    return [
        "stable transcript contains a repetition loop: "
        f"{loop['repeat_count']}x {loop['unit_chars']}-char unit at normalized char {loop['start_char']}"
    ]


def _build_reference_char_timeline(
    *,
    reference_srt: str,
    start_sec: float,
    duration_sec: float,
    strip_ruby: bool,
) -> tuple[list[str], list[tuple[float, float]]]:
    from tools.sweep_cer_vs_srt import load_srt

    window_start = float(start_sec)
    window_end = window_start + float(duration_sec)
    chars: list[str] = []
    times: list[tuple[float, float]] = []
    for entry in load_srt(reference_srt, strip_ruby=strip_ruby):
        entry_start = float(entry["start"])
        entry_end = float(entry["end"])
        if entry_end <= window_start or entry_start >= window_end:
            continue
        entry_chars = _normalized_chars(str(entry.get("text") or ""))
        if not entry_chars:
            continue
        clipped_start = max(entry_start, window_start) - window_start
        clipped_end = min(entry_end, window_end) - window_start
        duration = max(0.0, clipped_end - clipped_start)
        for idx, ch in enumerate(entry_chars):
            char_start = clipped_start + duration * idx / len(entry_chars)
            char_end = clipped_start + duration * (idx + 1) / len(entry_chars)
            chars.append(ch)
            times.append((char_start, char_end))
    return chars, times


def _summarize_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": round(float(sum(values) / len(values)), 1),
        "p50": round(float(_percentile(values, 0.50)), 1),
        "p90": round(float(_percentile(values, 0.90)), 1),
        "p95": round(float(_percentile(values, 0.95)), 1),
        "max": round(float(max(values)), 1),
    }


def _compute_timestamp_quality(
    *,
    reference_srt: str | None,
    final_event: dict[str, Any] | None,
    stable_segments: list[dict[str, Any]] | None = None,
    start_sec: float,
    duration_sec: float,
    strip_ruby: bool,
) -> dict[str, Any] | None:
    if reference_srt is None or final_event is None:
        return None

    raw_segments = final_event.get("segments")
    if not isinstance(raw_segments, list):
        raw_segments = stable_segments or []
    segments = [segment for segment in raw_segments if isinstance(segment, dict)]
    if not segments:
        return None

    ref_chars, ref_times = _build_reference_char_timeline(
        reference_srt=reference_srt,
        start_sec=start_sec,
        duration_sec=duration_sec,
        strip_ruby=strip_ruby,
    )
    hyp_chars: list[str] = []
    segment_ranges: list[tuple[dict[str, Any], int, int]] = []
    for segment in segments:
        segment_chars = _normalized_chars(str(segment.get("text") or ""))
        begin = len(hyp_chars)
        hyp_chars.extend(segment_chars)
        segment_ranges.append((segment, begin, len(hyp_chars)))
    if not hyp_chars or not ref_chars:
        return None

    matcher = SequenceMatcher(
        None, "".join(hyp_chars), "".join(ref_chars), autojunk=False
    )
    hyp_to_ref: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            hyp_to_ref[block.a + offset] = block.b + offset

    rows: list[dict[str, Any]] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    boundary_abs_errors: list[float] = []
    matched_segments = 0
    aligned_segments = 0
    failed_segments = 0
    pending_segments = 0
    for segment, begin, end in segment_ranges:
        status = str(segment.get("timing_status") or "")
        if status == "aligned":
            aligned_segments += 1
        elif status == "failed":
            failed_segments += 1
        elif status == "pending":
            pending_segments += 1
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        ref_indexes = [
            hyp_to_ref[idx] for idx in range(begin, end) if idx in hyp_to_ref
        ]
        hyp_len = max(0, end - begin)
        match_ratio = len(ref_indexes) / hyp_len if hyp_len else 0.0
        row: dict[str, Any] = {
            "id": segment.get("id"),
            "text": segment.get("text"),
            "timing_status": status or None,
            "match_ratio": round(match_ratio, 3),
            "hyp_chars": hyp_len,
            "matched_chars": len(ref_indexes),
        }
        if (
            isinstance(start_ms, int)
            and isinstance(end_ms, int)
            and ref_indexes
            and match_ratio >= 0.6
        ):
            ref_start_sec = ref_times[min(ref_indexes)][0]
            ref_end_sec = ref_times[max(ref_indexes)][1]
            start_error = float(start_ms) - ref_start_sec * 1000.0
            end_error = float(end_ms) - ref_end_sec * 1000.0
            row.update(
                {
                    "ref_start_ms": round(ref_start_sec * 1000.0),
                    "ref_end_ms": round(ref_end_sec * 1000.0),
                    "start_error_ms": round(start_error, 1),
                    "end_error_ms": round(end_error, 1),
                    "boundary_abs_error_ms": round(
                        (abs(start_error) + abs(end_error)) / 2.0, 1
                    ),
                }
            )
            matched_segments += 1
            start_errors.append(start_error)
            end_errors.append(end_error)
            boundary_abs_errors.append((abs(start_error) + abs(end_error)) / 2.0)
        rows.append(row)

    return {
        "reference_srt": reference_srt,
        "method": "global normalized text match to SRT cue char timeline",
        "segment_count": len(segments),
        "aligned_segments": aligned_segments,
        "failed_segments": failed_segments,
        "pending_segments": pending_segments,
        "matched_segments": matched_segments,
        "matched_segment_ratio": round(matched_segments / len(segments), 3)
        if segments
        else None,
        "start_error_ms": _summarize_ms(start_errors),
        "end_error_ms": _summarize_ms(end_errors),
        "start_bias_ms": round(float(sum(start_errors) / len(start_errors)), 1)
        if start_errors
        else None,
        "end_bias_ms": round(float(sum(end_errors) / len(end_errors)), 1)
        if end_errors
        else None,
        "boundary_abs_error_ms": _summarize_ms(boundary_abs_errors),
        "segments": rows,
    }


def _read_rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024.0
    return None


def _read_gpu() -> tuple[int | None, int | None, int | None, int | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None, None, None, None
    line = out.splitlines()[0] if out.splitlines() else ""
    values = [item.strip() for item in line.split(",")]
    if len(values) < 4:
        return None, None, None, None
    try:
        return int(values[0]), int(values[1]), int(values[2]), int(values[3])
    except ValueError:
        return None, None, None, None


def _make_sample(
    *,
    start_time: float,
    audio_sent_sec: float,
    pid: int | None,
    counters: dict[str, int],
) -> ResourceSample:
    gpu_used, gpu_total, gpu_util, gpu_temp = _read_gpu()
    return ResourceSample(
        elapsed_sec=round(time.monotonic() - start_time, 3),
        audio_sent_sec=round(audio_sent_sec, 3),
        rss_mb=_read_rss_mb(pid),
        gpu_used_mb=gpu_used,
        gpu_total_mb=gpu_total,
        gpu_util_pct=gpu_util,
        gpu_temp_c=gpu_temp,
        transcript_update_count=counters.get("transcript_update", 0),
        transcript_final_count=counters.get("transcript_final", 0),
    )


async def _recv_events(
    ws: Any,
    *,
    start_time: float,
    audio_sent: list[float],
    counters: dict[str, int],
    last_event_time: list[float],
    final_event: list[dict[str, Any] | None],
    event_log: list[dict[str, Any]],
    event_times: list[dict[str, Any]],
    contract_state: dict[str, Any],
    contract_issues: list[str],
    max_logged_events: int,
) -> None:
    while True:
        msg = await ws.recv()
        last_event_time[0] = time.monotonic()
        event = json.loads(msg)
        event_type = str(event.get("type") or "")
        event_wall_sec = time.monotonic() - start_time
        timing = {
            "type": event_type,
            "wall_sec": round(event_wall_sec, 6),
            "audio_sent_sec": round(audio_sent[0], 6),
        }
        if event_type == "transcript_update":
            timing["stable_segment_ids"] = [
                str(segment.get("id") or "")
                for segment in event.get("stable_appends") or []
                if isinstance(segment, dict) and str(segment.get("id") or "")
            ]
        elif event_type == "transcript_timing_update":
            timing["source_segment_id"] = str(event.get("source_segment_id") or "")
            timing["timing_status"] = str(event.get("timing_status") or "")
        event_times.append(timing)
        counters[event_type] = counters.get(event_type, 0) + 1
        contract_issues.extend(_record_event_contract(contract_state, event))
        if len(event_log) < max_logged_events:
            event_log.append({**timing, "event": event})
        if event_type == "transcript_final":
            final_event[0] = event
            return


async def _monitor_resources(
    *,
    start_time: float,
    pid: int | None,
    counters: dict[str, int],
    audio_sent: list[float],
    last_event_time: list[float],
    samples: list[ResourceSample],
    stop_event: asyncio.Event,
    abort_reason: list[str | None],
    interval_sec: float,
    max_wall_sec: float,
    no_event_timeout_sec: float,
    max_rss_mb: float | None,
    max_gpu_used_mb: int | None,
    max_gpu_temp_c: int | None,
) -> None:
    while not stop_event.is_set():
        sample = _make_sample(
            start_time=start_time,
            audio_sent_sec=audio_sent[0],
            pid=pid,
            counters=counters,
        )
        samples.append(sample)

        if sample.elapsed_sec > max_wall_sec:
            abort_reason[0] = (
                f"wall timeout: {sample.elapsed_sec:.1f}s > {max_wall_sec:.1f}s"
            )
        elif time.monotonic() - last_event_time[0] > no_event_timeout_sec:
            abort_reason[0] = f"no event for {no_event_timeout_sec:.1f}s"
        elif (
            max_rss_mb is not None
            and sample.rss_mb is not None
            and sample.rss_mb > max_rss_mb
        ):
            abort_reason[0] = (
                f"rss exceeded: {sample.rss_mb:.1f}MB > {max_rss_mb:.1f}MB"
            )
        elif (
            max_gpu_used_mb is not None
            and sample.gpu_used_mb is not None
            and sample.gpu_used_mb > max_gpu_used_mb
        ):
            abort_reason[0] = (
                f"whole-gpu memory exceeded: {sample.gpu_used_mb}MB > {max_gpu_used_mb}MB"
            )
        elif (
            max_gpu_temp_c is not None
            and sample.gpu_temp_c is not None
            and sample.gpu_temp_c > max_gpu_temp_c
        ):
            abort_reason[0] = (
                f"gpu temperature exceeded: {sample.gpu_temp_c}C > {max_gpu_temp_c}C"
            )

        if abort_reason[0] is not None:
            stop_event.set()
            return
        await asyncio.sleep(interval_sec)


async def run_check(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.monotonic()
    check_start_gpu_used, _, _, _ = _read_gpu()
    counters: dict[str, int] = {}
    event_log: list[dict[str, Any]] = []
    event_times: list[dict[str, Any]] = []
    contract_state: dict[str, Any] = {}
    contract_issues: list[str] = []
    samples: list[ResourceSample] = []
    final_event: list[dict[str, Any] | None] = [None]
    ready_event: dict[str, Any] | None = None
    abort_reason: list[str | None] = [None]
    last_event_time = [start_time]
    audio_sent = [0.0]
    finish_sent_wall_sec: list[float | None] = [None]
    stop_event = asyncio.Event()

    try:
        with sf.SoundFile(args.audio) as audio_file:
            if audio_file.samplerate != 16000:
                raise ValueError(
                    f"audio sample rate must be 16000 Hz, got {audio_file.samplerate}"
                )
            if audio_file.channels != 1:
                raise ValueError(
                    f"audio must be mono, got {audio_file.channels} channels"
                )
            if args.start_sec > 0:
                audio_file.seek(int(args.start_sec * audio_file.samplerate))

            async with websockets.connect(
                args.url, max_size=None, ping_interval=None
            ) as ws:
                start_command = {
                    "type": "start",
                    "session_id": args.session_id,
                    "sample_rate": 16000,
                    "audio_format": "pcm_s16le",
                    "language": args.language,
                }
                if args.target_language:
                    start_command["target_language"] = args.target_language
                await ws.send(json.dumps(start_command, ensure_ascii=False))
                ready = json.loads(await ws.recv())
                if ready.get("type") != "ready":
                    raise RuntimeError(f"unexpected ready event: {ready}")
                ready_event = ready
                counters["ready"] = 1
                last_event_time[0] = time.monotonic()

                recv_task = asyncio.create_task(
                    _recv_events(
                        ws,
                        start_time=start_time,
                        audio_sent=audio_sent,
                        counters=counters,
                        last_event_time=last_event_time,
                        final_event=final_event,
                        event_log=event_log,
                        event_times=event_times,
                        contract_state=contract_state,
                        contract_issues=contract_issues,
                        max_logged_events=args.max_logged_events,
                    )
                )
                monitor_task = asyncio.create_task(
                    _monitor_resources(
                        start_time=start_time,
                        pid=args.pid,
                        counters=counters,
                        audio_sent=audio_sent,
                        last_event_time=last_event_time,
                        samples=samples,
                        stop_event=stop_event,
                        abort_reason=abort_reason,
                        interval_sec=args.monitor_interval_sec,
                        max_wall_sec=args.max_wall_sec,
                        no_event_timeout_sec=args.no_event_timeout_sec,
                        max_rss_mb=args.max_rss_mb,
                        max_gpu_used_mb=args.max_gpu_used_mb,
                        max_gpu_temp_c=args.max_gpu_temp_c,
                    )
                )

                chunk_frames = max(
                    1, int(round(args.chunk_sec * audio_file.samplerate))
                )
                target_frames = int(round(args.max_audio_sec * audio_file.samplerate))
                sent_frames = 0
                try:
                    while sent_frames < target_frames and not stop_event.is_set():
                        frames = min(chunk_frames, target_frames - sent_frames)
                        data = audio_file.read(frames, dtype="int16", always_2d=False)
                        if data.size == 0:
                            break
                        if data.ndim != 1:
                            data = data[:, 0]
                        payload = np.asarray(data, dtype="<i2").tobytes()
                        await ws.send(payload)
                        sent_frames += int(data.shape[0])
                        audio_sent[0] = sent_frames / audio_file.samplerate
                        if args.send_delay_sec > 0:
                            await asyncio.sleep(args.send_delay_sec)

                    if abort_reason[0] is None:
                        finish_sent_wall_sec[0] = time.monotonic() - start_time
                        await ws.send(json.dumps({"type": "finish"}))
                        try:
                            await asyncio.wait_for(
                                recv_task, timeout=args.finish_timeout_sec
                            )
                        except asyncio.TimeoutError:
                            abort_reason[0] = (
                                f"finish timeout after {args.finish_timeout_sec:.1f}s"
                            )
                finally:
                    stop_event.set()
                    monitor_task.cancel()
                    if not recv_task.done():
                        recv_task.cancel()
                    await asyncio.gather(
                        monitor_task, recv_task, return_exceptions=True
                    )
    except (ConnectionClosed, OSError, RuntimeError, ValueError) as exc:
        if abort_reason[0] is None:
            abort_reason[0] = f"{type(exc).__name__}: {exc}"

    end_sample = _make_sample(
        start_time=start_time,
        audio_sent_sec=audio_sent[0],
        pid=args.pid,
        counters=counters,
    )
    samples.append(end_sample)

    elapsed_sec = round(time.monotonic() - start_time, 3)
    audio_sent_sec = round(audio_sent[0], 3)
    gpu_values = [
        sample.gpu_used_mb for sample in samples if sample.gpu_used_mb is not None
    ]
    max_gpu_used_mb_seen = max(gpu_values, default=0)
    max_gpu_delta_from_check_start_mb_seen = (
        None
        if check_start_gpu_used is None or not gpu_values
        else max(0, max_gpu_used_mb_seen - check_start_gpu_used)
    )
    translation_issues = _translation_validation_issues(
        ready_event=ready_event,
        counters=counters,
        expect_translation=args.expect_translation,
        min_translation_stable=args.min_translation_stable,
        min_translation_preview=args.min_translation_preview,
        max_translation_status=args.max_translation_status,
    )
    event_stream_issues = contract_issues + _final_event_contract_issues(
        contract_state,
        final_event[0],
        expect_translation=args.expect_translation,
    )
    stable_segments = list(contract_state.get("source_stable_segments") or [])
    repetition_issues = _repetition_validation_issues(stable_segments)
    if abort_reason[0] is None and (
        translation_issues or event_stream_issues or repetition_issues
    ):
        abort_reason[0] = "; ".join(
            translation_issues + event_stream_issues + repetition_issues
        )

    summary = {
        "ok": abort_reason[0] is None and final_event[0] is not None,
        "abort_reason": abort_reason[0],
        "url": args.url,
        "pid": args.pid,
        "audio": args.audio,
        "start_sec": args.start_sec,
        "max_audio_sec": args.max_audio_sec,
        "elapsed_sec": elapsed_sec,
        "audio_sent_sec": audio_sent_sec,
        "counters": counters,
        "ready_event": ready_event,
        "translation_validation_issues": translation_issues,
        "event_stream_validation_issues": event_stream_issues,
        "repetition_validation_issues": repetition_issues,
        "segment_count": len(contract_state.get("source_stable_segments") or []),
        "timing": _summarize_event_timings(
            event_times,
            finish_sent_wall_sec=finish_sent_wall_sec[0],
            elapsed_sec=elapsed_sec,
            audio_sent_sec=audio_sent_sec,
        ),
        "cer": _compute_reference_cer(
            reference_srt=args.reference_srt,
            final_event=final_event[0],
            stable_segments=stable_segments,
            start_sec=args.start_sec,
            duration_sec=args.max_audio_sec,
            strip_ruby=args.strip_ruby,
        ),
        "timestamp_quality": _compute_timestamp_quality(
            reference_srt=args.reference_srt,
            final_event=final_event[0],
            stable_segments=stable_segments,
            start_sec=args.start_sec,
            duration_sec=args.max_audio_sec,
            strip_ruby=args.strip_ruby,
        ),
        "first_sample": asdict(samples[0]) if samples else None,
        "last_sample": asdict(samples[-1]) if samples else None,
        "max_rss_mb_seen": max((s.rss_mb or 0.0 for s in samples), default=0.0),
        "gpu_memory_scope": "whole_gpu_not_process",
        "gpu_memory_note": (
            "gpu_used_mb comes from nvidia-smi memory.used and includes display/driver/other processes."
        ),
        "check_start_gpu_used_mb": check_start_gpu_used,
        "max_gpu_used_mb_seen": max_gpu_used_mb_seen,
        "max_gpu_delta_from_check_start_mb_seen": max_gpu_delta_from_check_start_mb_seen,
        "samples": [asdict(sample) for sample in samples],
        "event_times": event_times,
        "final_event": final_event[0],
        "events": event_log,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protected realtime WebSocket E2E leak check."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/asr")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference-srt", default=None)
    parser.add_argument("--strip-ruby", action="store_true")
    parser.add_argument(
        "--pid", type=int, default=None, help="Service PID to monitor via /proc."
    )
    parser.add_argument("--session-id", default="leak-check")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--target-language", default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-audio-sec", type=float, default=600.0)
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    parser.add_argument("--send-delay-sec", type=float, default=0.02)
    parser.add_argument("--monitor-interval-sec", type=float, default=5.0)
    parser.add_argument("--max-wall-sec", type=float, default=900.0)
    parser.add_argument("--finish-timeout-sec", type=float, default=180.0)
    parser.add_argument("--no-event-timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-rss-mb", type=float, default=12000.0)
    parser.add_argument(
        "--max-gpu-used-mb",
        type=int,
        default=23000,
        help="Whole-GPU memory.used guard in MiB.",
    )
    parser.add_argument("--max-gpu-temp-c", type=int, default=86)
    parser.add_argument("--max-logged-events", type=int, default=20)
    parser.add_argument("--expect-translation", action="store_true")
    parser.add_argument("--min-translation-stable", type=int, default=0)
    parser.add_argument("--min-translation-preview", type=int, default=0)
    parser.add_argument("--max-translation-status", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    summary = asyncio.run(run_check(parse_args()))
    omitted = {"samples", "events", "event_times"}
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in omitted},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
