# coding=utf-8
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
        return [float(item["wall_sec"]) for item in event_times if item.get("type") == event_type]

    update_times = times_for("transcript_update")
    final_times = times_for("transcript_final")
    update_gaps = [right - left for left, right in zip(update_times, update_times[1:])]

    final_wall = final_times[0] if final_times else None
    return {
        "first_transcript_update_wall_sec": round(update_times[0], 3) if update_times else None,
        "final_wall_sec": round(final_wall, 3) if final_wall is not None else None,
        "finish_to_final_sec": (
            round(final_wall - finish_sent_wall_sec, 3)
            if final_wall is not None and finish_sent_wall_sec is not None
            else None
        ),
        "update_gap_p95_sec": (
            round(_percentile(update_gaps, 0.95), 3) if _percentile(update_gaps, 0.95) is not None else None
        ),
        "processing_speed_x": round(audio_sent_sec / elapsed_sec, 3) if elapsed_sec > 0 else None,
    }


def _compute_reference_cer(
    *,
    reference_srt: str | None,
    final_event: dict[str, Any] | None,
    start_sec: float,
    duration_sec: float,
    strip_ruby: bool,
) -> dict[str, Any] | None:
    if reference_srt is None:
        return None
    from tools.sweep_cer_vs_srt import _cer, _normalize_for_cer, load_srt, srt_text_in_window

    hyp_text = "".join(str(segment.get("text") or "") for segment in (final_event or {}).get("segments", []))
    ref_text = srt_text_in_window(load_srt(reference_srt, strip_ruby=strip_ruby), start_sec, duration_sec)
    return {
        "reference_srt": reference_srt,
        "cer": round(_cer(hyp_text, ref_text), 6),
        "hyp_chars": len(_normalize_for_cer(hyp_text)),
        "ref_chars": len(_normalize_for_cer(ref_text)),
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
        event_times.append(timing)
        counters[event_type] = counters.get(event_type, 0) + 1
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
            abort_reason[0] = f"wall timeout: {sample.elapsed_sec:.1f}s > {max_wall_sec:.1f}s"
        elif time.monotonic() - last_event_time[0] > no_event_timeout_sec:
            abort_reason[0] = f"no event for {no_event_timeout_sec:.1f}s"
        elif max_rss_mb is not None and sample.rss_mb is not None and sample.rss_mb > max_rss_mb:
            abort_reason[0] = f"rss exceeded: {sample.rss_mb:.1f}MB > {max_rss_mb:.1f}MB"
        elif (
            max_gpu_used_mb is not None
            and sample.gpu_used_mb is not None
            and sample.gpu_used_mb > max_gpu_used_mb
        ):
            abort_reason[0] = f"whole-gpu memory exceeded: {sample.gpu_used_mb}MB > {max_gpu_used_mb}MB"
        elif max_gpu_temp_c is not None and sample.gpu_temp_c is not None and sample.gpu_temp_c > max_gpu_temp_c:
            abort_reason[0] = f"gpu temperature exceeded: {sample.gpu_temp_c}C > {max_gpu_temp_c}C"

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
    samples: list[ResourceSample] = []
    final_event: list[dict[str, Any] | None] = [None]
    abort_reason: list[str | None] = [None]
    last_event_time = [start_time]
    audio_sent = [0.0]
    finish_sent_wall_sec: list[float | None] = [None]
    stop_event = asyncio.Event()

    try:
        with sf.SoundFile(args.audio) as audio_file:
            if audio_file.samplerate != 16000:
                raise ValueError(f"audio sample rate must be 16000 Hz, got {audio_file.samplerate}")
            if audio_file.channels != 1:
                raise ValueError(f"audio must be mono, got {audio_file.channels} channels")
            if args.start_sec > 0:
                audio_file.seek(int(args.start_sec * audio_file.samplerate))

            async with websockets.connect(args.url, max_size=None, ping_interval=None) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "start",
                            "session_id": args.session_id,
                            "sample_rate": 16000,
                            "audio_format": "pcm_s16le",
                            "language": args.language,
                        },
                        ensure_ascii=False,
                    )
                )
                ready = json.loads(await ws.recv())
                if ready.get("type") != "ready":
                    raise RuntimeError(f"unexpected ready event: {ready}")
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

                chunk_frames = max(1, int(round(args.chunk_sec * audio_file.samplerate)))
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
                            await asyncio.wait_for(recv_task, timeout=args.finish_timeout_sec)
                        except asyncio.TimeoutError:
                            abort_reason[0] = f"finish timeout after {args.finish_timeout_sec:.1f}s"
                finally:
                    stop_event.set()
                    monitor_task.cancel()
                    if not recv_task.done():
                        recv_task.cancel()
                    await asyncio.gather(monitor_task, recv_task, return_exceptions=True)
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
    gpu_values = [sample.gpu_used_mb for sample in samples if sample.gpu_used_mb is not None]
    max_gpu_used_mb_seen = max(gpu_values, default=0)
    max_gpu_delta_from_check_start_mb_seen = (
        None if check_start_gpu_used is None or not gpu_values else max(0, max_gpu_used_mb_seen - check_start_gpu_used)
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
        "segment_count": len((final_event[0] or {}).get("segments", [])),
        "timing": _summarize_event_timings(
            event_times,
            finish_sent_wall_sec=finish_sent_wall_sec[0],
            elapsed_sec=elapsed_sec,
            audio_sent_sec=audio_sent_sec,
        ),
        "cer": _compute_reference_cer(
            reference_srt=args.reference_srt,
            final_event=final_event[0],
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
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protected realtime WebSocket E2E leak check.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/asr")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference-srt", default=None)
    parser.add_argument("--strip-ruby", action="store_true")
    parser.add_argument("--pid", type=int, default=None, help="Service PID to monitor via /proc.")
    parser.add_argument("--session-id", default="leak-check")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-audio-sec", type=float, default=600.0)
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    parser.add_argument("--send-delay-sec", type=float, default=0.02)
    parser.add_argument("--monitor-interval-sec", type=float, default=5.0)
    parser.add_argument("--max-wall-sec", type=float, default=900.0)
    parser.add_argument("--finish-timeout-sec", type=float, default=180.0)
    parser.add_argument("--no-event-timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-rss-mb", type=float, default=12000.0)
    parser.add_argument("--max-gpu-used-mb", type=int, default=23000, help="Whole-GPU memory.used guard in MiB.")
    parser.add_argument("--max-gpu-temp-c", type=int, default=86)
    parser.add_argument("--max-logged-events", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    summary = asyncio.run(run_check(parse_args()))
    omitted = {"samples", "events", "event_times"}
    print(json.dumps({k: v for k, v in summary.items() if k not in omitted}, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
