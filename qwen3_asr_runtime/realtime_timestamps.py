# coding=utf-8
from __future__ import annotations

import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Deque

import numpy as np

from .transcript_store import TranscriptStore
from .language_support import QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES
from .utils import SAMPLE_RATE
from .vad import normalize_pcm

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeTimestampConfig:
    pad_ms: int = 500
    finish_timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        if int(self.pad_ms) < 0:
            raise ValueError("pad_ms must be >= 0")
        if int(self.finish_timeout_ms) <= 0:
            raise ValueError("finish_timeout_ms must be > 0")


@dataclass(frozen=True)
class StableTimingJob:
    source_segment_id: str
    source_text: str
    source_language: str
    start_sample: int
    end_sample: int


class AudioTimelineBuffer:
    """Append-only absolute-sample audio buffer for realtime alignment crops."""

    def __init__(self) -> None:
        self.start_sample = 0
        self._chunks: Deque[np.ndarray] = deque()
        self._sample_count = 0

    @property
    def end_sample(self) -> int:
        return int(self.start_sample) + int(self._sample_count)

    def append(self, audio: np.ndarray) -> None:
        chunk = normalize_pcm(audio)
        if chunk.shape[0] == 0:
            return
        chunk = chunk.astype(np.float32, copy=True)
        self._chunks.append(chunk)
        self._sample_count += int(chunk.shape[0])

    def crop(self, *, start_sample: int, end_sample: int, pad_samples: int = 0) -> tuple[np.ndarray, int]:
        crop_start = max(int(self.start_sample), int(start_sample) - max(0, int(pad_samples)))
        crop_end = min(int(self.end_sample), int(end_sample) + max(0, int(pad_samples)))
        if crop_end <= crop_start:
            return np.zeros((0,), dtype=np.float32), crop_start
        return self._copy_range(crop_start, crop_end), crop_start

    def _copy_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        output = np.empty((int(end_sample) - int(start_sample),), dtype=np.float32)
        chunk_start = int(self.start_sample)
        write_pos = 0
        for chunk in self._chunks:
            chunk_end = chunk_start + int(chunk.shape[0])
            if chunk_end <= start_sample:
                chunk_start = chunk_end
                continue
            if chunk_start >= end_sample:
                break

            read_start = max(int(start_sample), chunk_start) - chunk_start
            read_end = min(int(end_sample), chunk_end) - chunk_start
            part = chunk[read_start:read_end]
            output[write_pos : write_pos + part.shape[0]] = part
            write_pos += int(part.shape[0])
            chunk_start = chunk_end

        if write_pos != output.shape[0]:
            return output[:write_pos].copy()
        return output


class TimestampModelActor:
    """Owns all calls into one forced-aligner model instance."""

    def __init__(self, aligner: Any, *, executor: ThreadPoolExecutor | None = None) -> None:
        self.aligner = aligner
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen3-aligner",
        )

    @property
    def model_path(self) -> str:
        direct = str(getattr(self.aligner, "model_path", "") or "")
        if direct:
            return direct
        model = getattr(self.aligner, "model", None)
        return str(getattr(model, "name_or_path", "") or "")

    def warmup(self, audio: np.ndarray, *, text: str, language: str) -> None:
        audio = normalize_pcm(audio)
        future = self._executor.submit(self.aligner.align, audio=(audio, SAMPLE_RATE), text=text, language=language)
        future.result()

    async def align_segment(
        self,
        audio: np.ndarray,
        *,
        text: str,
        language: str,
        timeout_sec: float | None,
    ) -> tuple[float | None, float | None, str | None]:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            partial(self._call_align, audio, text=text, language=language),
        )
        future.add_done_callback(_consume_future)
        try:
            result = await _wait_future_result(
                future,
                timeout_sec=None if timeout_sec is None else max(0.001, float(timeout_sec)),
            )
        except asyncio.TimeoutError:
            future.cancel()
            return None, None, "timeout"
        except Exception:
            _LOGGER.debug("Realtime forced alignment failed.", exc_info=True)
            return None, None, "failed"

        items = list(getattr(result, "items", []) or [])
        if not items:
            return None, None, "failed"
        raw_start_sec = float(getattr(items[0], "start_time", 0.0))
        raw_end_sec = float(getattr(items[-1], "end_time", raw_start_sec))
        audio_duration_sec = max(0.0, float(audio.shape[0]) / SAMPLE_RATE)
        if not np.isfinite(raw_start_sec) or not np.isfinite(raw_end_sec):
            return None, None, "failed"
        if raw_end_sec < raw_start_sec:
            return None, None, "failed"
        start_sec = max(0.0, raw_start_sec)
        end_sec = max(start_sec, raw_end_sec)
        if audio_duration_sec <= 0.0 or start_sec >= audio_duration_sec:
            return None, None, "failed"
        end_sec = min(end_sec, audio_duration_sec)
        start_sec = min(start_sec, end_sec)
        return start_sec, end_sec, None

    def close(self, *, wait: bool = False) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def _call_align(self, audio: np.ndarray, *, text: str, language: str) -> Any:
        results = self.aligner.align(audio=(audio, SAMPLE_RATE), text=text, language=language)
        if not results:
            raise RuntimeError("forced aligner returned no result")
        return results[0]


class RealtimeTimestampRuntime:
    """Async service-layer timestamp runtime for stable transcript segments."""

    def __init__(
        self,
        model_actor: TimestampModelActor,
        *,
        store: TranscriptStore,
        audio_buffer: AudioTimelineBuffer,
        config: RealtimeTimestampConfig,
        event_queue: asyncio.Queue[dict[str, Any] | None],
        store_lock: asyncio.Lock | None = None,
    ) -> None:
        self.model_actor = model_actor
        self.store = store
        self.audio_buffer = audio_buffer
        self.config = config
        self.event_queue = event_queue
        self.store_lock = store_lock

        self._stable_queue: Deque[StableTimingJob] = deque()
        self._finish_mode = False
        self._closed = False
        self._finish_events: list[dict[str, Any]] = []

        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None

    def ready_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": self.model_actor.model_path,
            "source": "forced_aligner",
            "allowed_source_languages": list(QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES),
            "stable": {
                "initial_status": "pending",
                "patch_event": "transcript_timing_update",
                "finish_timeout_ms": int(self.config.finish_timeout_ms),
                "pad_ms": int(self.config.pad_ms),
            },
        }

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._stable_queue.clear()
            self._wake.set()
        await self._stop_worker()

    def accept_audio(self, audio: np.ndarray) -> None:
        self.audio_buffer.append(audio)

    async def accept_jobs(self, jobs: list[StableTimingJob]) -> None:
        if not jobs:
            return
        async with self._lock:
            if self._finish_mode or self._closed:
                return
            self._stable_queue.extend(jobs)
            self._wake.set()

    async def finish(self, jobs: list[StableTimingJob]) -> list[dict[str, Any]]:
        deadline = asyncio.get_running_loop().time() + self._finish_timeout_sec()
        async with self._lock:
            self._finish_mode = True
            queued_jobs = list(self._stable_queue)
            self._stable_queue.clear()
            self._wake.set()
        await self._stop_worker()

        async with self._lock:
            events = list(self._finish_events)
            self._finish_events.clear()
        for job in queued_jobs + list(jobs):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                events.append(await self._timing_event(job, start_ms=None, end_ms=None, timing_status="failed"))
                continue
            events.append(await self._run_job(job, publish=False, timeout_sec=remaining))
        return events

    async def _worker_loop(self) -> None:
        while True:
            await self._wake.wait()
            while True:
                job: StableTimingJob | None = None
                async with self._lock:
                    self._wake.clear()
                    if self._closed or self._finish_mode:
                        return
                    if self._stable_queue:
                        job = self._stable_queue.popleft()
                if job is None:
                    break
                await self._run_job(job, publish=True, timeout_sec=self._finish_timeout_sec())

    async def _run_job(
        self,
        job: StableTimingJob,
        *,
        publish: bool,
        timeout_sec: float,
    ) -> dict[str, Any]:
        if job.source_language not in QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES:
            event = await self._timing_event(job, start_ms=None, end_ms=None, timing_status="failed")
            if publish:
                await self.event_queue.put(event)
            return event

        audio, crop_start_sample = self.audio_buffer.crop(
            start_sample=job.start_sample,
            end_sample=job.end_sample,
            pad_samples=self._pad_samples(),
        )
        if audio.shape[0] == 0:
            event = await self._timing_event(job, start_ms=None, end_ms=None, timing_status="failed")
        else:
            start_sec, end_sec, _error_code = await self.model_actor.align_segment(
                audio,
                text=job.source_text,
                language=job.source_language,
                timeout_sec=timeout_sec,
            )
            if start_sec is None or end_sec is None:
                event = await self._timing_event(job, start_ms=None, end_ms=None, timing_status="failed")
            else:
                crop_start_ms = int(round(1000 * int(crop_start_sample) / SAMPLE_RATE))
                start_ms = crop_start_ms + int(round(start_sec * 1000))
                end_ms = crop_start_ms + int(round(end_sec * 1000))
                event = await self._timing_event(job, start_ms=start_ms, end_ms=end_ms, timing_status="aligned")

        async with self._lock:
            should_publish = publish and not self._closed and not self._finish_mode
            if publish and not should_publish and self._finish_mode and not self._closed:
                self._finish_events.append(event)
        if should_publish:
            await self.event_queue.put(event)
        return event

    async def _timing_event(
        self,
        job: StableTimingJob,
        *,
        start_ms: int | None,
        end_ms: int | None,
        timing_status: str,
    ) -> dict[str, Any]:
        if self.store_lock is None:
            return self.store.update_segment_timing(
                source_segment_id=job.source_segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                timing_status=timing_status,
            )
        async with self.store_lock:
            return self.store.update_segment_timing(
                source_segment_id=job.source_segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                timing_status=timing_status,
            )

    async def _stop_worker(self) -> None:
        task = self._worker_task
        if task is None:
            return
        self._wake.set()
        await task
        self._worker_task = None

    def _pad_samples(self) -> int:
        return int(round(SAMPLE_RATE * int(self.config.pad_ms) / 1000))

    def _finish_timeout_sec(self) -> float:
        return max(0.001, int(self.config.finish_timeout_ms) / 1000.0)


def _consume_future(future: Any) -> None:
    try:
        future.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _wait_future_result(future: asyncio.Future[Any], *, timeout_sec: float | None) -> Any:
    loop = asyncio.get_running_loop()
    deadline = None if timeout_sec is None else loop.time() + float(timeout_sec)
    while not future.done():
        if deadline is None:
            await asyncio.sleep(0.01)
            continue
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        await asyncio.sleep(min(0.01, remaining))
    return future.result()


__all__ = [
    "AudioTimelineBuffer",
    "RealtimeTimestampConfig",
    "RealtimeTimestampRuntime",
    "StableTimingJob",
    "TimestampModelActor",
]
