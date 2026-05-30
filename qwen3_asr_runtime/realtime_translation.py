# coding=utf-8
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Deque

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeTranslationConfig:
    target_language: str
    source_language: str = ""
    stable_enabled: bool = True
    preview_enabled: bool = True
    preview_debounce_ms: int = 700
    preview_timeout_ms: int = 30_000
    max_new_tokens: int | None = None
    stable_batch_size: int = 1

    def __post_init__(self) -> None:
        if not str(self.target_language or "").strip():
            raise ValueError("target_language must not be empty")
        if int(self.preview_debounce_ms) < 0:
            raise ValueError("preview_debounce_ms must be >= 0")
        if int(self.preview_timeout_ms) <= 0:
            raise ValueError("preview_timeout_ms must be > 0")
        if int(self.stable_batch_size) <= 0:
            raise ValueError("stable_batch_size must be > 0")


@dataclass(frozen=True)
class _StableJob:
    source_revision: int
    source_segment_id: str
    source_segment_index: int
    source_text: str
    source_language: str
    target_language: str


@dataclass(frozen=True)
class _PreviewJob:
    generation: int
    source_revision: int
    source_text: str
    source_language: str
    target_language: str


class TranslationModelActor:
    """Owns all calls into one translation model instance."""

    def __init__(
        self,
        translator: Any,
        *,
        capture_lock: threading.Lock | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.translator = translator
        self._capture_lock = capture_lock
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hymt-translation",
        )

    @property
    def model_path(self) -> str:
        return str(getattr(self.translator, "model_path", ""))

    def warmup(
        self,
        texts: list[str] | tuple[str, ...],
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
        sync_cuda: bool,
        batch_size: int = 1,
    ) -> list[Any]:
        warmup = getattr(self.translator, "warmup", None)
        if warmup is None:
            raise RuntimeError("translation prewarm was requested but this translator does not support warmup")
        future = self._executor.submit(
            self._call_with_capture_lock,
            partial(
                warmup,
                texts,
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
                sync_cuda=sync_cuda,
                batch_size=batch_size,
            ),
        )
        return list(future.result())

    async def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
        timeout_sec: float | None,
    ) -> tuple[str | None, str | None]:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            partial(
                self._call_translate,
                text,
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
            ),
        )
        future.add_done_callback(_consume_future)
        try:
            translated = str(
                await _wait_future_result(
                    future,
                    timeout_sec=None if timeout_sec is None else max(0.001, float(timeout_sec)),
                )
            ).strip()
            if not translated:
                return None, "failed"
            return translated, None
        except asyncio.TimeoutError:
            return None, "timeout"
        except Exception:
            _LOGGER.debug("Realtime translation failed.", exc_info=True)
            return None, "failed"

    async def translate_batch(
        self,
        texts: list[str],
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
        timeout_sec: float | None,
    ) -> list[tuple[str | None, str | None]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            partial(
                self._call_translate_batch,
                list(texts),
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
            ),
        )
        future.add_done_callback(_consume_future)
        try:
            raw_outputs = await _wait_future_result(
                future,
                timeout_sec=None if timeout_sec is None else max(0.001, float(timeout_sec)),
            )
            outputs = [str(output).strip() for output in raw_outputs]
            if len(outputs) != len(texts):
                raise RuntimeError("translation batch output length mismatch")
            return [(output, None) if output else (None, "failed") for output in outputs]
        except asyncio.TimeoutError:
            return [(None, "timeout") for _ in texts]
        except Exception:
            _LOGGER.debug("Realtime translation batch failed.", exc_info=True)
            return [(None, "failed") for _ in texts]

    def close(self, *, wait: bool = False) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def _call_translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
    ) -> str:
        return str(
            self._call_with_capture_lock(
                partial(
                    self.translator.translate,
                    text,
                    target_language=target_language,
                    source_language=source_language,
                    max_new_tokens=max_new_tokens,
                )
            )
        )

    def _call_translate_batch(
        self,
        texts: list[str],
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
    ) -> list[str]:
        translate_batch = getattr(self.translator, "translate_batch", None)
        if translate_batch is not None:
            result = self._call_with_capture_lock(
                partial(
                    translate_batch,
                    texts,
                    target_language=target_language,
                    source_language=source_language,
                    max_new_tokens=max_new_tokens,
                )
            )
            return [str(item) for item in result]
        return [
            self._call_translate(
                text,
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
            )
            for text in texts
        ]

    def _call_with_capture_lock(self, call: Any) -> Any:
        if self._capture_lock is None:
            return call()
        with self._capture_lock:
            return call()


class RealtimeTranslationRuntime:
    """Async service-layer translation runtime for realtime transcript events."""

    def __init__(
        self,
        model_actor: TranslationModelActor,
        *,
        config: RealtimeTranslationConfig,
        event_queue: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        self.model_actor = model_actor
        self.config = config
        self.event_queue = event_queue
        self._target_language: str | None = str(config.target_language).strip()

        self._stable_queue: Deque[_StableJob] = deque()
        self._preview_slot: _PreviewJob | None = None
        self._preview_generation = 0
        self._running_job_kind: str | None = None

        self._finish_mode = False
        self._closed = False

        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None

    def ready_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "target_language": self.target_language,
            "model": self.model_actor.model_path,
            "stable": {
                "enabled": bool(self.config.stable_enabled),
                "reliable": True,
                "queue_size": None,
                "timeout_ms": None,
                "batch_size": int(self.config.stable_batch_size),
            },
            "preview": {
                "enabled": bool(self.config.preview_enabled),
                "debounce_ms": int(self.config.preview_debounce_ms),
                "timeout_ms": int(self.config.preview_timeout_ms),
            },
        }

    @property
    def target_language(self) -> str | None:
        return self._target_language

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._stable_queue.clear()
            self._preview_slot = None
            self._wake.set()
        await self._stop_worker(cancel_running_preview=True)

    async def set_target_language(self, target_language: str | None) -> None:
        normalized = str(target_language).strip() if target_language is not None else ""
        async with self._lock:
            self._target_language = normalized or None
            self._preview_generation += 1
            self._preview_slot = None
            self._stable_queue.clear()
            self._wake.set()

    async def accept_source_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "transcript_update":
            return

        target_language = self.target_language
        if not target_language:
            await self._cancel_preview()
            return

        revision = int(event.get("revision") or 0)
        partial = event.get("partial")
        if self.config.preview_enabled:
            if isinstance(partial, dict) and str(partial.get("text") or "").strip():
                await self._update_preview(revision, partial, target_language=target_language)
            else:
                await self._cancel_preview()

        if self.config.stable_enabled:
            for segment in event.get("stable_appends") or []:
                if isinstance(segment, dict):
                    await self._enqueue_stable(revision, segment, target_language=target_language)

    async def cancel_preview(self) -> None:
        await self._cancel_preview()

    async def finish(self, transcript_updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        async with self._lock:
            self._finish_mode = True
            self._preview_generation += 1
            self._preview_slot = None
            queued_jobs = list(self._stable_queue)
            self._stable_queue.clear()
            self._wake.set()
        await self._stop_worker(cancel_running_preview=True)

        events: list[dict[str, Any]] = []

        target_language = self.target_language
        finish_jobs: list[_StableJob] = []
        for event in transcript_updates:
            revision = int(event.get("revision") or 0)
            for segment in event.get("stable_appends") or []:
                if isinstance(segment, dict):
                    job = self._make_stable_job(revision, segment, target_language=target_language)
                    if job is not None:
                        finish_jobs.append(job)

        pending_jobs = finish_jobs + queued_jobs
        for jobs in self._stable_batches(pending_jobs):
            events.extend(await self._run_stable_jobs(jobs, publish=False))
        return events

    async def _enqueue_stable(self, revision: int, segment: dict[str, Any], *, target_language: str) -> None:
        job = self._make_stable_job(revision, segment, target_language=target_language)
        if job is None:
            return

        async with self._lock:
            if self._finish_mode or self._closed:
                return
            self._stable_queue.append(job)
            self._wake.set()

    async def _update_preview(
        self,
        revision: int,
        partial_segment: dict[str, Any],
        *,
        target_language: str,
    ) -> None:
        source_text = str(partial_segment.get("text") or "").strip()
        if not source_text:
            await self._cancel_preview()
            return
        async with self._lock:
            if self._finish_mode or self._closed:
                return
            self._preview_generation += 1
            self._preview_slot = _PreviewJob(
                generation=self._preview_generation,
                source_revision=int(revision),
                source_text=source_text,
                source_language=str(partial_segment.get("language") or self.config.source_language or ""),
                target_language=target_language,
            )
            self._wake.set()

    async def _cancel_preview(self) -> None:
        async with self._lock:
            self._preview_generation += 1
            self._preview_slot = None
            self._wake.set()

    async def _worker_loop(self) -> None:
        while True:
            await self._wake.wait()
            while True:
                job: _PreviewJob | list[_StableJob] | None = None
                async with self._lock:
                    self._wake.clear()
                    if self._closed:
                        return
                    if self._finish_mode:
                        return
                    if self._preview_slot is not None:
                        job = self._preview_slot
                        self._preview_slot = None
                    elif self._stable_queue:
                        job = self._take_stable_batch()
                if job is None:
                    break
                kind = "preview" if isinstance(job, _PreviewJob) else "stable"
                async with self._lock:
                    self._running_job_kind = kind
                if isinstance(job, _PreviewJob):
                    try:
                        await self._run_preview_job(job)
                    finally:
                        await self._clear_running_job(kind)
                else:
                    try:
                        await self._run_stable_jobs(job, publish=True)
                    finally:
                        await self._clear_running_job(kind)

    async def _run_preview_job(self, job: _PreviewJob) -> None:
        if int(self.config.preview_debounce_ms) > 0:
            await asyncio.sleep(int(self.config.preview_debounce_ms) / 1000.0)
        async with self._lock:
            if self._closed or self._finish_mode:
                return
            if self._preview_slot is not None:
                job = self._preview_slot
                self._preview_slot = None
            elif job.generation != self._preview_generation:
                return

        translated, _error_code = await self._translate_text(
            job.source_text,
            source_language=job.source_language,
            target_language=job.target_language,
            timeout_sec=self._preview_timeout_sec(),
        )
        async with self._lock:
            is_current = (
                not self._closed
                and not self._finish_mode
                and job.generation == self._preview_generation
                and self._preview_slot is None
            )
        if not is_current or translated is None:
            return
        await self.event_queue.put(
            {
                "type": "translation_preview",
                "source_revision": int(job.source_revision),
                "target_language": job.target_language,
                "text": translated,
            }
        )

    async def _run_stable_jobs(self, jobs: list[_StableJob], *, publish: bool) -> list[dict[str, Any]]:
        if not jobs:
            return []
        translations = await self._translate_texts(
            [job.source_text for job in jobs],
            source_language=jobs[0].source_language,
            target_language=jobs[0].target_language,
            timeout_sec=None,
        )
        events = [
            self._stable_event(job, translated=translated, error_code=error_code)
            for job, (translated, error_code) in zip(jobs, translations)
        ]

        async with self._lock:
            should_publish = publish and not self._closed
            if should_publish:
                self._wake.set()

        if should_publish:
            for event in events:
                await self.event_queue.put(event)
        return events

    def _stable_event(
        self,
        job: _StableJob,
        *,
        translated: str | None,
        error_code: str | None,
    ) -> dict[str, Any]:
        event: dict[str, Any]
        if translated is None:
            code = error_code or "failed"
            message = "translation failed"
            event = self._stable_status(job, code, message)
        else:
            event = {
                "type": "translation_stable",
                "source_revision": int(job.source_revision),
                "source_segment_id": job.source_segment_id,
                "source_segment_index": int(job.source_segment_index),
                "target_language": job.target_language,
                "text": translated,
            }
        return event

    async def _translate_text(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        timeout_sec: float | None,
    ) -> tuple[str | None, str | None]:
        return await self.model_actor.translate(
            text,
            target_language=target_language,
            source_language=source_language,
            max_new_tokens=self.config.max_new_tokens,
            timeout_sec=timeout_sec,
        )

    async def _translate_texts(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        timeout_sec: float | None,
    ) -> list[tuple[str | None, str | None]]:
        if len(texts) == 1:
            return [
                await self._translate_text(
                    texts[0],
                    source_language=source_language,
                    target_language=target_language,
                    timeout_sec=timeout_sec,
                )
            ]
        return await self.model_actor.translate_batch(
            texts,
            target_language=target_language,
            source_language=source_language,
            max_new_tokens=self.config.max_new_tokens,
            timeout_sec=timeout_sec,
        )

    def _take_stable_batch(self) -> list[_StableJob]:
        if not self._stable_queue:
            return []
        batch = [self._stable_queue.popleft()]
        while self._stable_queue and self._can_append_to_stable_batch(batch, self._stable_queue[0]):
            batch.append(self._stable_queue.popleft())
        return batch

    def _stable_batches(self, jobs: list[_StableJob]) -> list[list[_StableJob]]:
        batches: list[list[_StableJob]] = []
        current: list[_StableJob] = []
        for job in jobs:
            if not current:
                current = [job]
                continue
            if not self._can_append_to_stable_batch(current, job):
                batches.append(current)
                current = [job]
            else:
                current.append(job)
        if current:
            batches.append(current)
        return batches

    def _can_append_to_stable_batch(self, batch: list[_StableJob], job: _StableJob) -> bool:
        return (
            bool(batch)
            and len(batch) < max(1, int(self.config.stable_batch_size))
            and job.source_language == batch[0].source_language
            and job.target_language == batch[0].target_language
        )

    async def _stop_worker(self, *, cancel_running_preview: bool = False) -> None:
        task = self._worker_task
        if task is None:
            return
        self._wake.set()
        if cancel_running_preview:
            async with self._lock:
                running_job_kind = self._running_job_kind
            if running_job_kind == "preview" and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                async with self._lock:
                    self._running_job_kind = None
                self._worker_task = None
                return
        await task
        self._worker_task = None

    async def _clear_running_job(self, kind: str) -> None:
        async with self._lock:
            if self._running_job_kind == kind:
                self._running_job_kind = None

    def _make_stable_job(
        self,
        revision: int,
        segment: dict[str, Any],
        *,
        target_language: str | None,
    ) -> _StableJob | None:
        source_text = str(segment.get("text") or "").strip()
        target = str(target_language or "").strip()
        if not source_text or not target:
            return None
        return _StableJob(
            source_revision=int(revision),
            source_segment_id=str(segment.get("id") or ""),
            source_segment_index=int(segment.get("index") or 0),
            source_text=source_text,
            source_language=str(segment.get("language") or self.config.source_language or ""),
            target_language=target,
        )

    def _stable_status(self, job: _StableJob, code: str, message: str) -> dict[str, Any]:
        return {
            "type": "translation_status",
            "scope": "stable",
            "code": str(code),
            "source_revision": int(job.source_revision),
            "source_segment_id": job.source_segment_id,
            "source_segment_index": int(job.source_segment_index),
            "target_language": job.target_language,
            "message": str(message),
        }

    def _preview_timeout_sec(self) -> float:
        return max(0.001, int(self.config.preview_timeout_ms) / 1000.0)


def _consume_future(future: Any) -> None:
    try:
        future.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _wait_future_result(future: asyncio.Future[Any], *, timeout_sec: float | None) -> Any:
    # ``future`` is a ``loop.run_in_executor`` future; it is resolved through
    # ``call_soon_threadsafe`` when the worker thread finishes, so awaiting it
    # wakes the event loop immediately. Awaiting directly (instead of polling on
    # a 0.01s timer) removes up to ~10ms of scheduling latency from every
    # translation result. On timeout ``wait_for`` cancels ``future``; the
    # orphaned worker job still drains on the single-worker executor and its
    # result is discarded by the ``_consume_future`` done-callback.
    if timeout_sec is None:
        return await future
    return await asyncio.wait_for(future, float(timeout_sec))


__all__ = ["RealtimeTranslationConfig", "RealtimeTranslationRuntime", "TranslationModelActor"]
