# coding=utf-8
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from qwen3_asr_runtime.realtime_timestamps import (
    AudioTimelineBuffer,
    RealtimeTimestampConfig,
    RealtimeTimestampRuntime,
    StableTimingJob,
    TimestampModelActor,
)
from qwen3_asr_runtime.transcript_store import TranscriptStore


@dataclass(frozen=True)
class FakeAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class FakeAlignResult:
    items: list[FakeAlignItem]


class FakeAligner:
    def __init__(self, items: list[FakeAlignItem]) -> None:
        self.items = items
        self.calls: list[tuple[np.ndarray, str, str]] = []

    def align(
        self, *, audio: object, text: str, language: str
    ) -> list[FakeAlignResult]:
        wav, sample_rate = audio  # type: ignore[misc]
        if sample_rate != 16_000:
            raise AssertionError(f"unexpected sample rate: {sample_rate}")
        self.calls.append((np.asarray(wav), text, language))
        return [FakeAlignResult(items=self.items)]


class SlowFakeAligner(FakeAligner):
    def __init__(self, items: list[FakeAlignItem], *, delay_sec: float) -> None:
        super().__init__(items)
        self._delay_sec = float(delay_sec)

    def align(
        self, *, audio: object, text: str, language: str
    ) -> list[FakeAlignResult]:
        time.sleep(self._delay_sec)
        return super().align(audio=audio, text=text, language=language)


class BlockingFakeAligner(FakeAligner):
    def __init__(self, items: list[FakeAlignItem]) -> None:
        super().__init__(items)
        self.started = threading.Event()
        self.release = threading.Event()

    def align(
        self, *, audio: object, text: str, language: str
    ) -> list[FakeAlignResult]:
        wav, sample_rate = audio  # type: ignore[misc]
        if sample_rate != 16_000:
            raise AssertionError(f"unexpected sample rate: {sample_rate}")
        self.calls.append((np.asarray(wav), text, language))
        self.started.set()
        self.release.wait(timeout=2.0)
        return [FakeAlignResult(items=self.items)]


async def wait_for_thread_event(
    event: threading.Event, *, timeout: float = 0.5
) -> None:
    if not await asyncio.to_thread(event.wait, timeout):
        raise AssertionError("timed out waiting for aligner thread")


@pytest.fixture
def timestamp_actor():
    actors: list[TimestampModelActor] = []

    def make(aligner: FakeAligner) -> TimestampModelActor:
        actor = TimestampModelActor(aligner)
        actors.append(actor)
        return actor

    yield make

    for actor in reversed(actors):
        actor.close(wait=True)


class TestRealtimeTimestampRuntime:
    def test_ready_payload_exposes_forced_aligner_language_contract(
        self, timestamp_actor
    ) -> None:
        actor = timestamp_actor(FakeAligner([]))
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        runtime = RealtimeTimestampRuntime(
            actor,
            store=TranscriptStore(transcript_id="t1"),
            audio_buffer=AudioTimelineBuffer(),
            config=RealtimeTimestampConfig(),
            event_queue=queue,
        )

        ready = runtime.ready_payload()

        assert "Japanese" in ready["allowed_source_languages"]
        assert "Arabic" not in ready["allowed_source_languages"]

    def test_model_actor_warmup_uses_aligner_without_requiring_items(
        self, timestamp_actor
    ) -> None:
        aligner = FakeAligner([])
        actor = timestamp_actor(aligner)

        actor.warmup(
            np.zeros(4000, dtype=np.float32), text="你好。", language="Chinese"
        )

        assert len(aligner.calls) == 1
        assert aligner.calls[0][0].shape == (4000,)
        assert aligner.calls[0][1:] == ("你好。", "Chinese")

    async def test_model_actor_align_items_exposes_item_level_result(
        self, timestamp_actor
    ) -> None:
        items = [FakeAlignItem(text="你", start_time=0.1, end_time=0.2)]
        aligner = FakeAligner(items)
        actor = timestamp_actor(aligner)

        result, error = await actor.align_items(
            np.zeros(4000, dtype=np.float32),
            text="你",
            language="Chinese",
            timeout_sec=1.0,
        )

        assert error is None
        assert result is not None
        assert result.items == items
        assert aligner.calls[0][1:] == ("你", "Chinese")

    async def test_model_actor_cancels_queued_align_items_when_task_is_cancelled(
        self, timestamp_actor
    ) -> None:
        items = [FakeAlignItem(text="你", start_time=0.1, end_time=0.2)]
        aligner = BlockingFakeAligner(items)
        actor = timestamp_actor(aligner)
        running = asyncio.create_task(
            actor.align_items(
                np.zeros(4000, dtype=np.float32),
                text="one",
                language="Chinese",
                timeout_sec=None,
            )
        )
        await wait_for_thread_event(aligner.started)

        queued = asyncio.create_task(
            actor.align_items(
                np.zeros(4000, dtype=np.float32),
                text="two",
                language="Chinese",
                timeout_sec=None,
            )
        )
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued

        aligner.release.set()
        result, error = await asyncio.wait_for(running, timeout=0.5)
        await asyncio.to_thread(actor.close, wait=True)

        assert error is None
        assert result is not None
        assert [call[1] for call in aligner.calls] == ["one"]

    def test_audio_timeline_buffer_crops_across_appended_chunks(self) -> None:
        buffer = AudioTimelineBuffer()
        buffer.append(np.arange(0, 5, dtype=np.float32))
        buffer.append(np.arange(5, 10, dtype=np.float32))

        crop, crop_start = buffer.crop(start_sample=3, end_sample=8, pad_samples=1)

        assert crop_start == 2
        assert buffer.end_sample == 10
        np.testing.assert_array_equal(crop, np.arange(2, 9, dtype=np.float32))

        crop[0] = -1
        recrop, _ = buffer.crop(start_sample=3, end_sample=8, pad_samples=1)
        np.testing.assert_array_equal(recrop, np.arange(2, 9, dtype=np.float32))

    def test_audio_timeline_buffer_trims_retained_prefix(self) -> None:
        buffer = AudioTimelineBuffer()
        buffer.append(np.arange(0, 5, dtype=np.float32))
        buffer.append(np.arange(5, 10, dtype=np.float32))

        buffer.trim_before(6)

        assert buffer.start_sample == 6
        assert buffer.end_sample == 10
        crop, crop_start = buffer.crop(start_sample=0, end_sample=10)
        assert crop_start == 6
        np.testing.assert_array_equal(crop, np.arange(6, 10, dtype=np.float32))

    async def test_aligns_segment_from_crop_relative_items_and_updates_store(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(16_000, dtype=np.float32))
        aligner = FakeAligner(
            [
                FakeAlignItem(text="第", start_time=0.100, end_time=0.180),
                FakeAlignItem(text="句", start_time=0.420, end_time=0.560),
            ]
        )
        actor = timestamp_actor(aligner)
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=queue,
        )
        await runtime.start()

        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=segment.id,
                    source_text=segment.text,
                    source_language=segment.language,
                    start_sample=0,
                    end_sample=16_000,
                )
            ]
        )
        timing_update = await asyncio.wait_for(queue.get(), timeout=1.0)
        await runtime.close()

        assert timing_update == {
            "type": "transcript_timing_update",
            "source_segment_id": "seg_000001",
            "start_ms": 100,
            "end_ms": 560,
            "timing_status": "aligned",
        }
        assert store.stable_segments[0].start_ms == 100
        assert store.stable_segments[0].end_ms == 560
        assert store.stable_segments[0].timing_status == "aligned"
        assert len(aligner.calls) == 1
        assert aligner.calls[0][1:] == ("第一句", "Chinese")

    async def test_runtime_trims_audio_before_completed_timestamp_jobs(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        second = store.append_stable_segment(
            text="第二句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.arange(0, 32_000, dtype=np.float32))
        aligner = FakeAligner([FakeAlignItem(text="句", start_time=0.0, end_time=0.5)])
        actor = timestamp_actor(aligner)
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=queue,
        )
        await runtime.start()

        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=first.id,
                    source_text=first.text,
                    source_language=first.language,
                    start_sample=0,
                    end_sample=16_000,
                )
            ]
        )
        await asyncio.wait_for(queue.get(), timeout=1.0)

        assert buffer.start_sample == 16_000

        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=second.id,
                    source_text=second.text,
                    source_language=second.language,
                    start_sample=16_000,
                    end_sample=32_000,
                )
            ]
        )
        await asyncio.wait_for(queue.get(), timeout=1.0)
        await runtime.close()

        assert buffer.start_sample == 32_000
        np.testing.assert_array_equal(
            aligner.calls[0][0], np.arange(0, 16_000, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            aligner.calls[1][0], np.arange(16_000, 32_000, dtype=np.float32)
        )

    async def test_full_event_queue_defers_live_timing_update_until_finish(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(16_000, dtype=np.float32))
        actor = timestamp_actor(
            FakeAligner([FakeAlignItem(text="句", start_time=0.0, end_time=0.5)])
        )
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=1)
        queue.put_nowait({"type": "ready"})
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=queue,
        )
        await runtime.start()

        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=segment.id,
                    source_text=segment.text,
                    source_language=segment.language,
                    start_sample=0,
                    end_sample=16_000,
                )
            ]
        )
        for _ in range(20):
            if store.stable_segments[0].timing_status == "aligned":
                break
            await asyncio.sleep(0.01)

        assert queue.qsize() == 1
        assert store.stable_segments[0].timing_status == "aligned"
        events = await runtime.finish([])

        assert events[0]["type"] == "transcript_timing_update"
        assert events[0]["source_segment_id"] == segment.id

    async def test_unsupported_forced_aligner_language_fails_without_model_call(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="مرحبا",
            start_ms=None,
            end_ms=None,
            language="Arabic",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(16_000, dtype=np.float32))
        aligner = FakeAligner(
            [FakeAlignItem(text="مرحبا", start_time=0.1, end_time=0.4)]
        )
        actor = timestamp_actor(aligner)
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=queue,
        )
        await runtime.start()

        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=segment.id,
                    source_text=segment.text,
                    source_language=segment.language,
                    start_sample=0,
                    end_sample=16_000,
                )
            ]
        )
        timing_update = await asyncio.wait_for(queue.get(), timeout=1.0)
        await runtime.close()

        assert timing_update["timing_status"] == "failed"
        assert aligner.calls == []
        assert buffer.start_sample == 16_000

    async def test_finish_marks_unaligned_segment_failed_before_final_snapshot(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="尾句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(16_000, dtype=np.float32))
        actor = timestamp_actor(FakeAligner([]))
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=asyncio.Queue(),
        )
        await runtime.start()

        events = await runtime.finish(
            [
                StableTimingJob(
                    source_segment_id=segment.id,
                    source_text=segment.text,
                    source_language=segment.language,
                    start_sample=0,
                    end_sample=16_000,
                )
            ]
        )

        assert events[0]["type"] == "transcript_timing_update"
        assert events[0]["source_segment_id"] == "seg_000001"
        assert events[0]["start_ms"] is None
        assert events[0]["end_ms"] is None
        assert events[0]["timing_status"] == "failed"
        assert store.final_event()["segments"][0]["timing_status"] == "failed"

    async def test_timing_patch_waits_for_shared_store_lock(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="锁住",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(16_000, dtype=np.float32))
        store_lock = asyncio.Lock()
        actor = timestamp_actor(
            FakeAligner([FakeAlignItem(text="锁", start_time=0.1, end_time=0.2)])
        )
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=asyncio.Queue(),
            store_lock=store_lock,
        )

        async with store_lock:
            finish_task = asyncio.create_task(
                runtime.finish(
                    [
                        StableTimingJob(
                            source_segment_id=segment.id,
                            source_text=segment.text,
                            source_language=segment.language,
                            start_sample=0,
                            end_sample=16_000,
                        )
                    ]
                )
            )
            await asyncio.sleep(0.05)
            assert not finish_task.done()
            assert store.stable_segments[0].timing_status == "pending"

        events = await asyncio.wait_for(finish_task, timeout=1.0)
        assert events[0]["timing_status"] == "aligned"
        assert store.stable_segments[0].timing_status == "aligned"

    async def test_finish_aligns_queued_segments_before_new_flush_segments(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        second = store.append_stable_segment(
            text="第二句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(32_000, dtype=np.float32))
        aligner = FakeAligner([FakeAlignItem(text="句", start_time=0.0, end_time=0.5)])
        actor = timestamp_actor(aligner)
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=asyncio.Queue(),
        )
        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=first.id,
                    source_text=first.text,
                    source_language=first.language,
                    start_sample=0,
                    end_sample=16_000,
                )
            ]
        )

        await runtime.finish(
            [
                StableTimingJob(
                    source_segment_id=second.id,
                    source_text=second.text,
                    source_language=second.language,
                    start_sample=16_000,
                    end_sample=32_000,
                )
            ]
        )

        assert [call[1] for call in aligner.calls] == ["第一句", "第二句"]
        assert store.stable_segments[0].start_ms == 0
        assert store.stable_segments[1].start_ms == 1000

    def test_runtime_requires_store_that_keeps_segments(self, timestamp_actor) -> None:
        # Timing patches reference retained stable segments; a keep_segments=False store
        # has nothing to patch, so the runtime must reject it up front instead of raising
        # ValueError on every alignment.
        actor = timestamp_actor(FakeAligner([]))
        with pytest.raises(ValueError, match="keep_segments"):
            RealtimeTimestampRuntime(
                actor,
                store=TranscriptStore(transcript_id="t1", keep_segments=False),
                audio_buffer=AudioTimelineBuffer(),
                config=RealtimeTimestampConfig(),
                event_queue=asyncio.Queue(),
            )

    async def test_finish_marks_jobs_failed_when_alignment_exceeds_deadline(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        second = store.append_stable_segment(
            text="第二句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.zeros(32_000, dtype=np.float32))
        actor = timestamp_actor(
            SlowFakeAligner(
                [FakeAlignItem(text="句", start_time=0.0, end_time=0.5)], delay_sec=0.5
            )
        )
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=40),
            event_queue=asyncio.Queue(),
        )

        started = asyncio.get_running_loop().time()
        events = await runtime.finish(
            [
                StableTimingJob(
                    source_segment_id=first.id,
                    source_text=first.text,
                    source_language=first.language,
                    start_sample=0,
                    end_sample=16_000,
                ),
                StableTimingJob(
                    source_segment_id=second.id,
                    source_text=second.text,
                    source_language=second.language,
                    start_sample=16_000,
                    end_sample=32_000,
                ),
            ]
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert [event["timing_status"] for event in events] == ["failed", "failed"]
        assert all(
            event["start_ms"] is None and event["end_ms"] is None for event in events
        )
        # finish must honor the deadline, not wait for both 0.5s alignments to complete.
        assert elapsed < 0.4
        assert store.stable_segments[0].timing_status == "failed"
        assert store.stable_segments[1].timing_status == "failed"

    async def test_trim_floor_retains_pad_before_completed_job(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        segment = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.arange(0, 32_000, dtype=np.float32))
        actor = timestamp_actor(
            FakeAligner([FakeAlignItem(text="句", start_time=0.0, end_time=0.5)])
        )
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=500, finish_timeout_ms=1000),
            event_queue=queue,
        )
        await runtime.start()

        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=segment.id,
                    source_text=segment.text,
                    source_language=segment.language,
                    start_sample=8_000,
                    end_sample=16_000,
                )
            ]
        )
        await asyncio.wait_for(queue.get(), timeout=1.0)
        await runtime.close()

        # pad_ms=500 -> 8_000 samples; the trim floor keeps one pad of audio behind the
        # completed segment end (16_000) instead of trimming all the way to it.
        assert buffer.start_sample == 16_000 - 8_000

    async def test_trim_preserves_audio_for_still_queued_jobs(
        self, timestamp_actor
    ) -> None:
        store = TranscriptStore(transcript_id="t1")
        first = store.append_stable_segment(
            text="第一句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        second = store.append_stable_segment(
            text="第二句",
            start_ms=None,
            end_ms=None,
            language="Chinese",
            timing_status="pending",
        )
        buffer = AudioTimelineBuffer()
        buffer.append(np.arange(0, 32_000, dtype=np.float32))
        aligner = FakeAligner([FakeAlignItem(text="句", start_time=0.0, end_time=0.5)])
        actor = timestamp_actor(aligner)
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        runtime = RealtimeTimestampRuntime(
            actor,
            store=store,
            audio_buffer=buffer,
            config=RealtimeTimestampConfig(pad_ms=0, finish_timeout_ms=1000),
            event_queue=queue,
        )
        await runtime.start()

        # Enqueue both jobs at once so the second is still queued when the first completes
        # and advances the trim floor. The protective floor must keep the second window.
        await runtime.accept_jobs(
            [
                StableTimingJob(
                    source_segment_id=first.id,
                    source_text=first.text,
                    source_language=first.language,
                    start_sample=0,
                    end_sample=16_000,
                ),
                StableTimingJob(
                    source_segment_id=second.id,
                    source_text=second.text,
                    source_language=second.language,
                    start_sample=16_000,
                    end_sample=32_000,
                ),
            ]
        )
        await asyncio.wait_for(queue.get(), timeout=1.0)
        await asyncio.wait_for(queue.get(), timeout=1.0)
        await runtime.close()

        np.testing.assert_array_equal(
            aligner.calls[0][0], np.arange(0, 16_000, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            aligner.calls[1][0], np.arange(16_000, 32_000, dtype=np.float32)
        )
        assert buffer.start_sample == 32_000
