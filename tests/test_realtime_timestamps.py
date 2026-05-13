# coding=utf-8
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import unittest

import numpy as np

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

    def align(self, *, audio: object, text: str, language: str) -> list[FakeAlignResult]:
        wav, sample_rate = audio  # type: ignore[misc]
        if sample_rate != 16_000:
            raise AssertionError(f"unexpected sample rate: {sample_rate}")
        self.calls.append((np.asarray(wav), text, language))
        return [FakeAlignResult(items=self.items)]


class RealtimeTimestampRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def test_audio_timeline_buffer_crops_across_appended_chunks(self) -> None:
        buffer = AudioTimelineBuffer()
        buffer.append(np.arange(0, 5, dtype=np.float32))
        buffer.append(np.arange(5, 10, dtype=np.float32))

        crop, crop_start = buffer.crop(start_sample=3, end_sample=8, pad_samples=1)

        self.assertEqual(crop_start, 2)
        self.assertEqual(buffer.end_sample, 10)
        np.testing.assert_array_equal(crop, np.arange(2, 9, dtype=np.float32))

        crop[0] = -1
        recrop, _ = buffer.crop(start_sample=3, end_sample=8, pad_samples=1)
        np.testing.assert_array_equal(recrop, np.arange(2, 9, dtype=np.float32))

    async def test_aligns_segment_from_crop_relative_items_and_updates_store(self) -> None:
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
        actor = TimestampModelActor(aligner)
        self.addCleanup(actor.close, wait=True)
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

        self.assertEqual(
            timing_update,
            {
                "type": "transcript_timing_update",
                "source_segment_id": "seg_000001",
                "start_ms": 100,
                "end_ms": 560,
                "timing_status": "aligned",
            },
        )
        self.assertEqual(store.stable_segments[0].start_ms, 100)
        self.assertEqual(store.stable_segments[0].end_ms, 560)
        self.assertEqual(store.stable_segments[0].timing_status, "aligned")
        self.assertEqual(len(aligner.calls), 1)
        self.assertEqual(aligner.calls[0][1:], ("第一句", "Chinese"))

    async def test_finish_marks_unaligned_segment_failed_before_final_snapshot(self) -> None:
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
        actor = TimestampModelActor(FakeAligner([]))
        self.addCleanup(actor.close, wait=True)
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

        self.assertEqual(events[0]["type"], "transcript_timing_update")
        self.assertEqual(events[0]["source_segment_id"], "seg_000001")
        self.assertIsNone(events[0]["start_ms"])
        self.assertIsNone(events[0]["end_ms"])
        self.assertEqual(events[0]["timing_status"], "failed")
        self.assertEqual(store.final_event()["segments"][0]["timing_status"], "failed")

    async def test_timing_patch_waits_for_shared_store_lock(self) -> None:
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
        actor = TimestampModelActor(FakeAligner([FakeAlignItem(text="锁", start_time=0.1, end_time=0.2)]))
        self.addCleanup(actor.close, wait=True)
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
            self.assertFalse(finish_task.done())
            self.assertEqual(store.stable_segments[0].timing_status, "pending")

        events = await asyncio.wait_for(finish_task, timeout=1.0)
        self.assertEqual(events[0]["timing_status"], "aligned")
        self.assertEqual(store.stable_segments[0].timing_status, "aligned")

    async def test_finish_aligns_queued_segments_before_new_flush_segments(self) -> None:
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
        actor = TimestampModelActor(aligner)
        self.addCleanup(actor.close, wait=True)
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

        self.assertEqual([call[1] for call in aligner.calls], ["第一句", "第二句"])
        self.assertEqual(store.stable_segments[0].start_ms, 0)
        self.assertEqual(store.stable_segments[1].start_ms, 1000)


if __name__ == "__main__":
    unittest.main()
