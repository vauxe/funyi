# coding=utf-8
from __future__ import annotations

import asyncio
import threading
import time
import unittest

from qwen3_asr_runtime.realtime_translation import RealtimeTranslationConfig, RealtimeTranslationRuntime


class FakeTranslator:
    model_path = "fake-hymt"

    def __init__(
        self,
        *,
        delays: dict[str, float] | None = None,
        failures: set[str] | None = None,
        empty_outputs: set[str] | None = None,
    ) -> None:
        self.delays = dict(delays or {})
        self.failures = set(failures or set())
        self.empty_outputs = set(empty_outputs or set())
        self.calls: list[str] = []

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        del source_language, max_new_tokens
        self.calls.append(text)
        if text in self.delays:
            time.sleep(self.delays[text])
        if text in self.failures:
            raise RuntimeError("boom")
        if text in self.empty_outputs:
            return ""
        return f"{target_language}:{text}"


class BlockingTextTranslator:
    model_path = "blocking-hymt"

    def __init__(self, blocked_text: str) -> None:
        self.blocked_text = blocked_text
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        del source_language, max_new_tokens
        self.calls.append(text)
        if text == self.blocked_text:
            self.started.set()
            self.release.wait(timeout=2.0)
        return f"{target_language}:{text}"


def stable_segment(index: int, text: str) -> dict[str, object]:
    return {
        "id": f"seg_{index:06d}",
        "index": index,
        "start_ms": index * 1000,
        "end_ms": index * 1000 + 900,
        "text": text,
        "language": "Chinese",
    }


def transcript_update(
    revision: int,
    *,
    stable_appends: list[dict[str, object]] | None = None,
    partial: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "transcript_update",
        "revision": revision,
        "stable_base": 0,
        "stable_count": len(stable_appends or []),
        "stable_appends": list(stable_appends or []),
        "partial": partial,
    }


async def get_event(queue: asyncio.Queue[dict[str, object]], *, timeout: float = 1.0) -> dict[str, object]:
    event = await asyncio.wait_for(queue.get(), timeout=timeout)
    queue.task_done()
    return event


async def wait_for_thread_event(event: threading.Event, *, timeout: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    if not event.is_set():
        raise AssertionError("timed out waiting for translator thread")


class RealtimeTranslationRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            await runtime.close()

    async def make_runtime(
        self,
        translator: FakeTranslator,
        *,
        stable_enabled: bool = True,
        preview_enabled: bool = True,
        preview_debounce_ms: int = 0,
        preview_timeout_ms: int = 1000,
    ) -> tuple[RealtimeTranslationRuntime, asyncio.Queue[dict[str, object]]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.runtime = RealtimeTranslationRuntime(
            translator,
            config=RealtimeTranslationConfig(
                target_language="English",
                stable_enabled=stable_enabled,
                preview_enabled=preview_enabled,
                preview_debounce_ms=preview_debounce_ms,
                preview_timeout_ms=preview_timeout_ms,
            ),
            event_queue=queue,
        )
        return self.runtime, queue

    async def test_stable_history_never_drops_segments_under_backlog_pressure(self) -> None:
        runtime, queue = await self.make_runtime(FakeTranslator(), preview_enabled=False)

        expected_ids: list[str] = []
        expected_texts: list[str] = []
        for index, text in [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")]:
            expected_ids.append(f"seg_{index:06d}")
            expected_texts.append(f"English:{text}")
            await runtime.accept_source_event(
                transcript_update(index, stable_appends=[stable_segment(index, text)])
            )
        await runtime.start()

        events = [await get_event(queue) for _ in range(3)]
        self.assertEqual([event["type"] for event in events], ["translation_stable"] * 3)
        self.assertEqual([event["source_segment_id"] for event in events], expected_ids[:3])
        self.assertEqual([event["text"] for event in events], expected_texts[:3])
        events.extend([await get_event(queue) for _ in range(2)])
        self.assertEqual([event["source_segment_id"] for event in events], expected_ids)
        self.assertEqual([event["text"] for event in events], expected_texts)

    async def test_stable_history_does_not_timeout_while_waiting_in_backlog(self) -> None:
        translator = FakeTranslator(delays={"one": 0.05})
        runtime, queue = await self.make_runtime(
            translator,
            preview_enabled=False,
            preview_timeout_ms=10,
        )
        await runtime.start()

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "one")]))
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(2, "two")]))

        events = [await get_event(queue), await get_event(queue)]
        self.assertEqual([event["type"] for event in events], ["translation_stable", "translation_stable"])
        self.assertEqual([event["source_segment_id"] for event in events], ["seg_000001", "seg_000002"])

    async def test_finish_returns_every_queued_stable_translation_once(self) -> None:
        runtime, _queue = await self.make_runtime(FakeTranslator(), preview_enabled=False)
        for index, text in [(1, "one"), (2, "two"), (3, "three")]:
            await runtime.accept_source_event(
                transcript_update(index, stable_appends=[stable_segment(index, text)])
            )

        events = await runtime.finish([])

        self.assertEqual([event["type"] for event in events], ["translation_stable"] * 3)
        self.assertEqual(
            [event["source_segment_id"] for event in events],
            ["seg_000001", "seg_000002", "seg_000003"],
        )

    async def test_late_stable_accept_after_finish_is_ignored(self) -> None:
        runtime, queue = await self.make_runtime(FakeTranslator(), preview_enabled=False)

        self.assertEqual(await runtime.finish([]), [])
        await runtime.accept_source_event(
            transcript_update(1, stable_appends=[stable_segment(1, "late")])
        )

        self.assertTrue(queue.empty())

    async def test_preview_keeps_only_latest_partial(self) -> None:
        translator = FakeTranslator()
        runtime, queue = await self.make_runtime(
            translator,
            stable_enabled=False,
            preview_debounce_ms=25,
        )
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "old", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await asyncio.sleep(0.005)
        await runtime.accept_source_event(
            transcript_update(2, partial={"text": "new", "language": "Chinese", "start_ms": 0, "end_ms": 200})
        )

        event = await get_event(queue)
        self.assertEqual(event["type"], "translation_preview")
        self.assertEqual(event["source_revision"], 2)
        self.assertEqual(event["text"], "English:new")
        await asyncio.sleep(0.05)
        self.assertTrue(queue.empty())

    async def test_preview_debounce_coalesces_continuous_partials(self) -> None:
        runtime, queue = await self.make_runtime(
            FakeTranslator(),
            stable_enabled=False,
            preview_debounce_ms=30,
        )
        await runtime.start()

        async def send_partials() -> None:
            for revision in range(1, 9):
                await runtime.accept_source_event(
                    transcript_update(
                        revision,
                        partial={
                            "text": f"draft {revision}",
                            "language": "Chinese",
                            "start_ms": 0,
                            "end_ms": revision * 100,
                        },
                    )
                )
                await asyncio.sleep(0.01)

        sender = asyncio.create_task(send_partials())
        event = await get_event(queue, timeout=0.2)
        await sender

        self.assertEqual(event["type"], "translation_preview")
        self.assertEqual(event["source_revision"], 8)

    async def test_preview_drops_result_if_new_partial_arrives_while_translating(self) -> None:
        translator = FakeTranslator(delays={"old": 0.05})
        runtime, queue = await self.make_runtime(
            translator,
            stable_enabled=False,
            preview_debounce_ms=0,
        )
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "old", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        for _ in range(100):
            if translator.calls:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(translator.calls, ["old"])
        await runtime.accept_source_event(
            transcript_update(2, partial={"text": "new", "language": "Chinese", "start_ms": 0, "end_ms": 200})
        )

        event = await get_event(queue)
        self.assertEqual(event["type"], "translation_preview")
        self.assertEqual(event["source_revision"], 2)
        self.assertEqual(event["text"], "English:new")

    async def test_preview_cancel_drops_pending_preview(self) -> None:
        runtime, queue = await self.make_runtime(
            FakeTranslator(),
            stable_enabled=False,
            preview_debounce_ms=25,
        )
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await asyncio.sleep(0.005)
        await runtime.accept_source_event(transcript_update(2, partial=None))

        await asyncio.sleep(0.05)
        self.assertTrue(queue.empty())

    async def test_preview_is_visible_before_normal_stable_backlog_and_history_still_drains(self) -> None:
        runtime, queue = await self.make_runtime(FakeTranslator(), preview_debounce_ms=0)

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "stable")]))
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(2, "more stable")]))
        await runtime.accept_source_event(
            transcript_update(3, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await runtime.start()

        events = [await get_event(queue), await get_event(queue), await get_event(queue)]
        self.assertEqual(events[0]["type"], "translation_preview")
        self.assertEqual(events[0]["source_revision"], 3)
        self.assertEqual(
            [event["source_segment_id"] for event in events[1:]],
            ["seg_000001", "seg_000002"],
        )

    async def test_preview_from_same_update_is_visible_before_stable_translation(self) -> None:
        runtime, queue = await self.make_runtime(FakeTranslator(), preview_debounce_ms=0)
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(
                1,
                stable_appends=[stable_segment(1, "stable")],
                partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100},
            )
        )

        events = [await get_event(queue), await get_event(queue)]
        self.assertEqual(events[0]["type"], "translation_preview")
        self.assertEqual(events[0]["source_revision"], 1)
        self.assertEqual(events[1]["type"], "translation_stable")
        self.assertEqual(events[1]["source_segment_id"], "seg_000001")

    async def test_finish_translates_finish_segments_before_old_stable_backlog(self) -> None:
        runtime, _queue = await self.make_runtime(FakeTranslator(), preview_enabled=False)
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "old")]))

        events = await runtime.finish([transcript_update(2, stable_appends=[stable_segment(2, "tail")])])

        self.assertEqual(events[0]["type"], "translation_stable")
        self.assertEqual(events[0]["source_segment_id"], "seg_000002")
        self.assertEqual(events[0]["text"], "English:tail")
        self.assertEqual(events[1]["type"], "translation_stable")
        self.assertEqual(events[1]["source_segment_id"], "seg_000001")

    async def test_finish_waits_for_running_stable_without_retranslating_it(self) -> None:
        translator = FakeTranslator(delays={"old": 0.03})
        runtime, queue = await self.make_runtime(translator, preview_enabled=False)
        await runtime.start()
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "old")]))
        await asyncio.sleep(0.01)

        events = await runtime.finish([transcript_update(2, stable_appends=[stable_segment(2, "tail")])])

        running_event = await get_event(queue)
        self.assertEqual(running_event["type"], "translation_stable")
        self.assertEqual(running_event["source_segment_id"], "seg_000001")
        self.assertEqual(events[0]["type"], "translation_stable")
        self.assertEqual(events[0]["source_segment_id"], "seg_000002")
        self.assertEqual(translator.calls.count("old"), 1)
        self.assertEqual(translator.calls.count("tail"), 1)

    async def test_close_waits_for_running_stable_translation_thread(self) -> None:
        class BlockingTranslator:
            model_path = "blocking"

            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()
                self.finished = threading.Event()

            def translate(
                self,
                text: str,
                *,
                target_language: str,
                source_language: str = "",
                max_new_tokens: int | None = None,
            ) -> str:
                del source_language, max_new_tokens
                self.started.set()
                self.release.wait(timeout=1.0)
                self.finished.set()
                return f"{target_language}:{text}"

        translator = BlockingTranslator()
        runtime, queue = await self.make_runtime(translator, preview_enabled=False, preview_timeout_ms=10)
        await runtime.start()
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "slow")]))
        for _ in range(100):
            if translator.started.is_set():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(translator.started.is_set())

        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.02)
        self.assertFalse(close_task.done())
        translator.release.set()
        await asyncio.wait_for(close_task, timeout=1.0)
        self.assertTrue(translator.finished.is_set())
        self.assertTrue(queue.empty())
        self.runtime = None

    async def test_preview_timeout_drops_preview_result(self) -> None:
        runtime, queue = await self.make_runtime(
            FakeTranslator(delays={"slow": 0.05}),
            stable_enabled=False,
            preview_timeout_ms=10,
        )
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "slow", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await asyncio.sleep(0.08)

        self.assertTrue(queue.empty())

    async def test_preview_timeout_does_not_block_later_stable_translation(self) -> None:
        translator = BlockingTextTranslator(blocked_text="draft")
        runtime, queue = await self.make_runtime(translator, preview_timeout_ms=10)
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await wait_for_thread_event(translator.started)
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(1, "stable")]))

        event = await get_event(queue, timeout=0.2)

        self.assertEqual(event["type"], "translation_stable")
        self.assertEqual(event["source_segment_id"], "seg_000001")
        translator.release.set()

    async def test_finish_does_not_wait_for_running_preview_before_stable_history(self) -> None:
        translator = BlockingTextTranslator(blocked_text="draft")
        runtime, _queue = await self.make_runtime(translator, preview_timeout_ms=1000)
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await wait_for_thread_event(translator.started)

        events = await asyncio.wait_for(
            runtime.finish([transcript_update(2, stable_appends=[stable_segment(1, "tail")])]),
            timeout=0.2,
        )

        self.assertEqual([event["type"] for event in events], ["translation_stable"])
        self.assertEqual(events[0]["source_segment_id"], "seg_000001")
        self.assertEqual(events[0]["text"], "English:tail")
        translator.release.set()

    async def test_stable_translation_failure_emits_failed_status(self) -> None:
        runtime, queue = await self.make_runtime(FakeTranslator(failures={"bad"}), preview_enabled=False)
        await runtime.start()

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "bad")]))
        event = await get_event(queue)

        self.assertEqual(event["type"], "translation_status")
        self.assertEqual(event["code"], "failed")
        self.assertEqual(event["source_segment_id"], "seg_000001")

    async def test_empty_stable_translation_emits_failed_status(self) -> None:
        runtime, queue = await self.make_runtime(FakeTranslator(empty_outputs={"empty"}), preview_enabled=False)
        await runtime.start()

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "empty")]))
        event = await get_event(queue)

        self.assertEqual(event["type"], "translation_status")
        self.assertEqual(event["code"], "failed")
        self.assertEqual(event["source_segment_id"], "seg_000001")


if __name__ == "__main__":
    unittest.main()
