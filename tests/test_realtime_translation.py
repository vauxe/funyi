# coding=utf-8
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from qwen3_asr_runtime.realtime_translation import (
    RealtimeTranslationConfig,
    RealtimeTranslationRuntime,
    TranslationModelActor,
)


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
        self.batch_calls: list[list[str]] = []

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

    def translate_batch(
        self,
        texts: list[str],
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del source_language, max_new_tokens
        self.batch_calls.append(list(texts))
        return [self.translate(text, target_language=target_language) for text in texts]


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


async def assert_no_event(queue: asyncio.Queue[dict[str, object]], *, timeout: float = 0.05) -> None:
    with pytest.raises(asyncio.TimeoutError):
        await get_event(queue, timeout=timeout)


async def wait_for_thread_event(event: threading.Event, *, timeout: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    if not event.is_set():
        raise AssertionError("timed out waiting for translator thread")


@pytest.fixture
def translation_actor():
    actors: list[TranslationModelActor] = []

    def make(translator: object) -> TranslationModelActor:
        actor = TranslationModelActor(translator)
        actors.append(actor)
        return actor

    yield make

    for actor in reversed(actors):
        actor.close(wait=False)


@pytest.fixture
async def make_translation_runtime(translation_actor):
    runtimes: list[RealtimeTranslationRuntime] = []

    async def make(
        translator: object,
        *,
        stable_enabled: bool = True,
        preview_enabled: bool = True,
        preview_debounce_ms: int = 0,
        preview_timeout_ms: int = 1000,
        stable_batch_size: int = 1,
    ) -> tuple[RealtimeTranslationRuntime, asyncio.Queue[dict[str, object]]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        actor = translation_actor(translator)
        runtime = RealtimeTranslationRuntime(
            actor,
            config=RealtimeTranslationConfig(
                target_language="English",
                stable_enabled=stable_enabled,
                preview_enabled=preview_enabled,
                preview_debounce_ms=preview_debounce_ms,
                preview_timeout_ms=preview_timeout_ms,
                stable_batch_size=stable_batch_size,
            ),
            event_queue=queue,
        )
        runtimes.append(runtime)
        return runtime, queue

    yield make

    for runtime in reversed(runtimes):
        await runtime.close()


@pytest.fixture
async def track_translation_runtime(translation_actor):
    del translation_actor
    runtimes: list[RealtimeTranslationRuntime] = []

    def track(runtime: RealtimeTranslationRuntime) -> RealtimeTranslationRuntime:
        runtimes.append(runtime)
        return runtime

    yield track

    for runtime in reversed(runtimes):
        await runtime.close()


class TestRealtimeTranslationRuntime:

    async def test_stable_history_never_drops_segments_under_backlog_pressure(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(FakeTranslator(), preview_enabled=False)

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
        assert [event['type'] for event in events] == ['translation_stable'] * 3
        assert [event['source_segment_id'] for event in events] == expected_ids[:3]
        assert [event['text'] for event in events] == expected_texts[:3]
        events.extend([await get_event(queue) for _ in range(2)])
        assert [event['source_segment_id'] for event in events] == expected_ids
        assert [event['text'] for event in events] == expected_texts

    async def test_stable_history_does_not_timeout_while_waiting_in_backlog(self, make_translation_runtime) -> None:
        translator = FakeTranslator(delays={"one": 0.05})
        runtime, queue = await make_translation_runtime(
            translator,
            preview_enabled=False,
            preview_timeout_ms=10,
        )
        await runtime.start()

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "one")]))
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(2, "two")]))

        events = [await get_event(queue), await get_event(queue)]
        assert [event['type'] for event in events] == ['translation_stable', 'translation_stable']
        assert [event['source_segment_id'] for event in events] == ['seg_000001', 'seg_000002']

    async def test_stable_batch_mode_preserves_order(self, make_translation_runtime) -> None:
        translator = FakeTranslator()
        runtime, queue = await make_translation_runtime(
            translator,
            preview_enabled=False,
            stable_batch_size=3,
        )
        await runtime.accept_source_event(
            transcript_update(
                1,
                stable_appends=[
                    stable_segment(1, "one"),
                    stable_segment(2, "two"),
                    stable_segment(3, "three"),
                ],
            )
        )
        await runtime.start()

        events = [await get_event(queue), await get_event(queue), await get_event(queue)]

        assert translator.batch_calls == [['one', 'two', 'three']]
        assert [event['source_segment_id'] for event in events] == ['seg_000001', 'seg_000002', 'seg_000003']
        assert [event['text'] for event in events] == ['English:one', 'English:two', 'English:three']

    async def test_stable_batch_mode_reports_empty_items_individually(self, make_translation_runtime) -> None:
        translator = FakeTranslator(empty_outputs={"two"})
        runtime, queue = await make_translation_runtime(
            translator,
            preview_enabled=False,
            stable_batch_size=2,
        )
        await runtime.start()
        await runtime.accept_source_event(
            transcript_update(
                1,
                stable_appends=[stable_segment(1, "one"), stable_segment(2, "two")],
            )
        )

        events = [await get_event(queue), await get_event(queue)]

        assert [event['type'] for event in events] == ['translation_stable', 'translation_status']
        assert [event['source_segment_id'] for event in events] == ['seg_000001', 'seg_000002']
        assert events[0]['text'] == 'English:one'
        assert events[1]['code'] == 'failed'

    async def test_stable_batch_mode_splits_source_languages(self, make_translation_runtime) -> None:
        translator = FakeTranslator()
        runtime, queue = await make_translation_runtime(
            translator,
            preview_enabled=False,
            stable_batch_size=3,
        )
        second = stable_segment(2, "two")
        third = stable_segment(3, "three")
        second["language"] = "Japanese"
        third["language"] = "Japanese"

        await runtime.accept_source_event(
            transcript_update(1, stable_appends=[stable_segment(1, "one"), second, third])
        )
        await runtime.start()
        events = [await get_event(queue), await get_event(queue), await get_event(queue)]

        assert translator.batch_calls == [['two', 'three']]
        assert translator.calls == ['one', 'two', 'three']
        assert [event['source_segment_id'] for event in events] == ['seg_000001', 'seg_000002', 'seg_000003']

    async def test_target_switch_clears_pending_stable_queue(self, make_translation_runtime) -> None:
        translator = FakeTranslator()
        runtime, queue = await make_translation_runtime(
            translator,
            preview_enabled=False,
        )

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "old")]))
        await runtime.set_target_language("Japanese")
        await runtime.start()
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(2, "new")]))

        event = await get_event(queue)
        assert event['type'] == 'translation_stable'
        assert event['source_segment_id'] == 'seg_000002'
        assert event['target_language'] == 'Japanese'
        assert event['text'] == 'Japanese:new'
        await assert_no_event(queue, timeout=0.02)
        assert translator.calls == ['new']

    async def test_target_switch_cancels_pending_preview_and_uses_new_target(self, make_translation_runtime) -> None:
        translator = FakeTranslator()
        runtime, queue = await make_translation_runtime(
            translator,
            stable_enabled=False,
            preview_debounce_ms=25,
        )
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "old", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await asyncio.sleep(0.005)
        await runtime.set_target_language("Japanese")
        await runtime.accept_source_event(
            transcript_update(2, partial={"text": "new", "language": "Chinese", "start_ms": 0, "end_ms": 200})
        )

        event = await get_event(queue)
        assert event['type'] == 'translation_preview'
        assert event['source_revision'] == 2
        assert event['target_language'] == 'Japanese'
        assert event['text'] == 'Japanese:new'
        await assert_no_event(queue, timeout=0.05)

    async def test_target_none_disables_future_translation(self, make_translation_runtime) -> None:
        translator = FakeTranslator()
        runtime, queue = await make_translation_runtime(translator, preview_debounce_ms=0)
        await runtime.start()

        await runtime.set_target_language(None)
        await runtime.accept_source_event(
            transcript_update(
                1,
                stable_appends=[stable_segment(1, "stable")],
                partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100},
            )
        )

        await assert_no_event(queue, timeout=0.02)
        assert translator.calls == []

    async def test_finish_returns_every_queued_stable_translation_once(self, make_translation_runtime) -> None:
        runtime, _queue = await make_translation_runtime(FakeTranslator(), preview_enabled=False)
        for index, text in [(1, "one"), (2, "two"), (3, "three")]:
            await runtime.accept_source_event(
                transcript_update(index, stable_appends=[stable_segment(index, text)])
            )

        events = await runtime.finish([])

        assert [event['type'] for event in events] == ['translation_stable'] * 3
        assert [event['source_segment_id'] for event in events] == ['seg_000001', 'seg_000002', 'seg_000003']

    async def test_late_stable_accept_after_finish_is_ignored(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(FakeTranslator(), preview_enabled=False)

        assert await runtime.finish([]) == []
        await runtime.accept_source_event(
            transcript_update(1, stable_appends=[stable_segment(1, "late")])
        )

        await assert_no_event(queue, timeout=0.02)

    async def test_preview_keeps_only_latest_partial(self, make_translation_runtime) -> None:
        translator = FakeTranslator()
        runtime, queue = await make_translation_runtime(
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
        assert event['type'] == 'translation_preview'
        assert event['source_revision'] == 2
        assert event['text'] == 'English:new'
        await assert_no_event(queue, timeout=0.05)

    async def test_preview_debounce_coalesces_continuous_partials(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(
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

        assert event['type'] == 'translation_preview'
        assert event['source_revision'] == 8

    async def test_preview_drops_result_if_new_partial_arrives_while_translating(
        self,
        make_translation_runtime,
    ) -> None:
        translator = FakeTranslator(delays={"old": 0.05})
        runtime, queue = await make_translation_runtime(
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
        assert translator.calls == ['old']
        await runtime.accept_source_event(
            transcript_update(2, partial={"text": "new", "language": "Chinese", "start_ms": 0, "end_ms": 200})
        )

        event = await get_event(queue)
        assert event['type'] == 'translation_preview'
        assert event['source_revision'] == 2
        assert event['text'] == 'English:new'

    async def test_preview_cancel_drops_pending_preview(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(
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

        await assert_no_event(queue, timeout=0.05)

    async def test_preview_is_visible_before_normal_stable_backlog_and_history_still_drains(
        self,
        make_translation_runtime,
    ) -> None:
        runtime, queue = await make_translation_runtime(FakeTranslator(), preview_debounce_ms=0)

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "stable")]))
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(2, "more stable")]))
        await runtime.accept_source_event(
            transcript_update(3, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await runtime.start()

        events = [await get_event(queue), await get_event(queue), await get_event(queue)]
        assert events[0]['type'] == 'translation_preview'
        assert events[0]['source_revision'] == 3
        assert [event['source_segment_id'] for event in events[1:]] == ['seg_000001', 'seg_000002']

    async def test_preview_from_same_update_is_visible_before_stable_translation(
        self,
        make_translation_runtime,
    ) -> None:
        runtime, queue = await make_translation_runtime(FakeTranslator(), preview_debounce_ms=0)
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(
                1,
                stable_appends=[stable_segment(1, "stable")],
                partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100},
            )
        )

        events = [await get_event(queue), await get_event(queue)]
        assert events[0]['type'] == 'translation_preview'
        assert events[0]['source_revision'] == 1
        assert events[1]['type'] == 'translation_stable'
        assert events[1]['source_segment_id'] == 'seg_000001'

    async def test_finish_translates_finish_segments_before_old_stable_backlog(self, make_translation_runtime) -> None:
        runtime, _queue = await make_translation_runtime(FakeTranslator(), preview_enabled=False)
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "old")]))

        events = await runtime.finish([transcript_update(2, stable_appends=[stable_segment(2, "tail")])])

        assert events[0]['type'] == 'translation_stable'
        assert events[0]['source_segment_id'] == 'seg_000002'
        assert events[0]['text'] == 'English:tail'
        assert events[1]['type'] == 'translation_stable'
        assert events[1]['source_segment_id'] == 'seg_000001'

    async def test_finish_waits_for_running_stable_without_retranslating_it(self, make_translation_runtime) -> None:
        translator = FakeTranslator(delays={"old": 0.03})
        runtime, queue = await make_translation_runtime(translator, preview_enabled=False)
        await runtime.start()
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "old")]))
        await asyncio.sleep(0.01)

        events = await runtime.finish([transcript_update(2, stable_appends=[stable_segment(2, "tail")])])

        running_event = await get_event(queue)
        assert running_event['type'] == 'translation_stable'
        assert running_event['source_segment_id'] == 'seg_000001'
        assert events[0]['type'] == 'translation_stable'
        assert events[0]['source_segment_id'] == 'seg_000002'
        assert translator.calls.count('old') == 1
        assert translator.calls.count('tail') == 1

    async def test_close_waits_for_running_stable_translation_thread(self, make_translation_runtime) -> None:
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
        runtime, queue = await make_translation_runtime(translator, preview_enabled=False, preview_timeout_ms=10)
        await runtime.start()
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "slow")]))
        for _ in range(100):
            if translator.started.is_set():
                break
            await asyncio.sleep(0.01)
        assert translator.started.is_set()

        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.02)
        assert not close_task.done()
        translator.release.set()
        await asyncio.wait_for(close_task, timeout=1.0)
        assert translator.finished.is_set()
        await assert_no_event(queue, timeout=0.01)

    async def test_preview_timeout_drops_preview_result(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(
            FakeTranslator(delays={"slow": 0.05}),
            stable_enabled=False,
            preview_timeout_ms=10,
        )
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "slow", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await assert_no_event(queue, timeout=0.08)

    async def test_preview_timeout_drops_result_but_does_not_preempt_running_model_call(
        self,
        make_translation_runtime,
    ) -> None:
        translator = BlockingTextTranslator(blocked_text="draft")
        runtime, queue = await make_translation_runtime(translator, preview_timeout_ms=10)
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await wait_for_thread_event(translator.started)
        await runtime.accept_source_event(transcript_update(2, stable_appends=[stable_segment(1, "stable")]))

        with pytest.raises(asyncio.TimeoutError):
            await get_event(queue, timeout=0.05)

        translator.release.set()
        event = await get_event(queue, timeout=0.5)
        assert event['type'] == 'translation_stable'
        assert event['source_segment_id'] == 'seg_000001'

    async def test_model_actor_serializes_concurrent_translate_calls(self, translation_actor) -> None:
        class ConcurrentDetectingTranslator(FakeTranslator):
            def __init__(self) -> None:
                super().__init__(delays={"one": 0.03, "two": 0.03})
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def translate(
                self,
                text: str,
                *,
                target_language: str,
                source_language: str = "",
                max_new_tokens: int | None = None,
            ) -> str:
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    return super().translate(
                        text,
                        target_language=target_language,
                        source_language=source_language,
                        max_new_tokens=max_new_tokens,
                    )
                finally:
                    with self.lock:
                        self.active -= 1

        translator = ConcurrentDetectingTranslator()
        actor = translation_actor(translator)
        results = await asyncio.gather(
            actor.translate(
                "one",
                target_language="English",
                source_language="Chinese",
                max_new_tokens=None,
                timeout_sec=None,
            ),
            actor.translate(
                "two",
                target_language="English",
                source_language="Chinese",
                max_new_tokens=None,
                timeout_sec=None,
            ),
        )

        assert results == [('English:one', None), ('English:two', None)]
        assert translator.max_active == 1

    async def test_model_actor_runs_prewarm_and_runtime_on_same_thread(
        self,
        translation_actor,
        track_translation_runtime,
    ) -> None:
        class ThreadRecordingTranslator(FakeTranslator):
            def __init__(self) -> None:
                super().__init__()
                self.warmup_threads: list[int] = []
                self.translate_threads: list[int] = []

            def warmup(self, texts: object, **kwargs: object) -> list[object]:
                del kwargs
                text_list = list(texts)  # type: ignore[arg-type]
                self.warmup_threads.append(threading.get_ident())
                return [object() for _ in text_list]

            def translate(
                self,
                text: str,
                *,
                target_language: str,
                source_language: str = "",
                max_new_tokens: int | None = None,
            ) -> str:
                self.translate_threads.append(threading.get_ident())
                return super().translate(
                    text,
                    target_language=target_language,
                    source_language=source_language,
                    max_new_tokens=max_new_tokens,
                )

        translator = ThreadRecordingTranslator()
        actor = translation_actor(translator)
        results = actor.warmup(
            ["short", "medium"],
            target_language="English",
            source_language="Chinese",
            max_new_tokens=16,
            sync_cuda=True,
        )
        runtime = track_translation_runtime(
            RealtimeTranslationRuntime(
                actor,
                config=RealtimeTranslationConfig(target_language="English", preview_enabled=False),
                event_queue=asyncio.Queue(),
            )
        )
        await runtime.start()
        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "stable")]))

        assert len(results) == 2
        for _ in range(100):
            if translator.translate_threads:
                break
            await asyncio.sleep(0.005)
        assert translator.warmup_threads == translator.translate_threads

    async def test_finish_waits_for_running_preview_model_call_before_stable_history(
        self,
        make_translation_runtime,
    ) -> None:
        translator = BlockingTextTranslator(blocked_text="draft")
        runtime, _queue = await make_translation_runtime(translator, preview_timeout_ms=1000)
        await runtime.start()

        await runtime.accept_source_event(
            transcript_update(1, partial={"text": "draft", "language": "Chinese", "start_ms": 0, "end_ms": 100})
        )
        await wait_for_thread_event(translator.started)

        finish_task = asyncio.create_task(
            runtime.finish([transcript_update(2, stable_appends=[stable_segment(1, "tail")])])
        )
        await asyncio.sleep(0.05)
        assert not finish_task.done()
        translator.release.set()
        events = await asyncio.wait_for(finish_task, timeout=0.5)

        assert [event['type'] for event in events] == ['translation_stable']
        assert events[0]['source_segment_id'] == 'seg_000001'
        assert events[0]['text'] == 'English:tail'

    async def test_stable_translation_failure_emits_failed_status(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(FakeTranslator(failures={"bad"}), preview_enabled=False)
        await runtime.start()

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "bad")]))
        event = await get_event(queue)

        assert event['type'] == 'translation_status'
        assert event['code'] == 'failed'
        assert event['source_segment_id'] == 'seg_000001'

    async def test_empty_stable_translation_emits_failed_status(self, make_translation_runtime) -> None:
        runtime, queue = await make_translation_runtime(FakeTranslator(empty_outputs={"empty"}), preview_enabled=False)
        await runtime.start()

        await runtime.accept_source_event(transcript_update(1, stable_appends=[stable_segment(1, "empty")]))
        event = await get_event(queue)

        assert event['type'] == 'translation_status'
        assert event['code'] == 'failed'
        assert event['source_segment_id'] == 'seg_000001'
