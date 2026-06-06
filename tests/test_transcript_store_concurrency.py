# coding=utf-8
"""Thread-safety regression for TranscriptStore.

Realtime ASR moved off the event loop (stable appends run on the ASR executor
thread) while forced-aligner timing patches still run on the event-loop thread,
so TranscriptStore must serialize concurrent writers.

The decisive test (`test_lock_serializes_concurrent_writers`) artificially widens
the critical section and asserts the two appends never overlap. Its companion
control (`test_harness_detects_a_missing_lock`) disables the lock and asserts the
same instrumentation *does* observe overlap — proving the decisive test would fail
if the lock were ever removed, rather than passing vacuously under the GIL.
"""

from __future__ import annotations

import contextlib
import threading
import time
from unittest import mock

from qwen3_asr_runtime.transcript_store import TranscriptStore


def _peak_concurrency_during_two_appends(store: TranscriptStore) -> int:
    """Run two appends concurrently and return the peak number of threads inside
    the locked ``append_stable_segment`` body.

    ``_previous_known_end`` is called inside the locked body; patching it to sleep
    widens the critical section so two *unsynchronized* appends provably overlap
    (peak 2), while two *synchronized* appends cannot (peak 1).
    """
    active = 0
    peak = 0
    guard = threading.Lock()
    start = threading.Barrier(2)
    original = TranscriptStore._previous_known_end

    def slow_previous_known_end(self: TranscriptStore, **kwargs: object) -> int:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return original(self, **kwargs)

    def worker() -> None:
        start.wait()
        store.append_stable_segment(text="x", start_ms=0, end_ms=10, language="English")

    with mock.patch.object(
        TranscriptStore, "_previous_known_end", slow_previous_known_end
    ):
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return peak


class TestTranscriptStoreLocking:
    def test_lock_serializes_concurrent_writers(self) -> None:
        assert _peak_concurrency_during_two_appends(TranscriptStore()) == 1

    def test_harness_detects_a_missing_lock(self) -> None:
        store = TranscriptStore()
        store._lock = contextlib.nullcontext()  # disable serialization
        assert _peak_concurrency_during_two_appends(store) == 2

    def test_synchronized_methods_are_reentrant(self) -> None:
        # update_event -> stable_count and clear_partial -> replace_partial are nested
        # synchronized calls, so the store lock must be reentrant (RLock). A plain Lock
        # would self-deadlock; run in a worker thread and fail (not hang) if it does.
        store = TranscriptStore()
        finished = threading.Event()
        error: list[BaseException] = []

        def worker() -> None:
            try:
                store.clear_partial()  # -> replace_partial (both @_synchronized)
                store.update_event(stable_base=0, stable_appends=[])  # -> stable_count
            except BaseException as exc:  # noqa: BLE001 - surface any failure to the assert
                error.append(exc)
            finally:
                finished.set()

        threading.Thread(target=worker, daemon=True).start()
        assert finished.wait(timeout=5.0), (
            "reentrant synchronized calls deadlocked (RLock required)"
        )
        assert error == []
