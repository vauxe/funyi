# coding=utf-8
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import json
from pathlib import Path

import pytest

import realtime_server
from qwen3_asr_runtime.offline_transcription import (
    OfflineTranscriptionInputError,
    OfflineTranscriptionOptions,
    OfflineTranscriptionStreamEvent,
    OfflineTranslationUnit,
)
from qwen3_asr_runtime.transcription_document import TranscriptDocument, TranscriptSegment
from realtime_server import TranslationServiceConfig, build_app


class FakeTranslationActor:
    def __init__(self, outputs: list[str] | None = None, *, delay_sec: float = 0.0) -> None:
        self.outputs = list(outputs or [])
        self.delay_sec = float(delay_sec)
        self.calls: list[dict[str, object]] = []

    async def translate_batch(
        self,
        texts: list[str],
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
        timeout_sec: float | None,
    ) -> list[tuple[str | None, str | None]]:
        self.calls.append(
            {
                "texts": texts,
                "target_language": target_language,
                "source_language": source_language,
                "max_new_tokens": max_new_tokens,
                "timeout_sec": timeout_sec,
            }
        )
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        return [(self.outputs.pop(0), None) for _ in texts]


@pytest.mark.asyncio
async def test_offline_transcription_route_streams_upload_and_returns_document(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_transcribe_file(
        model: object,
        audio_source: str,
        *,
        options: OfflineTranscriptionOptions,
        timestamp_actor: object | None,
        translation_actor: object | None,
        translation_max_new_tokens: int | None,
        asr_executor: ThreadPoolExecutor | None,
    ) -> TranscriptDocument:
        calls.append(
            {
                "model": model,
                "body": Path(audio_source).read_bytes(),
                "options": options,
                "timestamp_actor": timestamp_actor,
                "translation_actor": translation_actor,
                "translation_max_new_tokens": translation_max_new_tokens,
                "asr_executor": asr_executor,
            }
        )
        return TranscriptDocument(
            duration_ms=1000,
            language="Chinese",
            segments=[
                TranscriptSegment(
                    id="seg_000001",
                    index=1,
                    start_ms=0,
                    end_ms=1000,
                    text="你好",
                    language="Chinese",
                    timing_status="aligned",
                    translation="hello",
                )
            ],
        )

    monkeypatch.setattr(realtime_server, "transcribe_file", fake_transcribe_file)
    model = object()
    translation_actor = FakeTranslationActor()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(
            model=model,
            asr_executor=executor,
            translation_actor=translation_actor,
            translation_service_config=TranslationServiceConfig(max_new_tokens=123),
        )
        response = await asgi_post(
            app,
            "/api/transcriptions",
            query="language=Chinese&context=terms&targetLanguage=traditional%20chinese&filename=clip.wav",
            body=b"audio-bytes",
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    assert response["json"] == {
        "schemaVersion": 1,
        "durationMs": 1000,
        "language": "Chinese",
        "text": "你好",
        "segments": [
            {
                "id": "seg_000001",
                "index": 1,
                "startMs": 0,
                "endMs": 1000,
                "text": "你好",
                "language": "Chinese",
                "timingStatus": "aligned",
                "translation": "hello",
            }
        ],
    }
    assert calls[0]["model"] is model
    assert calls[0]["body"] == b"audio-bytes"
    assert calls[0]["translation_actor"] is translation_actor
    assert calls[0]["translation_max_new_tokens"] == 123
    assert calls[0]["asr_executor"] is executor
    options = calls[0]["options"]
    assert isinstance(options, OfflineTranscriptionOptions)
    assert options.language == "Chinese"
    assert options.context == "terms"
    assert options.target_language == "Traditional Chinese"


@pytest.mark.asyncio
async def test_offline_transcription_route_rejects_translation_without_model() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_post(
            app,
            "/api/transcriptions",
            query="targetLanguage=English&filename=clip.wav",
            body=b"audio-bytes",
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 400
    assert response["json"]["error"]["code"] == "translation_unavailable"


@pytest.mark.asyncio
async def test_offline_transcription_route_sanitizes_backend_value_errors(monkeypatch) -> None:
    async def fake_transcribe_file(*_args: object, **_kwargs: object) -> TranscriptDocument:
        raise ValueError("private backend details from /tmp/secret-model-path")

    monkeypatch.setattr(realtime_server, "transcribe_file", fake_transcribe_file)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_post(
            app,
            "/api/transcriptions",
            query="filename=clip.wav",
            body=b"audio-bytes",
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 500
    assert response["json"] == {
        "error": {"code": "internal_error", "message": "Offline transcription failed."}
    }


@pytest.mark.asyncio
async def test_offline_transcription_route_sanitizes_backend_runtime_errors(monkeypatch) -> None:
    async def fake_transcribe_file(*_args: object, **_kwargs: object) -> TranscriptDocument:
        raise RuntimeError("private backend details from /tmp/secret-model-path")

    monkeypatch.setattr(realtime_server, "transcribe_file", fake_transcribe_file)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_post(
            app,
            "/api/transcriptions",
            query="filename=clip.wav",
            body=b"audio-bytes",
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 500
    assert response["json"] == {
        "error": {"code": "internal_error", "message": "Offline transcription failed."}
    }


@pytest.mark.asyncio
async def test_offline_transcription_stream_route_returns_incremental_events(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_stream_transcribe_file(
        model: object,
        audio_source: str,
        *,
        options: OfflineTranscriptionOptions,
        timestamp_actor: object | None,
        asr_executor: ThreadPoolExecutor | None,
    ):
        calls.append(
            {
                "model": model,
                "body": Path(audio_source).read_bytes(),
                "options": options,
                "timestamp_actor": timestamp_actor,
                "asr_executor": asr_executor,
            }
        )
        first = TranscriptSegment(
            id="seg_000001",
            index=1,
            start_ms=100,
            end_ms=900,
            text="你好",
            language="Chinese",
            timing_status="aligned",
        )
        second = TranscriptSegment(
            id="seg_000002",
            index=2,
            start_ms=1000,
            end_ms=1800,
            text="世界",
            language="Chinese",
            timing_status="aligned",
        )
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=first)
        yield OfflineTranscriptionStreamEvent(
            kind="translation_unit",
            translation_unit=OfflineTranslationUnit(
                source_text=first.text,
                source_language=first.language,
                source_segment_ids=(first.id,),
                source_segment_indices=(first.index,),
                anchor_segment_list_index=0,
            ),
        )
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=second)
        yield OfflineTranscriptionStreamEvent(
            kind="translation_unit",
            translation_unit=OfflineTranslationUnit(
                source_text=second.text,
                source_language=second.language,
                source_segment_ids=(second.id,),
                source_segment_indices=(second.index,),
                anchor_segment_list_index=1,
            ),
        )
        yield OfflineTranscriptionStreamEvent(
            kind="complete",
            document=TranscriptDocument(duration_ms=2000, language="Chinese", segments=[first, second]),
        )

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    model = object()
    translation_actor = FakeTranslationActor(["hello", "world"], delay_sec=0.01)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(
            model=model,
            asr_executor=executor,
            translation_actor=translation_actor,
            translation_service_config=TranslationServiceConfig(max_new_tokens=123),
        )
        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"language=Chinese&targetLanguage=English&filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    assert str(response["headers"]["content-type"]).startswith("application/x-ndjson")
    events = [json.loads(line) for line in bytes(response["body"]).decode("utf-8").splitlines()]
    assert [event["type"] for event in events] == [
        "transcript_update",
        "transcript_update",
        "translation_stable",
        "translation_stable",
        "transcript_final",
    ]
    assert events[0]["stable_appends"][0] == {
        "id": "seg_000001",
        "index": 1,
        "start_ms": 100,
        "end_ms": 900,
        "text": "你好",
        "language": "Chinese",
        "timing_status": "aligned",
    }
    assert events[1]["stable_appends"][0] == {
        "id": "seg_000002",
        "index": 2,
        "start_ms": 1000,
        "end_ms": 1800,
        "text": "世界",
        "language": "Chinese",
        "timing_status": "aligned",
    }
    assert events[2] == {
        "type": "translation_stable",
        "source_revision": 1,
        "source_segment_id": "seg_000001",
        "source_segment_index": 1,
        "source_segment_ids": ["seg_000001"],
        "source_segment_indices": [1],
        "text": "hello",
        "target_language": "English",
    }
    assert events[3] == {
        "type": "translation_stable",
        "source_revision": 2,
        "source_segment_id": "seg_000002",
        "source_segment_index": 2,
        "source_segment_ids": ["seg_000002"],
        "source_segment_indices": [2],
        "text": "world",
        "target_language": "English",
    }
    assert events[4]["revision"] == 2
    assert events[4]["final_revision"] == 2
    assert events[4]["stable_count"] == 2
    assert events[4]["segments"][0]["translation"] == "hello"
    assert events[4]["segments"][1]["translation"] == "world"
    assert events[4]["document"]["segments"][0]["translation"] == "hello"
    assert events[4]["document"]["segments"][1]["translation"] == "world"
    assert calls[0]["model"] is model
    assert calls[0]["body"] == b"audio-bytes"
    options = calls[0]["options"]
    assert isinstance(options, OfflineTranscriptionOptions)
    assert options.target_language is None
    assert translation_actor.calls == [
        {
            "texts": ["你好"],
            "target_language": "English",
            "source_language": "Chinese",
            "max_new_tokens": 123,
            "timeout_sec": 30.0,
        },
        {
            "texts": ["世界"],
            "target_language": "English",
            "source_language": "Chinese",
            "max_new_tokens": 123,
            "timeout_sec": 30.0,
        },
    ]


@pytest.mark.asyncio
async def test_offline_transcription_stream_route_times_out_translation_and_finishes(monkeypatch) -> None:
    async def fake_stream_transcribe_file(
        model: object,
        audio_source: str,
        *,
        options: OfflineTranscriptionOptions,
        timestamp_actor: object | None,
        asr_executor: ThreadPoolExecutor | None,
    ):
        del model, audio_source, options, timestamp_actor, asr_executor
        segment = TranscriptSegment(
            id="seg_000001",
            index=1,
            start_ms=100,
            end_ms=900,
            text="你好",
            language="Chinese",
            timing_status="aligned",
        )
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=segment)
        yield OfflineTranscriptionStreamEvent(
            kind="translation_unit",
            translation_unit=OfflineTranslationUnit(
                source_text=segment.text,
                source_language=segment.language,
                source_segment_ids=(segment.id,),
                source_segment_indices=(segment.index,),
                anchor_segment_list_index=0,
            ),
        )
        yield OfflineTranscriptionStreamEvent(
            kind="complete",
            document=TranscriptDocument(duration_ms=1000, language="Chinese", segments=[segment]),
        )

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    translation_actor = FakeTranslationActor(["hello"], delay_sec=1.0)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(
            model=object(),
            asr_executor=executor,
            translation_actor=translation_actor,
            translation_service_config=TranslationServiceConfig(max_new_tokens=123, stable_timeout_ms=1),
        )
        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"language=Chinese&targetLanguage=English&filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    events = [json.loads(line) for line in bytes(response["body"]).decode("utf-8").splitlines()]
    assert [event["type"] for event in events] == ["transcript_update", "translation_status", "transcript_final"]
    assert events[1] == {
        "type": "translation_status",
        "scope": "stable",
        "code": "timeout",
        "source_revision": 1,
        "source_segment_id": "seg_000001",
        "source_segment_index": 1,
        "source_segment_ids": ["seg_000001"],
        "source_segment_indices": [1],
        "target_language": "English",
        "message": "translation failed",
    }
    assert "translation" not in events[2]["document"]["segments"][0]
    assert events[2]["document"]["segments"][0]["translationStatus"] == "timeout"
    assert events[2]["document"]["segments"][0]["translationMessage"] == "translation failed"
    assert events[2]["segments"][0]["translation_status"] == "timeout"
    assert events[2]["segments"][0]["translation_message"] == "translation failed"


@pytest.mark.asyncio
async def test_offline_transcription_stream_route_translates_source_unit_covering_multiple_cues(
    monkeypatch,
) -> None:
    first = TranscriptSegment(
        id="seg_000001",
        index=1,
        start_ms=0,
        end_ms=2000,
        text="今天讨论字幕显示问题，",
        language="Chinese",
        timing_status="aligned",
    )
    second = TranscriptSegment(
        id="seg_000002",
        index=2,
        start_ms=2000,
        end_ms=3800,
        text="并且保持翻译输入完整。",
        language="Chinese",
        timing_status="aligned",
    )

    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=first)
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=second)
        yield OfflineTranscriptionStreamEvent(
            kind="translation_unit",
            translation_unit=OfflineTranslationUnit(
                source_text="今天讨论字幕显示问题，并且保持翻译输入完整。",
                source_language="Chinese",
                source_segment_ids=("seg_000001", "seg_000002"),
                source_segment_indices=(1, 2),
                anchor_segment_list_index=1,
            ),
        )
        yield OfflineTranscriptionStreamEvent(
            kind="complete",
            document=TranscriptDocument(duration_ms=4000, language="Chinese", segments=[first, second]),
        )

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    translation_actor = FakeTranslationActor(["We discuss subtitle display while preserving translation context."])
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(
            model=object(),
            asr_executor=executor,
            translation_actor=translation_actor,
            translation_service_config=TranslationServiceConfig(max_new_tokens=123),
        )
        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"language=Chinese&targetLanguage=English&filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    events = [json.loads(line) for line in bytes(response["body"]).decode("utf-8").splitlines()]
    assert [event["type"] for event in events] == [
        "transcript_update",
        "transcript_update",
        "translation_stable",
        "transcript_final",
    ]
    assert events[2] == {
        "type": "translation_stable",
        "source_revision": 2,
        "source_segment_id": "seg_000002",
        "source_segment_index": 2,
        "source_segment_ids": ["seg_000001", "seg_000002"],
        "source_segment_indices": [1, 2],
        "text": "We discuss subtitle display while preserving translation context.",
        "target_language": "English",
    }
    assert events[3]["document"]["segments"][0].get("translation") is None
    assert events[3]["document"]["segments"][1]["translation"] == (
        "We discuss subtitle display while preserving translation context."
    )
    assert events[3]["segments"][0].get("translation") is None
    assert events[3]["segments"][1]["translation"] == (
        "We discuss subtitle display while preserving translation context."
    )
    assert events[3]["document"]["translationUnits"] == [
        {
            "text": "We discuss subtitle display while preserving translation context.",
            "targetLanguage": "English",
            "sourceSegmentIds": ["seg_000001", "seg_000002"],
            "sourceSegmentIndices": [1, 2],
        }
    ]
    assert translation_actor.calls == [
        {
            "texts": ["今天讨论字幕显示问题，并且保持翻译输入完整。"],
            "target_language": "English",
            "source_language": "Chinese",
            "max_new_tokens": 123,
            "timeout_sec": 30.0,
        }
    ]


@pytest.mark.asyncio
async def test_offline_stream_translation_units_apply_backpressure(monkeypatch) -> None:
    yielded_segments = 0
    translation_started = asyncio.Event()
    release_translation = asyncio.Event()

    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        nonlocal yielded_segments
        for index in range(1, 20):
            segment = TranscriptSegment(
                id=f"seg_{index:06d}",
                index=index,
                start_ms=(index - 1) * 1000,
                end_ms=index * 1000,
                text=f"source {index}",
                language="English",
            )
            yielded_segments += 1
            yield OfflineTranscriptionStreamEvent(kind="segment", segment=segment)
            yield OfflineTranscriptionStreamEvent(
                kind="translation_unit",
                translation_unit=OfflineTranslationUnit(
                    source_text=segment.text,
                    source_language=segment.language,
                    source_segment_ids=(segment.id,),
                    source_segment_indices=(segment.index,),
                    anchor_segment_list_index=index - 1,
                ),
            )
        yield OfflineTranscriptionStreamEvent(
            kind="complete",
            document=TranscriptDocument(duration_ms=20_000, language="English", segments=[]),
        )

    class BlockingTranslationActor:
        async def translate_batch(
            self,
            texts: list[str],
            *,
            target_language: str,
            source_language: str,
            max_new_tokens: int | None,
            timeout_sec: float | None,
        ) -> list[tuple[str | None, str | None]]:
            del texts, target_language, source_language, max_new_tokens, timeout_sec
            translation_started.set()
            await release_translation.wait()
            return [("translated", None)]

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    output_queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=100)
    task = asyncio.create_task(
        realtime_server._produce_offline_stream_payloads(
            object(),
            "unused.wav",
            options=OfflineTranscriptionOptions(),
            timestamp_actor=None,
            translation_actor=BlockingTranslationActor(),
            translation_target_language="English",
            translation_max_new_tokens=None,
            translation_timeout_sec=None,
            asr_executor=None,
            output_queue=output_queue,
        )
    )

    async def wait_for_output_segments(count: int) -> None:
        while output_queue.qsize() < count:
            await asyncio.sleep(0)

    expected_segments_before_backpressure = realtime_server._SERVICE_TRANSLATION_JOB_QUEUE_MAXSIZE + 2
    try:
        await asyncio.wait_for(translation_started.wait(), timeout=0.5)
        await asyncio.wait_for(
            wait_for_output_segments(expected_segments_before_backpressure),
            timeout=0.5,
        )
        await asyncio.sleep(0)

        assert yielded_segments == expected_segments_before_backpressure
        assert output_queue.qsize() == expected_segments_before_backpressure
    finally:
        task.cancel()
        release_translation.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_offline_transcription_stream_route_reports_runtime_errors_as_stream_errors(monkeypatch) -> None:
    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        raise RuntimeError("gpu exploded with private details")
        yield  # pragma: no cover

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    events = [json.loads(line) for line in bytes(response["body"]).decode("utf-8").splitlines()]
    assert events == [
        {
            "type": "error",
            "error": {"code": "internal_error", "message": "Offline transcription failed."},
            "fatal": True,
        }
    ]


@pytest.mark.asyncio
async def test_offline_transcription_stream_route_sanitizes_backend_value_errors(monkeypatch) -> None:
    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        raise ValueError("private backend details from /tmp/secret-model-path")
        yield  # pragma: no cover

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    events = [json.loads(line) for line in bytes(response["body"]).decode("utf-8").splitlines()]
    assert events == [
        {
            "type": "error",
            "error": {"code": "internal_error", "message": "Offline transcription failed."},
            "fatal": True,
        }
    ]


@pytest.mark.asyncio
async def test_offline_transcription_stream_route_reports_typed_input_errors(monkeypatch) -> None:
    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        raise OfflineTranscriptionInputError("Unsupported or unreadable audio file: clip.wav")
        yield  # pragma: no cover

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    events = [json.loads(line) for line in bytes(response["body"]).decode("utf-8").splitlines()]
    assert events == [
        {
            "type": "error",
            "error": {"code": "invalid_request", "message": "Unsupported or unreadable audio file: clip.wav"},
            "fatal": True,
        }
    ]


@pytest.mark.asyncio
async def test_offline_stream_producer_cancellation_does_not_wait_for_full_output_queue(monkeypatch) -> None:
    keep_stream_open = asyncio.Event()

    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        segment = TranscriptSegment(
            id="seg_000001",
            index=1,
            start_ms=0,
            end_ms=1000,
            text="hello",
            language="English",
        )
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=segment)
        await keep_stream_open.wait()

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    output_queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=1)
    task = asyncio.create_task(
        realtime_server._produce_offline_stream_payloads(
            object(),
            "unused.wav",
            options=OfflineTranscriptionOptions(),
            timestamp_actor=None,
            translation_actor=None,
            translation_target_language=None,
            translation_max_new_tokens=None,
            translation_timeout_sec=None,
            asr_executor=None,
            output_queue=output_queue,
        )
    )

    while output_queue.qsize() == 0:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_offline_transcription_stream_upload_cancellation_releases_active_session(monkeypatch) -> None:
    async def fake_stream_transcribe_file(*_args: object, **_kwargs: object):
        yield OfflineTranscriptionStreamEvent(
            kind="complete",
            document=TranscriptDocument(duration_ms=0, language="", segments=[]),
        )

    monkeypatch.setattr(realtime_server, "stream_transcribe_file", fake_stream_transcribe_file)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        first_chunk_received = asyncio.Event()
        finish_upload = asyncio.Event()
        received = False

        async def receive() -> dict[str, object]:
            nonlocal received
            if not received:
                received = True
                first_chunk_received.set()
                return {"type": "http.request", "body": b"partial", "more_body": True}
            await finish_upload.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_message: dict[str, object]) -> None:
            return

        task = asyncio.create_task(
            app(  # type: ignore[misc]
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/api/transcriptions/stream",
                    "raw_path": b"/api/transcriptions/stream",
                    "query_string": b"filename=clip.wav",
                    "headers": [(b"content-type", b"application/octet-stream")],
                    "client": ("127.0.0.1", 12345),
                    "server": ("127.0.0.1", 8000),
                },
                receive,
                send,
            )
        )
        await asyncio.wait_for(first_chunk_received.wait(), timeout=1.0)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        response = await asgi_request(
            app,
            "/api/transcriptions/stream",
            method="POST",
            query=b"filename=clip.wav",
            body=b"audio-bytes",
            headers=[(b"content-type", b"application/octet-stream")],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200


@pytest.mark.asyncio
async def test_offline_transcription_route_rejects_unreadable_audio_with_json_error() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_post(
            app,
            "/api/transcriptions",
            query="filename=clip.wav",
            body=b"not a wav file",
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 400
    assert response["json"]["error"]["code"] == "invalid_request"
    message = response["json"]["error"]["message"]
    assert "media file" in message or "ffmpeg" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/transcriptions", "/api/transcriptions/stream"])
async def test_offline_transcription_route_allows_loopback_cors_preflight(path: str) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_options(
            app,
            path,
            headers=[
                (b"origin", b"http://localhost:5173"),
                (b"access-control-request-method", b"POST"),
                (b"access-control-request-headers", b"content-type"),
            ],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 200
    assert response["headers"]["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response["headers"]["access-control-allow-methods"]
    assert "content-type" in response["headers"]["access-control-allow-headers"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/transcriptions", "/api/transcriptions/stream"])
async def test_offline_transcription_route_rejects_non_loopback_cors_preflight(path: str) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_options(
            app,
            path,
            headers=[
                (b"origin", b"http://127.0.0.1.evil.example:5173"),
                (b"access-control-request-method", b"POST"),
                (b"access-control-request-headers", b"content-type"),
            ],
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert response["status"] == 400
    assert "access-control-allow-origin" not in response["headers"]


async def asgi_post(app: object, path: str, *, query: str, body: bytes) -> dict[str, object]:
    response = await asgi_request(
        app,
        path,
        method="POST",
        query=query.encode("ascii"),
        body=body,
        headers=[(b"content-type", b"application/octet-stream")],
    )
    return {"status": response["status"], "json": json.loads(bytes(response["body"]).decode("utf-8"))}


async def asgi_options(app: object, path: str, *, headers: list[tuple[bytes, bytes]]) -> dict[str, object]:
    return await asgi_request(app, path, method="OPTIONS", headers=headers)


async def asgi_request(
    app: object,
    path: str,
    *,
    method: str,
    query: bytes = b"",
    body: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, object]:
    sent: list[dict[str, object]] = []
    received = False
    keep_connected = asyncio.Event()

    async def receive() -> dict[str, object]:
        nonlocal received
        if received:
            # StreamingResponse starts a disconnect listener after the request body
            # has been consumed. Keep the synthetic client connected until that
            # listener is cancelled by response completion.
            await keep_connected.wait()
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await asyncio.wait_for(
        app(  # type: ignore[misc]
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": query,
                "headers": headers or [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        ),
        timeout=2.0,
    )

    status = next(message for message in sent if message["type"] == "http.response.start")["status"]
    response_headers = {
        key.decode("latin1").lower(): value.decode("latin1")
        for message in sent
        if message["type"] == "http.response.start"
        for key, value in message.get("headers", [])
    }
    response_body = b"".join(
        bytes(message.get("body", b"")) for message in sent if message["type"] == "http.response.body"
    )
    return {"status": status, "headers": response_headers, "body": response_body}
