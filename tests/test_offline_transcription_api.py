# coding=utf-8
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

import realtime_server
from qwen3_asr_runtime.offline_transcription import OfflineTranscriptionOptions
from qwen3_asr_runtime.transcription_document import TranscriptDocument, TranscriptSegment
from realtime_server import TranslationServiceConfig, build_app


class FakeTranslationActor:
    pass


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
    assert "Unsupported or unreadable audio file" in response["json"]["error"]["message"]


@pytest.mark.asyncio
async def test_offline_transcription_route_allows_loopback_cors_preflight() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_options(
            app,
            "/api/transcriptions",
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
async def test_offline_transcription_route_rejects_non_loopback_cors_preflight() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        app = build_app(model=object(), asr_executor=executor)
        response = await asgi_options(
            app,
            "/api/transcriptions",
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

    async def receive() -> dict[str, object]:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(  # type: ignore[misc]
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
