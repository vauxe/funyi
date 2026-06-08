# coding=utf-8

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from uuid import uuid4
import wave

import numpy as np

from qwen3_asr_runtime.cuda_serialization import CUDA_GRAPH_CAPTURE_LOCK
from qwen3_asr_runtime.offline_transcription import (
    OfflineTranscriptionInputError,
    OfflineTranscriptionOptions,
    OfflineTranslationUnit,
    apply_unit_translation,
    stream_transcribe_file,
    transcribe_file,
)
from qwen3_asr_runtime.realtime_session import (
    RealtimeASRConfig,
    RealtimeConnectionSession,
)
from qwen3_asr_runtime.realtime_timestamps import (
    AudioTimelineBuffer,
    RealtimeTimestampConfig,
    RealtimeTimestampRuntime,
    TimestampModelActor,
)
from qwen3_asr_runtime.realtime_translation import (
    RealtimeTranslationConfig,
    RealtimeTranslationRuntime,
    TranslationModelActor,
)
from qwen3_asr_runtime.speech_gate import SpeechGate
from qwen3_asr_runtime.language_support import (
    HYMT_MODEL_CARD_LANGUAGES,
    QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES,
)
from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_FUSED_RMSNORM,
    DEFAULT_HYMT_MODEL,
    DEFAULT_HYMT_W8A16,
    HYMTGenerationConfig,
    HYMTTranslator,
)
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.transcription_document import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptTranslationUnit,
)
from qwen3_asr_runtime.utils import (
    SAMPLE_RATE,
    normalize_language_name,
    validate_language,
)
from qwen3_asr_runtime.vad import VadDecision

_SERVICE_SEND_TIMEOUT_SEC = 5.0
_SERVICE_EVENT_QUEUE_MAXSIZE = 128
# File-stream translation is a side track. Keep only a small backlog so ASR
# cannot run arbitrarily far ahead of a slow or timing-out HY-MT worker.
_SERVICE_TRANSLATION_JOB_QUEUE_MAXSIZE = 8
# DoS backstop on a single binary PCM frame (~500s of 16kHz mono s16le). Normal frames are
# fractions of a second; this only rejects pathological/abusive frames.
_SERVICE_MAX_PCM_FRAME_BYTES = 16_000_000
_SERVICE_DEBUG_PCM_SUMMARY_INTERVAL_MS = 1000
# Streaming translates short caption segments: observed output length tops out at
# ~80-87 tokens (p99 ~66) across the in-domain and opus eval sets. The library
# default of 512 is a cap, not a target (greedy stops at EOS, so it costs no
# decode time), but it sizes the static KV cache and lets a degenerate
# no-EOS run-on stall a segment for ~2.5s. 256 keeps ~3x headroom over the
# longest real output while halving that worst-case tail and the cache footprint.
_SERVICE_TRANSLATION_MAX_NEW_TOKENS = 256
_SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS = 700
_SERVICE_TRANSLATION_PREWARM_TARGET_LANGUAGES = ("Chinese", "English")
_SERVICE_TRANSLATION_PREWARM_TEXTS = (
    "你好。",
    "这个地方我先试一下。",
    "这个地方我先试一下，等会儿看转录和翻译是否正常返回。",
)
_SERVICE_TIMESTAMP_PREWARM_LANGUAGE = "Chinese"
_SERVICE_TIMESTAMP_PREWARM_TEXT = "你好。"
_SERVICE_TIMESTAMP_PREWARM_DURATION_SEC = 1.0
_DEFAULT_DEBUG_AUDIO_DIR = "local_data/realtime_debug_audio"
_SERVICE_CORS_ORIGIN_REGEX = (
    r"^(?:"
    r"https?://localhost(?::\d+)?"
    r"|https?://127(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?::\d+)?"
    r"|https?://tauri\.localhost(?::\d+)?"
    r"|tauri://localhost"
    r")$"
)
_START_COMMAND_FIELDS = frozenset(
    {
        "type",
        "session_id",
        "sample_rate",
        "audio_format",
        "language",
        "context",
        "target_language",
        "realtime_commit_mode",
    }
)
_LANGUAGE_COMMAND_FIELDS = frozenset({"type", "language", "target_language"})
_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


@dataclass(frozen=True)
class TranslationServiceConfig:
    preview_enabled: bool = True
    preview_debounce_ms: int = _SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS
    preview_timeout_ms: int = 30_000
    stable_timeout_ms: int = 30_000
    max_new_tokens: int | None = None
    stable_batch_size: int = 1
    prewarm_target_languages: tuple[str, ...] = (
        _SERVICE_TRANSLATION_PREWARM_TARGET_LANGUAGES
    )

    def __post_init__(self) -> None:
        if int(self.preview_debounce_ms) < 0:
            raise ValueError("preview_debounce_ms must be >= 0")
        if int(self.preview_timeout_ms) <= 0:
            raise ValueError("preview_timeout_ms must be > 0")
        if int(self.stable_timeout_ms) <= 0:
            raise ValueError("stable_timeout_ms must be > 0")
        if int(self.stable_batch_size) <= 0:
            raise ValueError("stable_batch_size must be > 0")


class _ActiveSessionGuard:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._open = False

    async def acquire(self) -> bool:
        async with self._lock:
            if self._open:
                return False
            self._open = True
            return True

    async def release(self) -> None:
        async with self._lock:
            self._open = False


@dataclass(frozen=True)
class _OfflineTranslationJob:
    unit: OfflineTranslationUnit
    revision: int


class WebSocketSendTimeout(RuntimeError):
    """Raised when a connected client stops consuming server output."""


class _PcmDebugSummary:
    def __init__(
        self,
        *,
        session_id: str,
        sample_rate: int = SAMPLE_RATE,
        interval_ms: int = _SERVICE_DEBUG_PCM_SUMMARY_INTERVAL_MS,
    ) -> None:
        self.session_id = session_id
        self.sample_rate = int(sample_rate)
        self.interval_samples = max(
            1, int(round(self.sample_rate * int(interval_ms) / 1000))
        )
        self.total_samples = 0
        self._next_log_sample = self.interval_samples
        self._reset_window()

    def accept(self, audio: np.ndarray, *, byte_count: int) -> str | None:
        samples, sum_squares, peak, zero_count = _pcm_debug_metrics(audio)
        if samples == 0:
            return None

        self.total_samples += samples
        self._window_frames += 1
        self._window_bytes += int(byte_count)
        self._window_samples += samples
        self._window_sum_squares += sum_squares
        self._window_peak = max(self._window_peak, peak)
        self._window_zero_count += zero_count

        if self.total_samples < self._next_log_sample:
            return None
        while self._next_log_sample <= self.total_samples:
            self._next_log_sample += self.interval_samples

        summary = _format_pcm_debug_metrics(
            samples=self._window_samples,
            sum_squares=self._window_sum_squares,
            peak=self._window_peak,
            zero_count=self._window_zero_count,
        )
        message = (
            f"PCM summary session_id={self.session_id} frames={self._window_frames} "
            f"bytes={self._window_bytes} total_ms={int(round(1000 * self.total_samples / self.sample_rate))} "
            f"{summary}"
        )
        self._reset_window()
        return message

    def _reset_window(self) -> None:
        self._window_frames = 0
        self._window_bytes = 0
        self._window_samples = 0
        self._window_sum_squares = 0.0
        self._window_peak = 0.0
        self._window_zero_count = 0


class _DebugAudioRecorder:
    def __init__(
        self, directory: str | Path, *, session_id: str, sample_rate: int = SAMPLE_RATE
    ) -> None:
        self.path = _debug_audio_path(directory, session_id=session_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(int(sample_rate))
        self._sample_rate = int(sample_rate)
        self._samples = 0
        _LOGGER.info("Saving realtime debug audio path=%s", self.path)

    def write(self, audio: np.ndarray) -> None:
        payload = _pcm_s16le_bytes(audio)
        if not payload:
            return
        self._wav.writeframes(payload)
        self._samples += len(payload) // 2

    def close(self) -> None:
        self._wav.close()
        duration_ms = int(round(1000 * self._samples / self._sample_rate))
        _LOGGER.info(
            "Saved realtime debug audio path=%s samples=%d duration_ms=%d",
            self.path,
            self._samples,
            duration_ms,
        )


# --- ASR executor helpers ---------------------------------------------------
# These run on the dedicated single-worker ASR executor so the ~55ms GPU forward
# no longer blocks the asyncio event loop. They bundle the session call with
# consume_stable_timing_jobs_for_events because both mutate session state and must
# run on the same thread; the returned (events, jobs) are handled back on the loop.


def _asr_step(
    session: Any, call: Any, *args: Any
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Run one session mutation `call(*args)` and consume its timing jobs.

    Both run on the ASR executor thread: they mutate session state, so they must
    not be split across threads. Returns (events, timing_jobs) for the loop.
    """
    events = call(*args)
    return events, session.consume_stable_timing_jobs_for_events(events)


def _asr_set_language(
    session: Any, language: str | None
) -> tuple[list[dict[str, Any]], list[Any], bool]:
    """Flush + switch ASR source language, only if it actually changed.

    The current language is compared on the executor thread (the only thread that
    writes ``session.config.language``), so the loop never reads session state.
    Returns (events, timing_jobs, changed).
    """
    if language == session.config.language:
        return [], [], False
    events = session.set_language(language)
    return events, session.consume_stable_timing_jobs_for_events(events), True


class _NoVadAdapter:
    def __init__(self) -> None:
        self._active = False
        self._samples_seen = 0

    @property
    def speech_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._samples_seen = 0

    def accept(self, audio: np.ndarray) -> VadDecision:
        sample_count = int(np.asarray(audio).reshape(-1).shape[0])
        if sample_count == 0:
            return VadDecision(speech_active=self._active)
        start_sample = self._samples_seen
        self._samples_seen += sample_count
        speech_started = not self._active
        self._active = True
        return VadDecision(
            speech_started=speech_started,
            has_speech=True,
            speech_active=True,
            speech_start_sample=start_sample if speech_started else None,
            last_speech_end_sample=self._samples_seen,
        )


def _build_speech_gate(*, no_vad: bool) -> SpeechGate:
    if no_vad:
        return SpeechGate(vad=_NoVadAdapter())
    return SpeechGate()


def build_app(
    *,
    model: Any,
    asr_executor: ThreadPoolExecutor,
    timestamp_actor: TimestampModelActor | None = None,
    timestamp_config: RealtimeTimestampConfig | None = None,
    translation_actor: TranslationModelActor | None = None,
    translation_service_config: TranslationServiceConfig | None = None,
    debug_audio_dir: str | Path | None = None,
    no_vad: bool = False,
) -> Any:
    try:
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "Install service dependencies with: uv sync --python 3.12"
        ) from exc

    lifespan = None
    if translation_actor is not None or timestamp_actor is not None:

        @asynccontextmanager
        async def model_actor_lifespan(app: Any) -> Any:
            del app
            try:
                yield
            finally:
                # wait=True: let any in-flight model call finish before teardown so we do not
                # leave a CUDA kernel/worker thread running into interpreter shutdown.
                if translation_actor is not None:
                    translation_actor.close(wait=True)
                if timestamp_actor is not None:
                    timestamp_actor.close(wait=True)

        lifespan = model_actor_lifespan

    app = FastAPI(title="Qwen3-ASR Runtime Realtime ASR Service", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_SERVICE_CORS_ORIGIN_REGEX,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    active_session = _ActiveSessionGuard()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def error_response(code: str, message: str, status_code: int) -> Any:
        return JSONResponse(
            {"error": {"code": code, "message": message}}, status_code=status_code
        )

    def busy_response() -> Any:
        return error_response("busy", "Another transcription session is active.", 409)

    @app.post("/api/transcriptions")
    async def create_transcription(
        request: Request,
        language: str | None = None,
        context: str = "",
        targetLanguage: str | None = None,
        target_language: str | None = None,
        timestamps: bool = True,
        filename: str = "",
    ) -> Any:
        if not await active_session.acquire():
            return busy_response()

        try:
            try:
                normalized_language = _normalize_optional_asr_language(language)
                target = _normalize_optional_translation_target(
                    targetLanguage or target_language
                )
            except ValueError as exc:
                return error_response("invalid_request", str(exc), 400)
            if target and translation_actor is None:
                return error_response(
                    "translation_unavailable",
                    "Translation was requested but the backend was not started with a translation model.",
                    400,
                )
            with tempfile.TemporaryDirectory(prefix="funyi-offline-") as tmpdir:
                audio_path = Path(tmpdir) / _offline_upload_filename(filename)
                try:
                    await _write_offline_upload(request, audio_path)
                except ValueError as exc:
                    return error_response("invalid_request", str(exc), 400)
                try:
                    await _ensure_translation_target_prewarmed(
                        translation_actor, translation_service_config, target
                    )
                except Exception:
                    _LOGGER.exception("Offline translation target prewarm failed.")
                    return error_response(
                        "internal_error", "Translation target prewarm failed.", 500
                    )

                options = OfflineTranscriptionOptions(
                    language=normalized_language,
                    context=str(context or ""),
                    target_language=target,
                    timestamps=bool(timestamps),
                )
                try:
                    document = await transcribe_file(
                        model,
                        str(audio_path),
                        options=options,
                        timestamp_actor=timestamp_actor,
                        translation_actor=translation_actor,
                        translation_max_new_tokens=(
                            translation_service_config.max_new_tokens
                            if translation_service_config is not None
                            else None
                        ),
                        asr_executor=asr_executor,
                    )
                except OfflineTranscriptionInputError as exc:
                    return error_response("invalid_request", str(exc), 400)
                except ValueError:
                    _LOGGER.exception(
                        "Offline transcription failed with a backend value error."
                    )
                    return error_response(
                        "internal_error", "Offline transcription failed.", 500
                    )
                except Exception:
                    _LOGGER.exception("Offline transcription failed.")
                    return error_response(
                        "internal_error", "Offline transcription failed.", 500
                    )
            return document.to_payload()
        finally:
            await active_session.release()

    @app.post("/api/transcriptions/stream")
    async def create_transcription_stream(
        request: Request,
        language: str | None = None,
        context: str = "",
        targetLanguage: str | None = None,
        target_language: str | None = None,
        timestamps: bool = True,
        filename: str = "",
    ) -> Any:
        if not await active_session.acquire():
            return busy_response()

        tmp_context: tempfile.TemporaryDirectory[str] | None = None
        stream_owns_resources = False
        try:
            try:
                normalized_language = _normalize_optional_asr_language(language)
                target = _normalize_optional_translation_target(
                    targetLanguage or target_language
                )
            except ValueError as exc:
                return error_response("invalid_request", str(exc), 400)
            if target and translation_actor is None:
                return error_response(
                    "translation_unavailable",
                    "Translation was requested but the backend was not started with a translation model.",
                    400,
                )

            tmp_context = tempfile.TemporaryDirectory(prefix="funyi-offline-")
            audio_path = Path(tmp_context.name) / _offline_upload_filename(filename)
            try:
                await _write_offline_upload(request, audio_path)
            except ValueError as exc:
                return error_response("invalid_request", str(exc), 400)
            try:
                await _ensure_translation_target_prewarmed(
                    translation_actor, translation_service_config, target
                )
            except Exception:
                _LOGGER.exception("Offline stream translation target prewarm failed.")
                return error_response(
                    "internal_error", "Translation target prewarm failed.", 500
                )

            stream_tmp_context = tmp_context

            async def event_stream() -> Any:
                output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
                    maxsize=_SERVICE_EVENT_QUEUE_MAXSIZE
                )
                translation_max_new_tokens = (
                    translation_service_config.max_new_tokens
                    if translation_service_config is not None
                    else None
                )
                translation_timeout_sec = (
                    _translation_stable_timeout_sec(translation_service_config)
                    if translation_service_config is not None
                    else None
                )
                producer_task = asyncio.create_task(
                    _produce_offline_stream_payloads(
                        model,
                        str(audio_path),
                        options=OfflineTranscriptionOptions(
                            language=normalized_language,
                            context=str(context or ""),
                            timestamps=bool(timestamps),
                        ),
                        timestamp_actor=timestamp_actor,
                        translation_actor=translation_actor,
                        translation_target_language=target,
                        translation_max_new_tokens=translation_max_new_tokens,
                        translation_timeout_sec=translation_timeout_sec,
                        asr_executor=asr_executor,
                        output_queue=output_queue,
                    )
                )
                try:
                    while True:
                        payload = await output_queue.get()
                        if payload is None:
                            break
                        yield _json_line(payload)
                    await producer_task
                finally:
                    try:
                        if not producer_task.done():
                            producer_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await producer_task
                    finally:
                        try:
                            stream_tmp_context.cleanup()
                        finally:
                            await active_session.release()

            response = StreamingResponse(
                event_stream(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-store"},
            )
            stream_owns_resources = True
            return response
        finally:
            if not stream_owns_resources:
                try:
                    if tmp_context is not None:
                        tmp_context.cleanup()
                finally:
                    await active_session.release()

    @app.websocket("/ws/asr")
    async def websocket_asr(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict[str, Any] | None] | None = None
        sender_task: asyncio.Task[None] | None = None
        translation: RealtimeTranslationRuntime | None = None
        timestamps: RealtimeTimestampRuntime | None = None
        audio_recorder: _DebugAudioRecorder | None = None

        if not await active_session.acquire():
            with suppress(WebSocketSendTimeout):
                await _send_error_and_close(
                    websocket, "Another realtime session is active.", code=1013
                )
            await _close_websocket(websocket, code=1013)
            return

        try:
            start_payload = await _receive_start(websocket)
            if start_payload is None:
                return
            if timestamp_actor is None or timestamp_config is None:
                await _send_error_and_close(
                    websocket,
                    "Realtime ASR requires --timestamp-model; ASR and forced aligner are one backend path.",
                    code=1011,
                )
                return
            session_id = str(start_payload.get("session_id") or "default")
            store = TranscriptStore(transcript_id=session_id, keep_segments=True)
            try:
                config = _build_realtime_session_config(start_payload)
            except ValueError as exc:
                await _send_error_and_close(websocket, str(exc), code=1003)
                return
            try:
                session_translation_config = _session_translation_config(
                    start_payload, translation_service_config
                )
            except ValueError as exc:
                await _send_error_and_close(websocket, str(exc), code=1003)
                return
            event_queue = asyncio.Queue(maxsize=_SERVICE_EVENT_QUEUE_MAXSIZE)
            sender_task = asyncio.create_task(
                _send_queued_events(websocket, event_queue)
            )
            session = RealtimeConnectionSession(
                model,
                transcript_store=store,
                config=config,
                speech_gate=_build_speech_gate(no_vad=no_vad),
            )
            # No asyncio store_lock needed: TranscriptStore is internally thread-safe
            # (its own RLock). ASR stable appends now run on the ASR executor thread while
            # forced-aligner timing patches run here on the event-loop thread; the store's
            # lock serializes both. Translation never writes the store.
            timestamps = RealtimeTimestampRuntime(
                timestamp_actor,
                store=store,
                audio_buffer=AudioTimelineBuffer(),
                config=timestamp_config,
                event_queue=event_queue,
            )
            await timestamps.start()
            if translation_actor is not None and session_translation_config is not None:
                await _ensure_translation_target_prewarmed(
                    translation_actor,
                    translation_service_config,
                    session_translation_config.target_language,
                )
                translation = RealtimeTranslationRuntime(
                    translation_actor,
                    config=session_translation_config,
                    event_queue=event_queue,
                )
                await translation.start()
            if debug_audio_dir is not None:
                audio_recorder = _DebugAudioRecorder(
                    debug_audio_dir, session_id=session_id, sample_rate=SAMPLE_RATE
                )

            ready: dict[str, Any] = {
                "type": "ready",
                "session_id": session_id,
                "sample_rate": SAMPLE_RATE,
                "audio_format": "pcm_s16le",
            }
            ready["streaming"] = _streaming_ready_payload(config)
            ready["timestamps"] = timestamps.ready_payload()
            if translation is not None:
                ready["translation"] = translation.ready_payload()
            await _queue_event(event_queue, ready, sender_task=sender_task)
            _LOGGER.info(
                "Realtime ASR session started session_id=%s language=%s mode=aligned_streaming vad=%s aligner=%s translation=%s",
                session_id,
                config.language or "auto",
                "none" if no_vad else "silero",
                timestamp_actor.model_path or "configured",
                translation is not None,
            )

            pcm_debug_summary = (
                _PcmDebugSummary(session_id=session_id)
                if _LOGGER.isEnabledFor(logging.DEBUG)
                else None
            )

            while True:
                message = await _receive_or_sender_failed(websocket, sender_task)
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    if len(message["bytes"]) > _SERVICE_MAX_PCM_FRAME_BYTES:
                        await _queue_event(
                            event_queue,
                            {
                                "type": "error",
                                "error": (
                                    f"PCM frame too large ({len(message['bytes'])} bytes); "
                                    f"max {_SERVICE_MAX_PCM_FRAME_BYTES} bytes per frame."
                                ),
                            },
                            sender_task=sender_task,
                        )
                        continue
                    audio = decode_pcm_s16le(message["bytes"])
                    if audio_recorder is not None:
                        audio_recorder.write(audio)
                    if pcm_debug_summary is not None:
                        summary = pcm_debug_summary.accept(
                            audio, byte_count=len(message["bytes"])
                        )
                        if summary is not None:
                            _LOGGER.debug(summary)
                    timestamps.accept_audio(audio)
                    events, timing_jobs = await loop.run_in_executor(
                        asr_executor, _asr_step, session, session.ingest_audio, audio
                    )
                    if await _publish_session_events(
                        event_queue, translation, events, sender_task=sender_task
                    ):
                        await _drain_and_close(
                            websocket, event_queue, sender_task, code=1011
                        )
                        return
                    await timestamps.accept_jobs(timing_jobs)
                    continue

                if message.get("text") is None:
                    continue

                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _queue_event(
                        event_queue, {"type": "error", "error": "Invalid JSON command."}
                    )
                    continue
                if not isinstance(command, dict):
                    await _queue_event(
                        event_queue,
                        {"type": "error", "error": "Command must be a JSON object."},
                    )
                    continue
                command_type = command.get("type")
                if command_type == "flush":
                    _LOGGER.debug(
                        "Realtime command session_id=%s type=flush", session_id
                    )
                    events, timing_jobs = await loop.run_in_executor(
                        asr_executor, _asr_step, session, session.flush
                    )
                    if await _publish_session_events(
                        event_queue, translation, events, sender_task=sender_task
                    ):
                        await _drain_and_close(
                            websocket, event_queue, sender_task, code=1011
                        )
                        return
                    await timestamps.accept_jobs(timing_jobs)
                elif command_type == "set_language":
                    _LOGGER.debug(
                        "Realtime command session_id=%s type=set_language payload=%s",
                        session_id,
                        command,
                    )
                    try:
                        language_update = _parse_language_config_update(
                            command,
                            translation_service_config,
                        )
                    except ValueError as exc:
                        await _queue_event(
                            event_queue, {"type": "error", "error": str(exc)}
                        )
                        continue

                    current_target = (
                        translation.target_language if translation is not None else None
                    )
                    target_changed = (
                        "target_language" in language_update
                        and language_update["target_language"] != current_target
                    )

                    if "language" in language_update:
                        (
                            events,
                            timing_jobs,
                            language_changed,
                        ) = await loop.run_in_executor(
                            asr_executor,
                            _asr_set_language,
                            session,
                            language_update["language"],
                        )
                    else:
                        events, timing_jobs, language_changed = [], [], False
                    # Target-only change: flush the current tail before switching the
                    # translation target. A language change already flushed via set_language.
                    if target_changed and not language_changed:
                        events, timing_jobs = await loop.run_in_executor(
                            asr_executor, _asr_step, session, session.flush
                        )
                    if await _publish_session_events(
                        event_queue, translation, events, sender_task=sender_task
                    ):
                        await _drain_and_close(
                            websocket, event_queue, sender_task, code=1011
                        )
                        return
                    await timestamps.accept_jobs(timing_jobs)

                    if target_changed:
                        try:
                            translation = await _set_session_translation_target(
                                language_update["target_language"],
                                translation,
                                translation_actor=translation_actor,
                                translation_service_config=translation_service_config,
                                event_queue=event_queue,
                            )
                        except ValueError as exc:
                            await _queue_event(
                                event_queue, {"type": "error", "error": str(exc)}
                            )
                            continue
                elif command_type == "finish":
                    _LOGGER.debug(
                        "Realtime command session_id=%s type=finish", session_id
                    )
                    events, timing_jobs = await loop.run_in_executor(
                        asr_executor, _asr_step, session, session.flush
                    )
                    timing_events = await timestamps.finish(timing_jobs)
                    events.extend(timing_events)
                    events.append(store.final_event())
                    await _publish_finish_events(
                        event_queue, translation, events, sender_task=sender_task
                    )
                    await _drain_and_close(
                        websocket, event_queue, sender_task, code=1000
                    )
                    return
                else:
                    await _queue_event(
                        event_queue,
                        {
                            "type": "error",
                            "error": f"Unsupported command: {command_type}",
                        },
                    )
        except WebSocketDisconnect:
            return
        except WebSocketSendTimeout:
            _LOGGER.warning("Realtime ASR WebSocket client stopped consuming output.")
            await _close_websocket(websocket, code=1011)
            return
        except Exception as exc:
            _LOGGER.exception("Realtime ASR WebSocket session failed.")
            try:
                if (
                    event_queue is not None
                    and sender_task is not None
                    and not sender_task.done()
                ):
                    await _queue_event(
                        event_queue,
                        {
                            "type": "error",
                            "error": str(exc) or type(exc).__name__,
                            "fatal": True,
                        },
                        sender_task=sender_task,
                    )
                    await _drain_and_close(
                        websocket, event_queue, sender_task, code=1011
                    )
                else:
                    await _send_error_and_close(
                        websocket, str(exc) or type(exc).__name__, code=1011
                    )
            except Exception:
                _LOGGER.exception("Failed to send realtime ASR error response.")
            return
        finally:
            if audio_recorder is not None:
                try:
                    audio_recorder.close()
                except Exception:
                    _LOGGER.exception("Failed to close realtime debug audio recorder.")
            if translation is not None:
                await translation.close()
            if timestamps is not None:
                await timestamps.close()
            if sender_task is not None and not sender_task.done():
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
            elif sender_task is not None:
                try:
                    sender_task.result()
                except Exception:
                    pass
            await active_session.release()

    return app


async def _write_offline_upload(request: Any, path: Path) -> None:
    total = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            handle.write(chunk)
    if total == 0:
        raise ValueError("offline upload body is empty")


async def _produce_offline_stream_payloads(
    model: Any,
    audio_path: str,
    *,
    options: OfflineTranscriptionOptions,
    timestamp_actor: Any | None,
    translation_actor: Any | None,
    translation_target_language: str | None,
    translation_max_new_tokens: int | None,
    translation_timeout_sec: float | None,
    asr_executor: Any | None,
    output_queue: asyncio.Queue[dict[str, Any] | None],
) -> None:
    segments: list[TranscriptSegment] = []
    translation_units: list[TranscriptTranslationUnit] = []
    revision = 0
    final_document: TranscriptDocument | None = None
    translation_jobs: asyncio.Queue[_OfflineTranslationJob | None] | None = None
    translation_task: asyncio.Task[None] | None = None
    target_language = str(translation_target_language or "").strip()

    if target_language and translation_actor is not None:
        translation_jobs = asyncio.Queue(maxsize=_SERVICE_TRANSLATION_JOB_QUEUE_MAXSIZE)
        translation_task = asyncio.create_task(
            _offline_translation_worker(
                translation_jobs,
                output_queue,
                segments,
                translation_units,
                translation_actor=translation_actor,
                target_language=target_language,
                source_language=str(options.language or ""),
                max_new_tokens=translation_max_new_tokens,
                timeout_sec=translation_timeout_sec,
            )
        )

    try:
        async for event in stream_transcribe_file(
            model,
            audio_path,
            options=options,
            timestamp_actor=timestamp_actor,
            asr_executor=asr_executor,
        ):
            if event.kind == "segment" and event.segment is not None:
                segment = event.segment
                segments.append(segment)
                revision = max(revision + 1, int(segment.index))
                await output_queue.put(
                    _offline_segment_stream_payload(segment, revision)
                )
            elif (
                event.kind == "translation_unit" and event.translation_unit is not None
            ):
                if translation_jobs is not None:
                    await translation_jobs.put(
                        _offline_translation_job_from_unit(
                            event.translation_unit, revision
                        )
                    )
            elif event.kind == "complete" and event.document is not None:
                final_document = event.document

        if translation_jobs is not None and translation_task is not None:
            await translation_jobs.put(None)
            await translation_task
            translation_jobs = None
            translation_task = None

        if final_document is None:
            final_document = TranscriptDocument(
                duration_ms=0, language="", segments=segments
            )
        await output_queue.put(
            _offline_final_stream_payload(
                replace(
                    final_document,
                    segments=segments,
                    translation_units=translation_units,
                ),
                revision,
            )
        )
    except OfflineTranscriptionInputError as exc:
        await _cancel_offline_translation_worker(translation_task)
        translation_task = None
        await output_queue.put(
            _offline_stream_error_payload("invalid_request", str(exc))
        )
    except ValueError:
        await _cancel_offline_translation_worker(translation_task)
        translation_task = None
        _LOGGER.exception(
            "Offline transcription stream failed with a backend value error."
        )
        await output_queue.put(
            _offline_stream_error_payload(
                "internal_error", "Offline transcription failed."
            )
        )
    except asyncio.CancelledError:
        await _cancel_offline_translation_worker(translation_task)
        translation_task = None
        raise
    except Exception:
        await _cancel_offline_translation_worker(translation_task)
        translation_task = None
        _LOGGER.exception("Offline transcription stream failed.")
        await output_queue.put(
            _offline_stream_error_payload(
                "internal_error", "Offline transcription failed."
            )
        )
    finally:
        await _cancel_offline_translation_worker(translation_task)
        if not _current_task_is_cancelling():
            await output_queue.put(None)


def _current_task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


async def _cancel_offline_translation_worker(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        _LOGGER.exception("Offline translation worker failed.")


def _offline_translation_job_from_unit(
    unit: OfflineTranslationUnit,
    revision: int,
) -> _OfflineTranslationJob:
    revision = max(
        int(revision), max(unit.source_segment_indices, default=int(revision))
    )
    return _OfflineTranslationJob(unit=unit, revision=revision)


async def _offline_translation_worker(
    jobs: asyncio.Queue[_OfflineTranslationJob | None],
    output_queue: asyncio.Queue[dict[str, Any] | None],
    segments: list[TranscriptSegment],
    translation_units: list[TranscriptTranslationUnit],
    *,
    translation_actor: Any,
    target_language: str,
    source_language: str,
    max_new_tokens: int | None,
    timeout_sec: float | None,
) -> None:
    while True:
        job = await jobs.get()
        try:
            if job is None:
                return
            unit = job.unit
            translated, error_code = await _translate_offline_unit(
                unit,
                translation_actor=translation_actor,
                target_language=target_language,
                source_language=unit.source_language or source_language,
                max_new_tokens=max_new_tokens,
                timeout_sec=timeout_sec,
            )
            document_unit = apply_unit_translation(
                segments,
                unit,
                target_language=target_language,
                text=translated,
                error=error_code,
            )
            if document_unit is not None:
                translation_units.append(document_unit)
            if translated and error_code is None:
                await output_queue.put(
                    _offline_translation_stable_payload(
                        job, translated, target_language
                    )
                )
            else:
                await output_queue.put(
                    _offline_translation_status_payload(
                        job,
                        target_language,
                        code=error_code or "failed",
                        message="translation failed",
                    )
                )
        finally:
            jobs.task_done()


async def _translate_offline_unit(
    unit: OfflineTranslationUnit,
    *,
    translation_actor: Any,
    target_language: str,
    source_language: str,
    max_new_tokens: int | None,
    timeout_sec: float | None,
) -> tuple[str | None, str | None]:
    try:
        translate_call = translation_actor.translate_batch(
            [unit.source_text],
            target_language=target_language,
            source_language=source_language,
            max_new_tokens=max_new_tokens,
            timeout_sec=timeout_sec,
        )
        if timeout_sec is None:
            outputs = await translate_call
        else:
            outputs = await asyncio.wait_for(
                translate_call, timeout=max(0.001, float(timeout_sec)) + 0.1
            )
    except asyncio.TimeoutError:
        return None, "timeout"
    except asyncio.CancelledError:
        raise
    except Exception:
        _LOGGER.exception("Offline stream translation failed.")
        return None, "failed"

    text, error = outputs[0] if outputs else (None, "missing translation output")
    translated = str(text or "").strip()
    if not translated:
        return None, str(error or "failed")
    return translated, None


def _offline_stream_error_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"code": str(code), "message": str(message)},
        "fatal": True,
    }


def _translation_stable_timeout_sec(config: TranslationServiceConfig) -> float:
    return max(0.001, int(config.stable_timeout_ms) / 1000.0)


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _offline_segment_stream_payload(segment: Any, revision: int) -> dict[str, Any]:
    return {
        "type": "transcript_update",
        "revision": int(revision),
        "stable_base": max(0, int(segment.index) - 1),
        "stable_count": int(segment.index),
        "stable_appends": [_offline_segment_realtime_payload(segment)],
        "partial": None,
    }


def _offline_translation_stable_payload(
    job: _OfflineTranslationJob,
    translated: str,
    target_language: str,
) -> dict[str, Any]:
    return {
        "type": "translation_stable",
        **_offline_translation_source_payload(job),
        "text": str(translated or ""),
        "target_language": target_language,
    }


def _offline_translation_status_payload(
    job: _OfflineTranslationJob,
    target_language: str,
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "type": "translation_status",
        "scope": "stable",
        "code": str(code),
        **_offline_translation_source_payload(job),
        "target_language": target_language,
        "message": str(message),
    }


def _offline_translation_source_payload(job: _OfflineTranslationJob) -> dict[str, Any]:
    unit = job.unit
    return {
        "source_revision": int(job.revision),
        "source_segment_id": unit.source_segment_ids[-1]
        if unit.source_segment_ids
        else "",
        "source_segment_index": int(unit.source_segment_indices[-1])
        if unit.source_segment_indices
        else 0,
        "source_segment_ids": list(unit.source_segment_ids),
        "source_segment_indices": [int(index) for index in unit.source_segment_indices],
    }


def _offline_final_stream_payload(document: Any, revision: int) -> dict[str, Any]:
    return {
        "type": "transcript_final",
        "revision": int(revision),
        "final_revision": int(revision),
        "stable_count": len(document.segments),
        "duration_ms": int(document.duration_ms),
        "language": document.language,
        "segments": [
            _offline_segment_realtime_payload(segment) for segment in document.segments
        ],
        "document": document.to_payload(),
    }


def _offline_segment_realtime_payload(segment: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": segment.id,
        "index": int(segment.index),
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "text": segment.text,
        "language": segment.language,
    }
    if segment.timing_status is not None:
        payload["timing_status"] = segment.timing_status
    if segment.translation is not None:
        payload["translation"] = segment.translation
    if segment.translation_status is not None:
        payload["translation_status"] = segment.translation_status
    if segment.translation_message is not None:
        payload["translation_message"] = segment.translation_message
    return payload


def _offline_upload_filename(filename: str) -> str:
    suffix = Path(str(filename or "")).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9][a-z0-9._-]{0,15}", suffix):
        suffix = ".audio"
    return f"source{suffix}"


def _normalize_optional_asr_language(language: str | None) -> str | None:
    value = str(language or "").strip()
    if not value:
        return None
    normalized = normalize_language_name(value)
    validate_language(normalized)
    return normalized


def _normalize_optional_translation_target(target: str | None) -> str | None:
    value = str(target or "").strip()
    if not value:
        return None
    return _normalize_translation_target_language(value)


def _translation_prewarm_target_languages(
    values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    raw_values = values or list(_SERVICE_TRANSLATION_PREWARM_TARGET_LANGUAGES)
    targets: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for raw_value in str(value or "").split(","):
            raw = raw_value.strip()
            if not raw:
                continue
            target = _normalize_translation_target_language(raw)
            if target in seen:
                continue
            seen.add(target)
            targets.append(target)
    return tuple(targets)


def decode_pcm_s16le(payload: bytes) -> np.ndarray:
    if len(payload) % 2 != 0:
        payload = payload[:-1]
    if not payload:
        return np.zeros((0,), dtype=np.int16)
    return np.frombuffer(payload, dtype="<i2").copy()


def _pcm_debug_metrics(audio: np.ndarray) -> tuple[int, float, float, int]:
    samples = int(audio.shape[0])
    if samples == 0:
        return 0, 0.0, 0.0, 0

    if audio.dtype == np.int16:
        x = audio.astype(np.float32) / 32768.0
    else:
        x = audio.astype(np.float32, copy=False)
    abs_x = np.abs(x)
    peak = float(abs_x.max(initial=0.0))
    sum_squares = float(np.sum(x * x))
    zero_count = int(np.count_nonzero(abs_x <= 1.0e-6))
    return samples, sum_squares, peak, zero_count


def _format_pcm_debug_metrics(
    *,
    samples: int,
    sum_squares: float,
    peak: float,
    zero_count: int,
) -> str:
    duration_ms = int(round(1000 * int(samples) / SAMPLE_RATE))
    if int(samples) == 0:
        return "samples=0 duration_ms=0 rms_db=-inf peak=0.0000 zero_pct=100.0"
    rms = float(np.sqrt(float(sum_squares) / int(samples)))
    rms_db = "-inf" if rms <= 0.0 else f"{20 * np.log10(rms):.1f}"
    zero_pct = 100.0 * int(zero_count) / int(samples)
    return f"samples={samples} duration_ms={duration_ms} rms_db={rms_db} peak={peak:.4f} zero_pct={zero_pct:.1f}"


def _pcm_s16le_bytes(audio: np.ndarray) -> bytes:
    x = np.asarray(audio)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.shape[0] == 0:
        return b""
    if x.dtype == np.int16:
        return x.astype("<i2", copy=False).tobytes()
    clipped = np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)
    return np.rint(clipped * np.iinfo(np.int16).max).astype("<i2").tobytes()


def _debug_audio_path(directory: str | Path, *, session_id: str) -> Path:
    safe_session_id = _safe_filename_component(session_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = uuid4().hex[:8]
    return Path(directory) / f"{safe_session_id}-{timestamp}-{suffix}.wav"


def _safe_filename_component(value: str, *, limit: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "session")).strip("._-")
    return (cleaned or "session")[:limit]


def _truncate_log_text(text: Any, *, limit: int = 80) -> str:
    value = str(text or "").replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _format_event_log_summary(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "")
    if event_type == "transcript_update":
        stable_texts = [
            _truncate_log_text(segment.get("text"))
            for segment in event.get("stable_appends") or []
            if isinstance(segment, dict)
        ]
        partial = event.get("partial")
        partial_text = (
            _truncate_log_text(partial.get("text")) if isinstance(partial, dict) else ""
        )
        return (
            "type=transcript_update "
            f"revision={event.get('revision')} stable_base={event.get('stable_base')} "
            f"stable_count={event.get('stable_count')} stable_texts={stable_texts!r} partial={partial_text!r}"
        )
    if event_type == "transcript_final":
        return f"type=transcript_final stable_count={event.get('stable_count')}"
    if event_type == "transcript_status":
        return (
            "type=transcript_status "
            f"status={event.get('status')} fatal={event.get('fatal')} "
            f"message={_truncate_log_text(event.get('message'))!r}"
        )
    if event_type == "ready":
        return (
            "type=ready "
            f"session_id={event.get('session_id')} sample_rate={event.get('sample_rate')} "
            f"audio_format={event.get('audio_format')}"
        )
    if event_type == "error":
        return f"type=error fatal={event.get('fatal')} error={_truncate_log_text(event.get('error'))!r}"
    return f"type={event_type}"


def _should_log_realtime_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "")
    if event_type == "transcript_update":
        return bool(event.get("stable_appends"))
    return event_type in {
        "ready",
        "error",
        "transcript_status",
        "transcript_final",
        "translation_stable",
        "translation_status",
    }


def _build_realtime_session_config(payload: dict[str, Any]) -> RealtimeASRConfig:
    mode = str(payload.get("realtime_commit_mode") or "aligned_windowed").strip()
    if mode != "aligned_windowed":
        raise ValueError("realtime_commit_mode must be aligned_windowed")
    language = payload.get("language")
    if language is not None and str(language).strip():
        language = _normalize_aligned_source_language(str(language))
    else:
        language = None
    return RealtimeASRConfig(
        context=str(payload.get("context") or ""),
        language=language,
        force_align_timestamps=True,
    )


def _streaming_ready_payload(config: RealtimeASRConfig) -> dict[str, Any]:
    return {
        "mode": "aligned_windowed",
        "requires": ["asr", "forced_aligner"],
        "stable": {
            "source": "asr_streaming_text_and_forced_aligner",
            "patch_event": "transcript_timing_update",
            "live_stability_delay_ms": int(config.live_stability_delay_ms),
        },
    }


def _session_translation_config(
    payload: dict[str, Any],
    service_config: TranslationServiceConfig | None,
) -> RealtimeTranslationConfig | None:
    if "target_language" not in payload:
        return None
    requested_target = str(payload.get("target_language") or "").strip()
    if not requested_target:
        raise ValueError("target_language must not be empty")
    if service_config is None:
        raise ValueError("target_language requires translation model to be configured")

    normalized_target = _normalize_translation_target_language(requested_target)
    return _translation_config_for_target(normalized_target, service_config)


def _translation_config_for_target(
    target_language: str,
    service_config: TranslationServiceConfig,
) -> RealtimeTranslationConfig:
    return RealtimeTranslationConfig(
        target_language=target_language,
        preview_enabled=service_config.preview_enabled,
        preview_debounce_ms=service_config.preview_debounce_ms,
        preview_timeout_ms=service_config.preview_timeout_ms,
        max_new_tokens=service_config.max_new_tokens,
        stable_batch_size=service_config.stable_batch_size,
    )


async def _set_session_translation_target(
    target_language: str | None,
    translation: RealtimeTranslationRuntime | None,
    *,
    translation_actor: TranslationModelActor | None,
    translation_service_config: TranslationServiceConfig | None,
    event_queue: asyncio.Queue[dict[str, Any] | None],
) -> RealtimeTranslationRuntime | None:
    if translation is not None:
        await translation.set_target_language(target_language)
        return translation

    if target_language is None:
        return None

    if translation_actor is None or translation_service_config is None:
        raise ValueError("target_language requires translation model to be configured")

    translation = RealtimeTranslationRuntime(
        translation_actor,
        config=_translation_config_for_target(
            target_language, translation_service_config
        ),
        event_queue=event_queue,
    )
    await translation.start()
    return translation


def _parse_language_config_update(
    command: dict[str, Any],
    service_config: TranslationServiceConfig | None,
) -> dict[str, str | None]:
    unknown_fields = sorted(set(command) - _LANGUAGE_COMMAND_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"Unsupported set_language command field(s): {', '.join(unknown_fields)}."
        )

    update: dict[str, str | None] = {}
    if "language" in command:
        raw_language = command.get("language")
        language: str | None = None
        if raw_language is not None and str(raw_language).strip():
            language = _normalize_aligned_source_language(str(raw_language))
        update["language"] = language

    if "target_language" in command:
        raw_target = command.get("target_language")
        target_language: str | None = None
        if raw_target is not None and str(raw_target).strip():
            if service_config is None:
                raise ValueError(
                    "target_language requires translation model to be configured"
                )
            target_language = _normalize_translation_target_language(str(raw_target))
        update["target_language"] = target_language

    return update


def _normalize_supported_language(language: str) -> str:
    normalized = normalize_language_name(str(language))
    validate_language(normalized)
    return normalized


def _normalize_aligned_source_language(language: str) -> str:
    normalized = _normalize_supported_language(language)
    if normalized not in QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES:
        raise ValueError(
            f"Forced aligner does not support source language: {normalized}."
        )
    return normalized


def _normalize_language_choice(
    language: str, allowed: tuple[str, ...], *, field_name: str
) -> str:
    raw = str(language or "").strip()
    if not raw:
        raise ValueError(f"{field_name} is empty")
    by_casefold = {item.casefold(): item for item in allowed}
    normalized = by_casefold.get(raw.casefold())
    if normalized is None:
        raise ValueError(f"Unsupported {field_name}: {raw}. Supported: {list(allowed)}")
    return normalized


def _normalize_translation_target_language(language: str) -> str:
    return _normalize_language_choice(
        language, HYMT_MODEL_CARD_LANGUAGES, field_name="target_language"
    )


async def _receive_start(websocket: Any) -> dict[str, Any] | None:
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        return None
    if message.get("text") is None:
        await _send_error_and_close(
            websocket, "First frame must be a JSON start command.", code=1003
        )
        return None
    try:
        payload = json.loads(message["text"])
    except json.JSONDecodeError:
        await _send_error_and_close(
            websocket, "Start command must be valid JSON.", code=1003
        )
        return None
    if not isinstance(payload, dict):
        await _send_error_and_close(
            websocket, "Start command must be a JSON object.", code=1003
        )
        return None
    if payload.get("type") != "start":
        await _send_error_and_close(
            websocket, "First command must be type=start.", code=1003
        )
        return None
    unknown_fields = sorted(set(payload) - _START_COMMAND_FIELDS)
    if unknown_fields:
        await _send_error_and_close(
            websocket,
            f"Unsupported start command field(s): {', '.join(unknown_fields)}.",
            code=1003,
        )
        return None
    try:
        sample_rate = int(payload.get("sample_rate", SAMPLE_RATE))
    except (TypeError, ValueError):
        await _send_error_and_close(websocket, "sample_rate must be 16000.", code=1003)
        return None
    audio_format = str(payload.get("audio_format") or "pcm_s16le").lower()
    if sample_rate != SAMPLE_RATE or audio_format != "pcm_s16le":
        await _send_error_and_close(
            websocket,
            "Only mono pcm_s16le at 16000 Hz is supported.",
            code=1003,
        )
        return None
    raw_language = payload.get("language")
    if raw_language is not None and str(raw_language).strip():
        try:
            payload["language"] = _normalize_aligned_source_language(str(raw_language))
        except ValueError as exc:
            await _send_error_and_close(websocket, str(exc), code=1003)
            return None
    return payload


async def _publish_session_events(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    translation: RealtimeTranslationRuntime | None,
    events: list[dict[str, Any]],
    *,
    sender_task: asyncio.Task[None] | None = None,
) -> bool:
    fatal_status = False
    for event in events:
        if translation is not None:
            await translation.accept_source_event(event)
        if event.get("type") == "transcript_status" and bool(event.get("fatal")):
            fatal_status = True
        await _queue_event(event_queue, event, sender_task=sender_task)
    return fatal_status


async def _publish_finish_events(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    translation: RealtimeTranslationRuntime | None,
    events: list[dict[str, Any]],
    *,
    sender_task: asyncio.Task[None] | None = None,
) -> None:
    if translation is not None:
        await translation.cancel_preview()

    transcript_updates: list[dict[str, Any]] = []
    final_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "transcript_update":
            transcript_updates.append(event)
            await _queue_event(event_queue, event, sender_task=sender_task)
        elif event.get("type") == "transcript_final":
            final_events.append(event)
        else:
            await _queue_event(event_queue, event, sender_task=sender_task)

    if translation is not None:
        translation_events = await translation.finish(transcript_updates)
        for event in translation_events:
            await _queue_event(event_queue, event, sender_task=sender_task)

    for event in final_events:
        await _queue_event(event_queue, event, sender_task=sender_task)


async def _drain_and_close(
    websocket: Any,
    event_queue: asyncio.Queue[dict[str, Any] | None],
    sender_task: asyncio.Task[None] | None,
    *,
    code: int,
) -> None:
    """Signal the sender to stop, wait for it, and close the socket once."""
    with suppress(Exception):
        await _queue_event(event_queue, None, sender_task=sender_task)
    if sender_task is not None:
        with suppress(Exception):
            await sender_task
    await _close_websocket(websocket, code=code)


async def _queue_event(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    event: dict[str, Any] | None,
    *,
    sender_task: asyncio.Task[None] | None = None,
) -> None:
    """Enqueue an event for the sender, applying natural backpressure.

    A full queue is not fatal: ASR has already consumed the audio, so dropping the
    event would skip published transcript text. Instead the producer blocks until the
    sender drains the queue, which in turn backpressures audio ingest (and the client
    socket). The only fatal condition is the sender task itself finishing while we wait
    — that means the client stopped reading (its send-side timeout fired) — in which
    case we surface the sender's failure rather than block forever.
    """
    try:
        event_queue.put_nowait(event)
        return
    except asyncio.QueueFull:
        pass

    if sender_task is None or sender_task.done():
        if sender_task is not None and sender_task.done():
            _raise_sender_failure(sender_task)
        await event_queue.put(event)
        return

    put_task = asyncio.ensure_future(event_queue.put(event))
    done, _pending = await asyncio.wait(
        {put_task, sender_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if put_task in done:
        put_task.result()
        return

    put_task.cancel()
    with suppress(asyncio.CancelledError):
        await put_task
    _raise_sender_failure(sender_task)


def _raise_sender_failure(sender_task: asyncio.Task[None]) -> None:
    exc = sender_task.exception()
    if exc is not None:
        raise exc
    raise WebSocketSendTimeout(
        "server output sender stopped before event could be queued"
    )


async def _send_queued_events(
    websocket: Any,
    event_queue: asyncio.Queue[dict[str, Any] | None],
    *,
    send_timeout_sec: float = _SERVICE_SEND_TIMEOUT_SEC,
) -> None:
    while True:
        event = await event_queue.get()
        try:
            if event is None:
                return
            if _LOGGER.isEnabledFor(logging.DEBUG) and _should_log_realtime_event(
                event
            ):
                _LOGGER.debug("Realtime event %s", _format_event_log_summary(event))
            await _send_json_with_timeout(
                websocket, event, timeout_sec=send_timeout_sec
            )
        finally:
            event_queue.task_done()


async def _receive_or_sender_failed(
    websocket: Any, sender_task: asyncio.Task[None]
) -> dict[str, Any]:
    if sender_task.done():
        sender_task.result()
        return {"type": "websocket.disconnect"}

    receive_task = asyncio.create_task(websocket.receive())
    try:
        done, _pending = await asyncio.wait(
            {receive_task, sender_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sender_task in done:
            if not receive_task.done():
                receive_task.cancel()
                with suppress(asyncio.CancelledError):
                    await receive_task
            sender_task.result()
            return {"type": "websocket.disconnect"}
        return receive_task.result()
    except BaseException:
        if not receive_task.done():
            receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await receive_task
        raise


async def _send_json_with_timeout(
    websocket: Any, payload: dict[str, Any], *, timeout_sec: float
) -> None:
    try:
        await asyncio.wait_for(
            _send_json(websocket, payload),
            timeout=max(0.001, float(timeout_sec)),
        )
    except asyncio.TimeoutError as exc:
        raise WebSocketSendTimeout(
            f"client did not consume WebSocket output within {float(timeout_sec):.3f}s"
        ) from exc


async def _send_error_and_close(websocket: Any, error: str, *, code: int) -> None:
    await _send_json_with_timeout(
        websocket,
        {"type": "error", "error": str(error), "fatal": True},
        timeout_sec=_SERVICE_SEND_TIMEOUT_SEC,
    )
    await _close_websocket(websocket, code=code)


async def _close_websocket(
    websocket: Any, *, code: int, timeout_sec: float = _SERVICE_SEND_TIMEOUT_SEC
) -> None:
    with suppress(Exception):
        await asyncio.wait_for(
            websocket.close(code=code),
            timeout=max(0.001, float(timeout_sec)),
        )


async def _send_json(websocket: Any, payload: dict[str, Any]) -> None:
    await websocket.send_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3-ASR runtime realtime ASR WebSocket service."
    )
    parser.add_argument(
        "--model", required=True, help="Model path or Hugging Face model id."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=_LOG_LEVELS,
        help="Service log verbosity. Use debug to inspect PCM summaries, ASR events, and transcript updates.",
    )
    parser.add_argument(
        "--save-debug-audio",
        action="store_true",
        help=(
            "Save backend-received audio as 16 kHz mono WAV files. "
            f"Default directory: {_DEFAULT_DEBUG_AUDIO_DIR}."
        ),
    )
    parser.add_argument(
        "--debug-audio-dir",
        default=_DEFAULT_DEBUG_AUDIO_DIR,
        help="Directory used by --save-debug-audio.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "transformers", "mlx"],
        help="ASR backend. 'auto' picks mlx on Apple Silicon (Metal) and transformers (CUDA) "
        "elsewhere. 'mlx' forces the Apple Silicon backend; CUDA-only flags are ignored.",
    )
    parser.add_argument(
        "--device-map", default=None, help="Transformers device_map. Default: cuda:0."
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Compute dtype. Default: bfloat16.",
    )
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--flashinfer", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fused-rmsnorm", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fused-linears", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--w8a16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="W8A16 quant. Default OFF for streaming (its fp32 Triton GEMM slows "
        "prefill ~3x at equal CER); pass --w8a16 to force on.",
    )
    parser.add_argument(
        "--cuda-graph-prewarm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cuda-graph-prewarm-language", default="Chinese")
    parser.add_argument("--cuda-graph-prewarm-window-sec", type=float, default=20.0)
    parser.add_argument("--cuda-graph-prewarm-prefix-tokens", type=int, default=64)
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Pass all received audio to ASR instead of using VAD speech gating.",
    )
    parser.add_argument(
        "--timestamp-model",
        default=None,
        help="Required forced-aligner model for realtime ASR.",
    )
    parser.add_argument(
        "--timestamp-device-map",
        default=None,
        help="Forced-aligner device_map. Default: cuda:0.",
    )
    parser.add_argument(
        "--timestamp-dtype",
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for the forced-aligner model. Default: bfloat16.",
    )
    parser.add_argument("--timestamp-attn-implementation", default=None)
    parser.add_argument(
        "--timestamp-backend",
        default="auto",
        choices=["auto", "torch", "mlx"],
        help="Forced-aligner backend. 'auto' follows the ASR backend (mlx on Apple Silicon). "
        "'mlx' runs the aligner on Metal; CUDA-only flags (device_map, attn_implementation, fused) are ignored.",
    )
    parser.add_argument(
        "--timestamp-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply fused RMSNorm to the forced aligner (~1.4x per-segment align speedup; this is "
        "the whole win, the aligner is prefill-bound so linear fusion is not applied; bf16 argmax "
        "can shift <=~1%% of timestamps by <=0.16s, no word-count change).",
    )
    parser.add_argument(
        "--timestamp-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timestamp-pad-ms", type=int, default=500)
    parser.add_argument("--timestamp-finish-timeout-ms", type=int, default=30_000)
    parser.add_argument(
        "--translation-model",
        nargs="?",
        const=DEFAULT_HYMT_MODEL,
        default=None,
        help=(
            "Enable realtime translation with this model path or Hugging Face id. "
            f"If no value is provided, uses {DEFAULT_HYMT_MODEL}."
        ),
    )
    parser.add_argument(
        "--translation-model-revision",
        default=None,
        help="Pin the HF model to an immutable commit/revision; only valid for a HF id, not a local path.",
    )
    parser.add_argument("--translation-device", default="cuda:0")
    parser.add_argument(
        "--translation-backend",
        default="auto",
        choices=["auto", "torch", "mlx"],
        help="Translation backend. 'auto' follows the ASR backend (mlx on Apple Silicon). "
        "'mlx' runs HY-MT on Metal; CUDA-only flags (device, w8a16, fused_rmsnorm, decode_backend) are ignored.",
    )
    parser.add_argument(
        "--translation-dtype",
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--translation-preview", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--translation-preview-debounce-ms",
        type=int,
        default=_SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS,
    )
    parser.add_argument("--translation-preview-timeout-ms", type=int, default=30_000)
    parser.add_argument(
        "--translation-max-new-tokens",
        type=int,
        default=_SERVICE_TRANSLATION_MAX_NEW_TOKENS,
    )
    parser.add_argument("--translation-stable-timeout-ms", type=int, default=30_000)
    parser.add_argument("--translation-stable-batch-size", type=int, default=1)
    parser.add_argument(
        "--translation-prewarm-target-language",
        action="append",
        default=None,
        help=(
            "Target language to prewarm at service startup. Repeat or comma-separate "
            "to prewarm multiple targets; defaults to Chinese and English for the "
            "local single-user profile."
        ),
    )
    parser.add_argument(
        "--translation-sample", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--translation-decode-backend",
        default=DEFAULT_HYMT_DECODE_BACKEND,
        choices=["fixed_mask", "generate"],
    )
    parser.add_argument(
        "--translation-attn-implementation", default=DEFAULT_HYMT_ATTN_IMPLEMENTATION
    )
    parser.add_argument(
        "--translation-w8a16",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HYMT_W8A16,
        help="Apply W8A16 to HY-MT gate/up linears. Enabled by default for the "
        "validated Hy-MT2 translation profile; pass --no-translation-w8a16 to disable.",
    )
    parser.add_argument(
        "--translation-fused-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HYMT_FUSED_RMSNORM,
        help="Apply fused F.rms_norm to HY-MT. Enabled by default for the validated "
        "Hy-MT2 translation profile; pass --no-translation-fused-rmsnorm to disable.",
    )
    parser.add_argument(
        "--translation-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--translation-trust-remote-code", action="store_true")
    return parser.parse_args()


def _configure_logging(level_name: str) -> int:
    normalized = str(level_name or "info").upper()
    level = int(getattr(logging, normalized))
    root_level = logging.INFO if level <= logging.DEBUG else level
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(root_level)
    _LOGGER.setLevel(level)
    logging.getLogger("qwen3_asr_runtime").setLevel(level)
    logging.getLogger("uvicorn").setLevel(root_level)
    logging.getLogger("websockets").setLevel(root_level)
    return level


def _uvicorn_log_level(service_log_level: int) -> str:
    if int(service_log_level) <= logging.DEBUG:
        return "info"
    return logging.getLevelName(service_log_level).lower()


def _prefer_mlx() -> bool:
    """True on Apple Silicon with MLX installed (the macOS Metal path)."""
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_service_backends(args: argparse.Namespace) -> None:
    """Resolve ``auto`` backend choices to concrete values for ASR + translation + aligner.

    ``auto`` -> mlx on Apple Silicon (Metal), else the framework default
    (transformers for ASR, torch for translation/aligner). Concrete ``mlx`` /
    ``transformers`` / ``torch`` choices are honored as-is. Mutates ``args`` so all
    downstream builders read a concrete backend.
    """
    prefer = _prefer_mlx()

    def resolve_asr(choice: str) -> str:
        choice = str(choice or "auto").lower()
        if choice == "auto":
            return "mlx" if prefer else "transformers"
        if choice in {"mlx", "transformers"}:
            return choice
        return "transformers"

    def resolve_aux(choice: str) -> str:
        choice = str(choice or "auto").lower()
        if choice in {"mlx", "torch"}:
            return choice
        if choice == "auto":
            return "mlx" if args.backend == "mlx" else "torch"
        return "torch"

    args.backend = resolve_asr(args.backend)
    args.translation_backend = resolve_aux(getattr(args, "translation_backend", "auto"))
    args.timestamp_backend = resolve_aux(getattr(args, "timestamp_backend", "auto"))


def _build_model_load(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.backend == "mlx":
        # MLX (Apple Silicon): no device_map/cuda_graph/flashinfer/fused/W8A16 -- the
        # backend silently drops them. Pass only the compute dtype.
        dtype_name = args.dtype or "bfloat16"
        kwargs: dict[str, Any] = {}
        if dtype_name != "auto":
            kwargs["dtype"] = dtype_name
        return "mlx", kwargs

    import torch

    device_map = args.device_map or "cuda:0"
    dtype_name = args.dtype or "bfloat16"
    dtype = None
    if dtype_name != "auto":
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]

    load_kwargs: dict[str, Any] = {
        "cuda_graph": True if args.cuda_graph is None else args.cuda_graph,
        "cuda_graph_len_bucket": 64,
        "flashinfer": True if args.flashinfer is None else args.flashinfer,
        "fused_rmsnorm": True if args.fused_rmsnorm is None else args.fused_rmsnorm,
        "fused_linears": True if args.fused_linears is None else args.fused_linears,
        # W8A16 is OFF by default for the streaming service: its Triton GEMM
        # (fp32 tl.dot) makes multi-token prefill ~3x slower, and streaming is
        # prefill-bound (decode is ~14% of a live20 step). The 80-window live20
        # CER gate shows OFF is 2.58x faster at equal CER (cer_mean 0.0961 vs
        # 0.0965; see local_goldens/cer/recheck_w8a16_{on,off}.json). W8A16
        # still helps the decode-bound offline path, where it stays opt-in.
        "quantized_linears": False if args.w8a16 is None else args.w8a16,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    return "transformers", load_kwargs


def _build_translation(
    args: argparse.Namespace,
) -> tuple[Any | None, TranslationServiceConfig | None]:
    model_path = str(args.translation_model or "").strip()
    if not model_path:
        return None, None
    dtype = None if args.translation_dtype in {None, "auto"} else args.translation_dtype
    config = TranslationServiceConfig(
        preview_enabled=bool(args.translation_preview),
        preview_debounce_ms=int(args.translation_preview_debounce_ms),
        preview_timeout_ms=int(args.translation_preview_timeout_ms),
        stable_timeout_ms=int(args.translation_stable_timeout_ms),
        max_new_tokens=int(args.translation_max_new_tokens),
        stable_batch_size=int(args.translation_stable_batch_size),
        prewarm_target_languages=_translation_prewarm_target_languages(
            args.translation_prewarm_target_language
        ),
    )
    generation_config = HYMTGenerationConfig(
        max_new_tokens=int(args.translation_max_new_tokens),
        do_sample=bool(args.translation_sample),
    )
    if args.translation_backend == "mlx":
        # MLX (Apple Silicon): ground-up Hunyuan forward; CUDA-only flags ignored.
        from qwen3_asr_runtime.mlx_translation import MLXHYMTTranslator

        translator = MLXHYMTTranslator(
            model_path,
            dtype=dtype,
            local_files_only=bool(args.translation_local_files_only),
            model_revision=args.translation_model_revision,
            generation_config=generation_config,
        )
        return translator, config
    translator = HYMTTranslator(
        model_path,
        device=str(args.translation_device),
        dtype=dtype,
        model_revision=args.translation_model_revision,
        local_files_only=bool(args.translation_local_files_only),
        trust_remote_code=bool(args.translation_trust_remote_code),
        attn_implementation=args.translation_attn_implementation,
        decode_backend=args.translation_decode_backend,
        generation_config=generation_config,
        w8a16=bool(args.translation_w8a16),
        fused_rmsnorm=bool(args.translation_fused_rmsnorm),
    )
    return translator, config


def _build_aligner(
    args: argparse.Namespace,
) -> tuple[Any | None, RealtimeTimestampConfig | None]:
    """Construct the forced aligner (not the actor). The caller wraps it in a
    TimestampModelActor bound to the executor the model was loaded on -- the MLX
    backend ties a model to its loading thread."""
    model_path = str(args.timestamp_model or "").strip()
    if not model_path:
        return None, None

    if args.timestamp_backend == "mlx":
        # MLX (Apple Silicon): ground-up aligner forward; CUDA-only flags ignored.
        from qwen3_asr_runtime.mlx_forced_aligner import MLXForcedAlignerBackend

        dtype_name = (
            args.timestamp_dtype
            if args.timestamp_dtype not in {None, "auto"}
            else "bfloat16"
        )
        aligner = MLXForcedAlignerBackend.from_pretrained(
            model_path,
            dtype=dtype_name,
            local_files_only=bool(args.timestamp_local_files_only),
        )
        config = RealtimeTimestampConfig(
            pad_ms=int(args.timestamp_pad_ms),
            finish_timeout_ms=int(args.timestamp_finish_timeout_ms),
        )
        return aligner, config

    import torch
    from qwen3_asr_runtime.forced_aligner import Qwen3ForcedAlignerBackend

    dtype_name = args.timestamp_dtype or "bfloat16"
    dtype = None
    if dtype_name != "auto":
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]

    load_kwargs: dict[str, Any] = {
        "local_files_only": bool(args.timestamp_local_files_only),
    }
    device_map = args.timestamp_device_map or "cuda:0"
    if device_map:
        load_kwargs["device_map"] = device_map
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    if args.timestamp_attn_implementation:
        load_kwargs["attn_implementation"] = args.timestamp_attn_implementation
    # Fused RMSNorm on the aligner: ~1.4x per-segment align speedup (interleaved
    # A/B; ~1.39x on short realtime segments), timestamp drift <=0.16s on <=~1% of
    # boundaries with no word-count change. fused_linears is intentionally NOT
    # applied: the aligner is prefill-bound, so linear fusion is launch-overhead-only
    # (helps ~4% on short segments, net-neutral-to-negative on long windows) and
    # carries none of the real win. On by default.
    load_kwargs["fused_rmsnorm"] = bool(args.timestamp_fused)

    aligner = Qwen3ForcedAlignerBackend.from_pretrained(model_path, **load_kwargs)
    config = RealtimeTimestampConfig(
        pad_ms=int(args.timestamp_pad_ms),
        finish_timeout_ms=int(args.timestamp_finish_timeout_ms),
    )
    return aligner, config


def _prewarm_translation_runtime(
    actor: TranslationModelActor,
    config: TranslationServiceConfig,
) -> None:
    for target_language in config.prewarm_target_languages:
        _prewarm_translation_target(actor, config, target_language)


def _prewarm_translation_target(
    actor: TranslationModelActor,
    config: TranslationServiceConfig,
    target_language: str,
) -> None:
    target = str(target_language or "").strip()
    if not target:
        return
    stable_batch_size = int(config.stable_batch_size)
    batch_sizes = (1,) if stable_batch_size == 1 else (1, stable_batch_size)
    for batch_size in batch_sizes:
        started = time.perf_counter()
        _LOGGER.info(
            "Prewarming translation model target_language=%s batch_size=%d texts=%d",
            target,
            batch_size,
            len(_SERVICE_TRANSLATION_PREWARM_TEXTS),
        )
        warmup_once = getattr(actor, "warmup_once", None)
        warmup = (
            warmup_once if warmup_once is not None else getattr(actor, "warmup", None)
        )
        if warmup is None:
            raise RuntimeError(
                "translation prewarm was requested but this actor does not support warmup"
            )
        warmup(
            _SERVICE_TRANSLATION_PREWARM_TEXTS,
            target_language=target,
            source_language="",
            max_new_tokens=config.max_new_tokens,
            sync_cuda=True,
            batch_size=batch_size,
            **(
                {
                    "cache_key": (
                        "service_translation_target",
                        target,
                        int(config.max_new_tokens)
                        if config.max_new_tokens is not None
                        else None,
                        int(batch_size),
                    )
                }
                if warmup_once is not None
                else {}
            ),
        )
        _LOGGER.info(
            "Prewarmed translation model target_language=%s batch_size=%d wall_ms=%d",
            target,
            batch_size,
            int(round((time.perf_counter() - started) * 1000)),
        )


async def _ensure_translation_target_prewarmed(
    actor: TranslationModelActor | None,
    config: TranslationServiceConfig | None,
    target_language: str | None,
) -> None:
    target = str(target_language or "").strip()
    if actor is None or config is None or not target:
        return
    if not hasattr(actor, "warmup_once") and not hasattr(actor, "warmup"):
        return
    await asyncio.to_thread(_prewarm_translation_target, actor, config, target)


def _timestamp_prewarm_audio(
    duration_sec: float = _SERVICE_TIMESTAMP_PREWARM_DURATION_SEC,
) -> np.ndarray:
    sample_count = max(1, int(round(float(duration_sec) * SAMPLE_RATE)))
    t = np.arange(sample_count, dtype=np.float32) / float(SAMPLE_RATE)
    return (0.01 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def _prewarm_timestamp_runtime(actor: TimestampModelActor) -> None:
    started = time.perf_counter()
    _LOGGER.info(
        "Prewarming timestamp model language=%s duration_ms=%d",
        _SERVICE_TIMESTAMP_PREWARM_LANGUAGE,
        int(round(_SERVICE_TIMESTAMP_PREWARM_DURATION_SEC * 1000)),
    )
    actor.warmup(
        _timestamp_prewarm_audio(),
        text=_SERVICE_TIMESTAMP_PREWARM_TEXT,
        language=_SERVICE_TIMESTAMP_PREWARM_LANGUAGE,
    )
    _LOGGER.info(
        "Prewarmed timestamp model language=%s wall_ms=%d",
        _SERVICE_TIMESTAMP_PREWARM_LANGUAGE,
        int(round((time.perf_counter() - started) * 1000)),
    )


def _cuda_graph_enabled(args: argparse.Namespace) -> bool:
    return True if args.cuda_graph is None else bool(args.cuda_graph)


def _prewarm_realtime_cuda_graph(model: Any, args: argparse.Namespace) -> bool:
    prewarm = getattr(model, "prewarm_realtime_cuda_graph", None)
    if prewarm is None:
        return False
    return bool(
        prewarm(
            language=args.cuda_graph_prewarm_language,
            max_window_sec=float(args.cuda_graph_prewarm_window_sec),
            max_prefix_tokens=int(args.cuda_graph_prewarm_prefix_tokens),
        )
    )


def _prepare_cuda_graph_runtime(model: Any, args: argparse.Namespace) -> None:
    if not _cuda_graph_enabled(args):
        return
    if args.cuda_graph_prewarm:
        started = time.perf_counter()
        _LOGGER.info(
            "Prewarming ASR cuda graph language=%s window_sec=%.1f prefix_tokens=%d",
            args.cuda_graph_prewarm_language,
            float(args.cuda_graph_prewarm_window_sec),
            int(args.cuda_graph_prewarm_prefix_tokens),
        )
        if not _prewarm_realtime_cuda_graph(model, args):
            raise RuntimeError(
                "cuda graph prewarm was requested but this backend does not support it"
            )
        _LOGGER.info(
            "Prewarmed ASR cuda graph language=%s wall_ms=%d",
            args.cuda_graph_prewarm_language,
            int(round((time.perf_counter() - started) * 1000)),
        )
        return


def _prewarm_mlx_asr(model: Any) -> None:
    """Compile the MLX ASR Metal kernels with a short silent transcribe.

    The cuda-graph prewarm does not apply to MLX; without this the first real audio
    chunk would pay the one-time Metal kernel compilation (seconds).
    """
    started = time.perf_counter()
    _LOGGER.info("Prewarming MLX ASR (compiling Metal kernels)")
    silence = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    model.transcribe((silence, SAMPLE_RATE))
    _LOGGER.info(
        "Prewarmed MLX ASR wall_ms=%d",
        int(round((time.perf_counter() - started) * 1000)),
    )


def _translation_capture_lock(
    args: argparse.Namespace, *, translation_enabled: bool
) -> Any | None:
    if not translation_enabled:
        return None
    if _cuda_graph_enabled(args) and not args.cuda_graph_prewarm:
        return CUDA_GRAPH_CAPTURE_LOCK
    return None


def _load_on_thread(
    build: Any,
    *,
    use_executor: bool,
    thread_name: str,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[Any, ThreadPoolExecutor | None]:
    """Build a model, on a single-worker executor when ``use_executor``.

    MLX ties a model to the thread that loaded it, and the service runs each model's
    forwards on its own worker; so an MLX model must be built on the executor that will
    run it. Returns ``(result, executor_or_None)`` -- the executor the model was loaded
    on (reuse it for forwards), or ``None`` for an inline (main-thread) build.
    """
    if not use_executor:
        return build(), None
    ex = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
    return ex.submit(build).result(), ex


def main() -> None:
    from qwen3_asr_runtime import Qwen3ASRModel

    args = _parse_args()
    log_level = _configure_logging(args.log_level)
    if not str(args.timestamp_model or "").strip():
        raise RuntimeError(
            "--timestamp-model is required; realtime ASR commits require ASR and forced aligner together."
        )
    _resolve_service_backends(args)
    if args.backend == "mlx":
        # MLX has no cuda graph; force the flag off so the cuda-graph prewarm and the
        # translation capture lock are skipped (the prewarm would otherwise raise).
        args.cuda_graph = False
    backend, load_kwargs = _build_model_load(args)

    _LOGGER.info(
        "Loading Qwen3-ASR model model=%s backend=%s device_map=%s log_level=%s",
        args.model,
        backend,
        load_kwargs.get("device_map"),
        args.log_level,
    )
    translation_actor: TranslationModelActor | None = None
    translation_service_config: TranslationServiceConfig | None = None
    timestamp_actor: TimestampModelActor | None = None
    timestamp_config: RealtimeTimestampConfig | None = None
    translation_executor: ThreadPoolExecutor | None = None
    timestamp_executor: ThreadPoolExecutor | None = None
    # Single-worker ASR executor: realtime ASR forwards run here, off the asyncio
    # event loop, so a ~55ms decode step no longer blocks event delivery, audio
    # ingest, or the aligner/translation runtimes. The CUDA-graph prewarm runs on
    # this same thread so graph capture and replay share one thread/stream. MLX ties
    # a model to its loading thread, so each MLX model is BOTH loaded and run on its
    # own single-worker executor.
    asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen3-asr")
    try:
        model, _ = _load_on_thread(
            lambda: Qwen3ASRModel.from_pretrained(
                args.model, backend=backend, **load_kwargs
            ),
            use_executor=(backend == "mlx"),
            executor=asr_executor,
            thread_name="qwen3-asr",
        )
        asr_executor.submit(_prepare_cuda_graph_runtime, model, args).result()
        if backend == "mlx":
            asr_executor.submit(_prewarm_mlx_asr, model).result()

        (translator, translation_service_config), translation_executor = (
            _load_on_thread(
                lambda: _build_translation(args),
                use_executor=(args.translation_backend == "mlx"),
                thread_name="hymt-translation",
            )
        )
        if translator is not None:
            translation_actor = TranslationModelActor(
                translator,
                capture_lock=_translation_capture_lock(args, translation_enabled=True),
                executor=translation_executor,
            )
        elif translation_executor is not None:
            translation_executor.shutdown(wait=False)
            translation_executor = None

        (aligner, timestamp_config), timestamp_executor = _load_on_thread(
            lambda: _build_aligner(args),
            use_executor=(args.timestamp_backend == "mlx"),
            thread_name="qwen3-timestamp",
        )
        if aligner is not None:
            timestamp_actor = TimestampModelActor(aligner, executor=timestamp_executor)
        elif timestamp_executor is not None:
            timestamp_executor.shutdown(wait=False)
            timestamp_executor = None
        if translation_actor is not None and translation_service_config is not None:
            _prewarm_translation_runtime(translation_actor, translation_service_config)
        if timestamp_actor is not None:
            _prewarm_timestamp_runtime(timestamp_actor)
    except Exception:
        if translation_actor is not None:
            translation_actor.close(wait=True)
        if timestamp_actor is not None:
            timestamp_actor.close(wait=True)
        for ex in (translation_executor, timestamp_executor):
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
        asr_executor.shutdown(wait=False, cancel_futures=True)
        raise
    app = build_app(
        model=model,
        asr_executor=asr_executor,
        timestamp_actor=timestamp_actor,
        timestamp_config=timestamp_config,
        translation_actor=translation_actor,
        translation_service_config=translation_service_config,
        debug_audio_dir=args.debug_audio_dir if args.save_debug_audio else None,
        no_vad=args.no_vad,
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Install service dependencies with: uv sync --python 3.12"
        ) from exc

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ws_ping_interval=None,
            log_level=_uvicorn_log_level(log_level),
        )
    finally:
        for ex in (translation_executor, timestamp_executor):
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
        asr_executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    main()
