# coding=utf-8

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import json
import logging
from typing import Any, Callable

import numpy as np

from qwen3_asr_runtime.cuda_serialization import CUDA_GRAPH_CAPTURE_LOCK
from qwen3_asr_runtime.realtime_timestamps import (
    AudioTimelineBuffer,
    RealtimeTimestampConfig,
    RealtimeTimestampRuntime,
    StableTimingJob,
    TimestampModelActor,
)
from qwen3_asr_runtime.realtime_translation import (
    RealtimeTranslationConfig,
    RealtimeTranslationRuntime,
    TranslationModelActor,
)
from qwen3_asr_runtime.language_support import (
    HYMT_MODEL_CARD_LANGUAGES,
    QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES,
)
from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_MAX_NEW_TOKENS,
    DEFAULT_HYMT_MODEL,
    HYMTGenerationConfig,
    HYMTTranslator,
)
from qwen3_asr_runtime.realtime_session import RealtimeASRConfig, RealtimeASRSession
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.utils import SAMPLE_RATE, normalize_language_name, validate_language

_SERVICE_SEND_TIMEOUT_SEC = 5.0
_SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS = 700
_SERVICE_LIVE_STABILITY_DELAY_MS = 12_000
_START_COMMAND_FIELDS = frozenset(
    {
        "type",
        "session_id",
        "sample_rate",
        "audio_format",
        "language",
        "context",
        "target_language",
    }
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationServiceConfig:
    preview_enabled: bool = True
    preview_debounce_ms: int = _SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS
    preview_timeout_ms: int = 30_000
    max_new_tokens: int | None = None
    stable_batch_size: int = 1

    def __post_init__(self) -> None:
        if int(self.preview_debounce_ms) < 0:
            raise ValueError("preview_debounce_ms must be >= 0")
        if int(self.preview_timeout_ms) <= 0:
            raise ValueError("preview_timeout_ms must be > 0")
        if int(self.stable_batch_size) <= 0:
            raise ValueError("stable_batch_size must be > 0")


class WebSocketSendTimeout(RuntimeError):
    """Raised when a connected client stops consuming server output."""


def build_app(
    *,
    model: Any,
    timestamp_actor: TimestampModelActor | None = None,
    timestamp_config: RealtimeTimestampConfig | None = None,
    translation_actor: TranslationModelActor | None = None,
    translation_service_config: TranslationServiceConfig | None = None,
    live_stability_delay_ms: int = _SERVICE_LIVE_STABILITY_DELAY_MS,
) -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install service dependencies with: uv sync --python 3.12") from exc

    lifespan = None
    if translation_actor is not None or timestamp_actor is not None:

        @asynccontextmanager
        async def model_actor_lifespan(app: Any) -> Any:
            del app
            try:
                yield
            finally:
                if translation_actor is not None:
                    translation_actor.close(wait=False)
                if timestamp_actor is not None:
                    timestamp_actor.close(wait=False)

        lifespan = model_actor_lifespan

    app = FastAPI(title="Qwen3-ASR Runtime Realtime ASR Service", lifespan=lifespan)
    active_lock = asyncio.Lock()
    active_connection = {"open": False}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/asr")
    async def websocket_asr(websocket: WebSocket) -> None:
        await websocket.accept()
        reject_connection = False
        async with active_lock:
            if active_connection["open"]:
                reject_connection = True
            else:
                active_connection["open"] = True
        if reject_connection:
            with suppress(WebSocketSendTimeout):
                await _send_error_and_close(websocket, "Another realtime session is active.", code=1013)
            await _close_websocket(websocket, code=1013)
            return

        try:
            start_payload = await _receive_start(websocket)
            if start_payload is None:
                return
            session_id = str(start_payload.get("session_id") or "default")
            store = TranscriptStore(transcript_id=session_id)
            store_lock = asyncio.Lock()
            timestamps_enabled = timestamp_actor is not None and timestamp_config is not None
            try:
                _validate_timestamp_start_language(start_payload, timestamps_enabled=timestamps_enabled)
            except ValueError as exc:
                await _send_error_and_close(websocket, str(exc), code=1003)
                return
            config = _build_session_config(
                start_payload,
                force_align_timestamps=timestamps_enabled,
                live_stability_delay_ms=live_stability_delay_ms,
            )
            try:
                session_translation_config = _session_translation_config(start_payload, translation_service_config)
            except ValueError as exc:
                await _send_error_and_close(websocket, str(exc), code=1003)
                return
            try:
                session = RealtimeASRSession(model, transcript_store=store, config=config)
            except RuntimeError as exc:
                await _send_error_and_close(websocket, str(exc), code=1011)
                return

            event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            sender_task = asyncio.create_task(_send_queued_events(websocket, event_queue))
            timestamp_runtime: RealtimeTimestampRuntime | None = None
            if timestamps_enabled and timestamp_actor is not None and timestamp_config is not None:
                timestamp_runtime = RealtimeTimestampRuntime(
                    timestamp_actor,
                    store=store,
                    audio_buffer=AudioTimelineBuffer(),
                    config=timestamp_config,
                    event_queue=event_queue,
                    store_lock=store_lock,
                )
                await timestamp_runtime.start()
            translation: RealtimeTranslationRuntime | None = None
            if translation_actor is not None and session_translation_config is not None:
                translation = RealtimeTranslationRuntime(
                    translation_actor,
                    config=session_translation_config,
                    event_queue=event_queue,
                )
                await translation.start()

            ready: dict[str, Any] = {
                "type": "ready",
                "session_id": session_id,
                "sample_rate": SAMPLE_RATE,
                "audio_format": "pcm_s16le",
            }
            if timestamp_runtime is not None:
                ready["timestamps"] = timestamp_runtime.ready_payload()
            if translation is not None:
                ready["translation"] = translation.ready_payload()
            await event_queue.put(ready)

            while True:
                message = await _receive_or_sender_failed(websocket, sender_task)
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    audio = decode_pcm_s16le(message["bytes"])
                    if timestamp_runtime is not None:
                        timestamp_runtime.accept_audio(audio)
                    events = await _run_store_write(store_lock, session.ingest_audio, audio)
                    await _publish_session_events(event_queue, translation, events, timestamp_runtime, session)
                    continue

                if message.get("text") is None:
                    continue

                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    await event_queue.put({"type": "error", "error": "Invalid JSON command."})
                    continue
                if not isinstance(command, dict):
                    await event_queue.put({"type": "error", "error": "Command must be a JSON object."})
                    continue
                command_type = command.get("type")
                if command_type == "flush":
                    events = await _run_store_write(store_lock, session.flush)
                    await _publish_session_events(event_queue, translation, events, timestamp_runtime, session)
                elif command_type == "finish":
                    if timestamp_runtime is None:
                        events = await _run_store_write(store_lock, session.finish)
                        await _publish_finish_events(event_queue, translation, events)
                    else:
                        events = await _run_store_write(store_lock, session.flush)
                        await _publish_finish_events(
                            event_queue,
                            translation,
                            events,
                            timestamp_runtime,
                            session.stable_timing_jobs_for_events(events),
                            final_event_factory=store.final_event,
                            store_lock=store_lock,
                        )
                    await event_queue.put(None)
                    await sender_task
                    await _close_websocket(websocket, code=1000)
                    return
                else:
                    await event_queue.put({"type": "error", "error": f"Unsupported command: {command_type}"})
        except WebSocketDisconnect:
            return
        except WebSocketSendTimeout:
            _LOGGER.warning("Realtime ASR WebSocket client stopped consuming output.")
            await _close_websocket(websocket, code=1011)
            return
        except Exception as exc:
            _LOGGER.exception("Realtime ASR WebSocket session failed.")
            try:
                if "event_queue" in locals() and "sender_task" in locals() and not sender_task.done():
                    await event_queue.put({"type": "error", "error": str(exc) or type(exc).__name__})
                    await event_queue.put(None)
                    await sender_task
                    await _close_websocket(websocket, code=1011)
                else:
                    await _send_error_and_close(websocket, str(exc) or type(exc).__name__, code=1011)
            except Exception:
                _LOGGER.exception("Failed to send realtime ASR error response.")
            return
        finally:
            if "timestamp_runtime" in locals() and timestamp_runtime is not None:
                await timestamp_runtime.close()
            if "translation" in locals() and translation is not None:
                await translation.close()
            if "sender_task" in locals() and not sender_task.done():
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
            elif "sender_task" in locals():
                try:
                    sender_task.result()
                except Exception:
                    pass
            async with active_lock:
                active_connection["open"] = False

    return app


def decode_pcm_s16le(payload: bytes) -> np.ndarray:
    if len(payload) % 2 != 0:
        payload = payload[:-1]
    if not payload:
        return np.zeros((0,), dtype=np.int16)
    return np.frombuffer(payload, dtype="<i2").copy()


async def _run_store_write(store_lock: asyncio.Lock, func: Callable[..., Any], *args: Any) -> Any:
    async with store_lock:
        return await asyncio.to_thread(func, *args)


def _build_session_config(
    payload: dict[str, Any],
    *,
    force_align_timestamps: bool = False,
    live_stability_delay_ms: int = _SERVICE_LIVE_STABILITY_DELAY_MS,
) -> RealtimeASRConfig:
    return RealtimeASRConfig(
        context=str(payload.get("context") or ""),
        language=payload.get("language"),
        live_stability_delay_ms=int(live_stability_delay_ms),
        force_align_timestamps=bool(force_align_timestamps),
    )


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
    return RealtimeTranslationConfig(
        target_language=normalized_target,
        preview_enabled=service_config.preview_enabled,
        preview_debounce_ms=service_config.preview_debounce_ms,
        preview_timeout_ms=service_config.preview_timeout_ms,
        max_new_tokens=service_config.max_new_tokens,
        stable_batch_size=service_config.stable_batch_size,
    )


def _normalize_supported_language(language: str) -> str:
    normalized = normalize_language_name(str(language))
    validate_language(normalized)
    return normalized


def _normalize_language_choice(language: str, allowed: tuple[str, ...], *, field_name: str) -> str:
    raw = str(language or "").strip()
    if not raw:
        raise ValueError(f"{field_name} is empty")
    by_casefold = {item.casefold(): item for item in allowed}
    normalized = by_casefold.get(raw.casefold())
    if normalized is None:
        raise ValueError(f"Unsupported {field_name}: {raw}. Supported: {list(allowed)}")
    return normalized


def _normalize_translation_target_language(language: str) -> str:
    return _normalize_language_choice(language, HYMT_MODEL_CARD_LANGUAGES, field_name="target_language")


def _validate_timestamp_start_language(payload: dict[str, Any], *, timestamps_enabled: bool) -> None:
    if not timestamps_enabled:
        return
    language = str(payload.get("language") or "").strip()
    if not language:
        return
    if language not in QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES:
        raise ValueError(
            "language must be one of "
            f"{list(QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES)} when forced-aligner timestamps are enabled, "
            f"got: {language}"
        )


def _arg_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer.")
    return parsed


async def _receive_start(websocket: Any) -> dict[str, Any] | None:
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        return None
    if message.get("text") is None:
        await _send_error_and_close(websocket, "First frame must be a JSON start command.", code=1003)
        return None
    try:
        payload = json.loads(message["text"])
    except json.JSONDecodeError:
        await _send_error_and_close(websocket, "Start command must be valid JSON.", code=1003)
        return None
    if not isinstance(payload, dict):
        await _send_error_and_close(websocket, "Start command must be a JSON object.", code=1003)
        return None
    if payload.get("type") != "start":
        await _send_error_and_close(websocket, "First command must be type=start.", code=1003)
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
            payload["language"] = _normalize_supported_language(str(raw_language))
        except ValueError as exc:
            await _send_error_and_close(websocket, str(exc), code=1003)
            return None
    return payload


async def _publish_session_events(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    translation: RealtimeTranslationRuntime | None,
    events: list[dict[str, Any]],
    timestamp_runtime: RealtimeTimestampRuntime | None = None,
    session: RealtimeASRSession | None = None,
) -> None:
    for event in events:
        if translation is not None:
            await translation.accept_source_event(event)
        await event_queue.put(event)
        if timestamp_runtime is not None and session is not None:
            await timestamp_runtime.accept_jobs(session.stable_timing_jobs(event))


async def _publish_finish_events(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    translation: RealtimeTranslationRuntime | None,
    events: list[dict[str, Any]],
    timestamp_runtime: RealtimeTimestampRuntime | None = None,
    timestamp_jobs: list[StableTimingJob] | None = None,
    *,
    final_event_factory: Any | None = None,
    store_lock: asyncio.Lock | None = None,
) -> None:
    if translation is not None:
        await translation.cancel_preview()

    transcript_updates: list[dict[str, Any]] = []
    final_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "transcript_update":
            transcript_updates.append(event)
            await event_queue.put(event)
        elif event.get("type") == "transcript_final":
            final_events.append(event)
        else:
            await event_queue.put(event)

    if timestamp_runtime is not None:
        timing_events = await timestamp_runtime.finish(list(timestamp_jobs or []))
        for event in timing_events:
            await event_queue.put(event)

    if translation is not None:
        translation_events = await translation.finish(transcript_updates)
        for event in translation_events:
            await event_queue.put(event)

    if final_event_factory is not None:
        if store_lock is None:
            final_events = [final_event_factory()]
        else:
            async with store_lock:
                final_events = [final_event_factory()]
    for event in final_events:
        await event_queue.put(event)


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
            await _send_json_with_timeout(websocket, event, timeout_sec=send_timeout_sec)
        finally:
            event_queue.task_done()


async def _receive_or_sender_failed(websocket: Any, sender_task: asyncio.Task[None]) -> dict[str, Any]:
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


async def _send_json_with_timeout(websocket: Any, payload: dict[str, Any], *, timeout_sec: float) -> None:
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
        {"type": "error", "error": str(error)},
        timeout_sec=_SERVICE_SEND_TIMEOUT_SEC,
    )
    await _close_websocket(websocket, code=code)


async def _close_websocket(websocket: Any, *, code: int, timeout_sec: float = _SERVICE_SEND_TIMEOUT_SEC) -> None:
    with suppress(Exception):
        await asyncio.wait_for(
            websocket.close(code=code),
            timeout=max(0.001, float(timeout_sec)),
        )


async def _send_json(websocket: Any, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3-ASR runtime realtime ASR WebSocket service.")
    parser.add_argument("--model", required=True, help="Model path or Hugging Face model id.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-map", default=None, help="Transformers device_map. Default: cuda:0.")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for the transformers backend. Default: bfloat16.",
    )
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flashinfer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-rmsnorm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-linears", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--w8a16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cuda-graph-prewarm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph-prewarm-language", default="Chinese")
    parser.add_argument("--cuda-graph-prewarm-window-sec", type=float, default=20.0)
    parser.add_argument("--cuda-graph-prewarm-prefix-tokens", type=int, default=64)
    parser.add_argument(
        "--live-stability-delay-ms",
        type=_arg_nonnegative_int,
        default=_SERVICE_LIVE_STABILITY_DELAY_MS,
        help=(
            "Minimum repeated-prefix delay before live text becomes stable. "
            "Lower values are more aggressive and can reduce transcript quality."
        ),
    )
    parser.add_argument("--timestamp-model", default=None, help="Enable forced-aligner timestamps with this model.")
    parser.add_argument("--timestamp-device-map", default=None, help="Forced-aligner device_map. Default: cuda:0.")
    parser.add_argument(
        "--timestamp-dtype",
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for the forced-aligner model. Default: bfloat16.",
    )
    parser.add_argument("--timestamp-attn-implementation", default=None)
    parser.add_argument("--timestamp-local-files-only", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--translation-device", default="cuda:0")
    parser.add_argument("--translation-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--translation-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--translation-preview-debounce-ms", type=int, default=_SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS)
    parser.add_argument("--translation-preview-timeout-ms", type=int, default=30_000)
    parser.add_argument("--translation-max-new-tokens", type=int, default=DEFAULT_HYMT_MAX_NEW_TOKENS)
    parser.add_argument("--translation-stable-batch-size", type=int, default=1)
    parser.add_argument("--translation-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--translation-decode-backend", default=DEFAULT_HYMT_DECODE_BACKEND, choices=["fixed_mask", "generate"])
    parser.add_argument("--translation-attn-implementation", default=DEFAULT_HYMT_ATTN_IMPLEMENTATION)
    parser.add_argument("--translation-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--translation-trust-remote-code", action="store_true")
    return parser.parse_args()


def _build_model_load(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
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
        "quantized_linears": True if args.w8a16 is None else args.w8a16,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    return "transformers", load_kwargs


def _build_translation(args: argparse.Namespace) -> tuple[Any | None, TranslationServiceConfig | None]:
    model_path = str(args.translation_model or "").strip()
    if not model_path:
        return None, None
    dtype = None if args.translation_dtype in {None, "auto"} else args.translation_dtype
    config = TranslationServiceConfig(
        preview_enabled=bool(args.translation_preview),
        preview_debounce_ms=int(args.translation_preview_debounce_ms),
        preview_timeout_ms=int(args.translation_preview_timeout_ms),
        max_new_tokens=int(args.translation_max_new_tokens),
        stable_batch_size=int(args.translation_stable_batch_size),
    )
    generation_config = HYMTGenerationConfig(
        max_new_tokens=int(args.translation_max_new_tokens),
        do_sample=bool(args.translation_sample),
    )
    translator = HYMTTranslator(
        model_path,
        device=str(args.translation_device),
        dtype=dtype,
        local_files_only=bool(args.translation_local_files_only),
        trust_remote_code=bool(args.translation_trust_remote_code),
        attn_implementation=args.translation_attn_implementation,
        decode_backend=args.translation_decode_backend,
        generation_config=generation_config,
    )
    return translator, config


def _build_timestamp_actor(args: argparse.Namespace) -> tuple[TimestampModelActor | None, RealtimeTimestampConfig | None]:
    model_path = str(args.timestamp_model or "").strip()
    if not model_path:
        return None, None

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

    aligner = Qwen3ForcedAlignerBackend.from_pretrained(model_path, **load_kwargs)
    config = RealtimeTimestampConfig(
        pad_ms=int(args.timestamp_pad_ms),
        finish_timeout_ms=int(args.timestamp_finish_timeout_ms),
    )
    return TimestampModelActor(aligner), config


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
        if not _prewarm_realtime_cuda_graph(model, args):
            raise RuntimeError("cuda graph prewarm was requested but this backend does not support it")
        return


def _translation_capture_lock(args: argparse.Namespace, *, translation_enabled: bool) -> Any | None:
    if not translation_enabled:
        return None
    if _cuda_graph_enabled(args) and not args.cuda_graph_prewarm:
        return CUDA_GRAPH_CAPTURE_LOCK
    return None


def main() -> None:
    from qwen3_asr_runtime import Qwen3ASRModel

    args = _parse_args()
    backend, load_kwargs = _build_model_load(args)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        backend=backend,
        **load_kwargs,
    )
    translation_enabled = bool(str(args.translation_model or "").strip())
    _prepare_cuda_graph_runtime(model, args)
    translation_capture_lock = _translation_capture_lock(args, translation_enabled=translation_enabled)
    translator, translation_service_config = _build_translation(args)
    translation_actor = (
        TranslationModelActor(translator, capture_lock=translation_capture_lock) if translator is not None else None
    )
    timestamp_actor: TimestampModelActor | None = None
    timestamp_config: RealtimeTimestampConfig | None = None
    try:
        timestamp_actor, timestamp_config = _build_timestamp_actor(args)
    except Exception:
        if translation_actor is not None:
            translation_actor.close(wait=True)
        if timestamp_actor is not None:
            timestamp_actor.close(wait=True)
        raise
    app = build_app(
        model=model,
        timestamp_actor=timestamp_actor,
        timestamp_config=timestamp_config,
        translation_actor=translation_actor,
        translation_service_config=translation_service_config,
        live_stability_delay_ms=args.live_stability_delay_ms,
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install service dependencies with: uv sync --python 3.12") from exc

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_interval=None)


if __name__ == "__main__":
    main()
