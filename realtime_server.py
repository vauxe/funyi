# coding=utf-8

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
from typing import Any

import numpy as np

from qwen3_asr_runtime.cuda_serialization import CUDA_GRAPH_CAPTURE_LOCK
from qwen3_asr_runtime.realtime_translation import (
    RealtimeTranslationConfig,
    RealtimeTranslationRuntime,
    TranslationModelActor,
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
from qwen3_asr_runtime.utils import SAMPLE_RATE

_SERVICE_SEND_TIMEOUT_SEC = 5.0
_SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS = 700
_TRANSLATION_PREWARM_TEXTS = (
    "今天天气很好。",
    "实时字幕翻译需要在会议开始前完成模型编译和缓存初始化。",
    (
        "长句预热用于覆盖会议场景里更长的稳定字幕段：发言人可能在同一句话里连续解释背景、"
        "补充条件、给出结论，并且包含数字、单位和专有名词，因此运行时不应该第一次遇到长文本才初始化。"
    ),
)
_LOGGER = logging.getLogger(__name__)


class WebSocketSendTimeout(RuntimeError):
    """Raised when a connected client stops consuming server output."""


def build_app(
    *,
    model: Any,
    translation_actor: TranslationModelActor | None = None,
    translation_config: RealtimeTranslationConfig | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install service dependencies with: uv sync --python 3.12") from exc

    lifespan = None
    if translation_actor is not None:

        @asynccontextmanager
        async def translation_lifespan(app: Any) -> Any:
            del app
            try:
                yield
            finally:
                translation_actor.close(wait=False)

        lifespan = translation_lifespan

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
            config = _build_session_config(start_payload)
            try:
                session_translation_config = _session_translation_config(start_payload, translation_config)
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
            if translation is not None:
                ready["translation"] = translation.ready_payload()
            elif translation_actor is not None and translation_config is not None:
                ready["translation"] = _disabled_translation_ready_payload(translation_config)
            await event_queue.put(ready)

            while True:
                message = await _receive_or_sender_failed(websocket, sender_task)
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    audio = decode_pcm_s16le(message["bytes"])
                    events = await asyncio.to_thread(session.ingest_audio, audio)
                    await _publish_session_events(event_queue, translation, events)
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
                    events = await asyncio.to_thread(session.flush)
                    await _publish_session_events(event_queue, translation, events)
                elif command_type == "finish":
                    events = await asyncio.to_thread(session.finish)
                    await _publish_finish_events(event_queue, translation, events)
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


def _build_session_config(payload: dict[str, Any]) -> RealtimeASRConfig:
    return RealtimeASRConfig(
        context=str(payload.get("context") or ""),
        language=payload.get("language"),
    )


def _session_translation_config(
    payload: dict[str, Any],
    base_config: RealtimeTranslationConfig | None,
) -> RealtimeTranslationConfig | None:
    if base_config is None:
        return None

    raw_translation = payload.get("translation")
    if raw_translation is None:
        enabled = _coerce_bool(payload.get("translation_enabled"), default=True)
        requested_target = str(payload.get("translation_target_language") or "").strip()
    elif isinstance(raw_translation, dict):
        enabled = _coerce_bool(raw_translation.get("enabled"), default=True)
        requested_target = str(raw_translation.get("target_language") or "").strip()
    else:
        enabled = _coerce_bool(raw_translation, default=True)
        requested_target = ""

    if not enabled:
        return None
    if requested_target and requested_target != base_config.target_language:
        raise ValueError(
            f"translation target_language must be {base_config.target_language!r} for this service"
        )
    return base_config


def _disabled_translation_ready_payload(config: RealtimeTranslationConfig) -> dict[str, Any]:
    return {
        "enabled": False,
        "available": True,
        "target_language": config.target_language,
    }


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


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
    return payload


async def _publish_session_events(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    translation: RealtimeTranslationRuntime | None,
    events: list[dict[str, Any]],
) -> None:
    for event in events:
        if translation is not None:
            await translation.accept_source_event(event)
        await event_queue.put(event)


async def _publish_finish_events(
    event_queue: asyncio.Queue[dict[str, Any] | None],
    translation: RealtimeTranslationRuntime | None,
    events: list[dict[str, Any]],
) -> None:
    if translation is None:
        for event in events:
            await event_queue.put(event)
        return

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

    translation_events = await translation.finish(transcript_updates)
    for event in translation_events:
        await event_queue.put(event)
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
    parser.add_argument("--translation-target-language", default=None, help="Enable realtime translation to this language.")
    parser.add_argument("--translation-model", default=DEFAULT_HYMT_MODEL)
    parser.add_argument("--translation-device", default="cuda:0")
    parser.add_argument("--translation-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--translation-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--translation-preview-debounce-ms", type=int, default=_SERVICE_TRANSLATION_PREVIEW_DEBOUNCE_MS)
    parser.add_argument("--translation-preview-timeout-ms", type=int, default=30_000)
    parser.add_argument("--translation-max-new-tokens", type=int, default=DEFAULT_HYMT_MAX_NEW_TOKENS)
    parser.add_argument("--translation-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--translation-decode-backend", default=DEFAULT_HYMT_DECODE_BACKEND, choices=["fixed_mask", "generate"])
    parser.add_argument("--translation-attn-implementation", default=DEFAULT_HYMT_ATTN_IMPLEMENTATION)
    parser.add_argument("--translation-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--translation-trust-remote-code", action="store_true")
    parser.add_argument("--translation-prewarm", action=argparse.BooleanOptionalAction, default=True)
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


def _build_translation(args: argparse.Namespace) -> tuple[Any | None, RealtimeTranslationConfig | None]:
    target_language = str(args.translation_target_language or "").strip()
    if not target_language:
        return None, None
    dtype = None if args.translation_dtype in {None, "auto"} else args.translation_dtype
    config = RealtimeTranslationConfig(
        target_language=target_language,
        preview_enabled=bool(args.translation_preview),
        preview_debounce_ms=int(args.translation_preview_debounce_ms),
        preview_timeout_ms=int(args.translation_preview_timeout_ms),
        max_new_tokens=int(args.translation_max_new_tokens),
    )
    generation_config = HYMTGenerationConfig(
        max_new_tokens=int(args.translation_max_new_tokens),
        do_sample=bool(args.translation_sample),
    )
    translator = HYMTTranslator(
        args.translation_model,
        device=str(args.translation_device),
        dtype=dtype,
        local_files_only=bool(args.translation_local_files_only),
        trust_remote_code=bool(args.translation_trust_remote_code),
        attn_implementation=args.translation_attn_implementation,
        decode_backend=args.translation_decode_backend,
        generation_config=generation_config,
    )
    return translator, config


def _prewarm_translation(
    translation_actor: TranslationModelActor,
    config: RealtimeTranslationConfig,
) -> None:
    try:
        results = translation_actor.warmup(
            _TRANSLATION_PREWARM_TEXTS,
            target_language=config.target_language,
            source_language=config.source_language,
            max_new_tokens=config.max_new_tokens,
            sync_cuda=True,
        )
    except Exception as exc:
        raise RuntimeError("translation prewarm failed") from exc
    if len(results) != len(_TRANSLATION_PREWARM_TEXTS):
        raise RuntimeError("translation prewarm did not run all warmup cases")


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
    translation_enabled = bool(str(args.translation_target_language or "").strip())
    _prepare_cuda_graph_runtime(model, args)
    translation_capture_lock = _translation_capture_lock(args, translation_enabled=translation_enabled)
    translator, translation_config = _build_translation(args)
    translation_actor = (
        TranslationModelActor(translator, capture_lock=translation_capture_lock) if translator is not None else None
    )
    try:
        if translation_actor is not None and translation_config is not None and args.translation_prewarm:
            _prewarm_translation(translation_actor, translation_config)
    except Exception:
        if translation_actor is not None:
            translation_actor.close(wait=True)
        raise
    app = build_app(
        model=model,
        translation_actor=translation_actor,
        translation_config=translation_config,
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install service dependencies with: uv sync --python 3.12") from exc

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_interval=None)


if __name__ == "__main__":
    main()
