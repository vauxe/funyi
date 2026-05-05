# coding=utf-8

import argparse
import asyncio
import json
import logging
from typing import Any

import numpy as np

from qwen3_asr_runtime.realtime_session import RealtimeASRConfig, RealtimeASRSession
from qwen3_asr_runtime.transcript_store import TranscriptStore
from qwen3_asr_runtime.utils import SAMPLE_RATE
from qwen3_asr_runtime.vad import SileroVadConfig

_SERVICE_PRE_ROLL_MS = 240
_SERVICE_INPUT_CHUNK_MS = 200
_LOGGER = logging.getLogger(__name__)


def build_app(
    *,
    model: Any,
) -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install service dependencies with: uv sync --python 3.12") from exc

    app = FastAPI(title="Qwen3-ASR Runtime Realtime ASR Service")
    active_lock = asyncio.Lock()
    active_connection = {"open": False}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/asr")
    async def websocket_asr(websocket: WebSocket) -> None:
        await websocket.accept()
        async with active_lock:
            if active_connection["open"]:
                await _send_json(websocket, {"type": "error", "error": "Another realtime session is active."})
                await websocket.close(code=1013)
                return
            active_connection["open"] = True

        try:
            start_payload = await _receive_start(websocket)
            if start_payload is None:
                return
            session_id = str(start_payload.get("session_id") or "default")
            store = TranscriptStore(transcript_id=session_id)
            config = _build_session_config(start_payload)
            try:
                session = RealtimeASRSession(model, transcript_store=store, config=config)
            except RuntimeError as exc:
                await _send_json(websocket, {"type": "error", "error": str(exc)})
                await websocket.close(code=1011)
                return

            await _send_json(
                websocket,
                {
                    "type": "ready",
                    "session_id": session_id,
                    "sample_rate": SAMPLE_RATE,
                    "audio_format": "pcm_s16le",
                },
            )

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    audio = decode_pcm_s16le(message["bytes"])
                    events = await asyncio.to_thread(session.ingest_audio, audio)
                    await _send_events(websocket, events)
                    continue

                if message.get("text") is None:
                    continue

                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send_json(websocket, {"type": "error", "error": "Invalid JSON command."})
                    continue
                if not isinstance(command, dict):
                    await _send_json(websocket, {"type": "error", "error": "Command must be a JSON object."})
                    continue
                command_type = command.get("type")
                if command_type == "flush":
                    events = await asyncio.to_thread(session.flush)
                    await _send_events(websocket, events)
                elif command_type == "finish":
                    events = await asyncio.to_thread(session.finish)
                    await _send_events(websocket, events)
                    await websocket.close(code=1000)
                    return
                else:
                    await _send_json(websocket, {"type": "error", "error": f"Unsupported command: {command_type}"})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            _LOGGER.exception("Realtime ASR WebSocket session failed.")
            try:
                await _send_json(websocket, {"type": "error", "error": str(exc) or type(exc).__name__})
                await websocket.close(code=1011)
            except Exception:
                _LOGGER.exception("Failed to send realtime ASR error response.")
            return
        finally:
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
        pre_roll_ms=_SERVICE_PRE_ROLL_MS,
        input_chunk_ms=_SERVICE_INPUT_CHUNK_MS,
        vad=SileroVadConfig(),
    )


async def _receive_start(websocket: Any) -> dict[str, Any] | None:
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        return None
    if message.get("text") is None:
        await _send_json(websocket, {"type": "error", "error": "First frame must be a JSON start command."})
        await websocket.close(code=1003)
        return None
    try:
        payload = json.loads(message["text"])
    except json.JSONDecodeError:
        await _send_json(websocket, {"type": "error", "error": "Start command must be valid JSON."})
        await websocket.close(code=1003)
        return None
    if not isinstance(payload, dict):
        await _send_json(websocket, {"type": "error", "error": "Start command must be a JSON object."})
        await websocket.close(code=1003)
        return None
    if payload.get("type") != "start":
        await _send_json(websocket, {"type": "error", "error": "First command must be type=start."})
        await websocket.close(code=1003)
        return None
    try:
        sample_rate = int(payload.get("sample_rate", SAMPLE_RATE))
    except (TypeError, ValueError):
        await _send_json(websocket, {"type": "error", "error": "sample_rate must be 16000."})
        await websocket.close(code=1003)
        return None
    audio_format = str(payload.get("audio_format") or "pcm_s16le").lower()
    if sample_rate != SAMPLE_RATE or audio_format != "pcm_s16le":
        await _send_json(
            websocket,
            {
                "type": "error",
                "error": "Only mono pcm_s16le at 16000 Hz is supported.",
            },
        )
        await websocket.close(code=1003)
        return None
    return payload


async def _send_events(websocket: Any, events: list[dict[str, Any]]) -> None:
    for event in events:
        await _send_json(websocket, event)


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


def main() -> None:
    from qwen3_asr_runtime import Qwen3ASRModel

    args = _parse_args()
    backend, load_kwargs = _build_model_load(args)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        backend=backend,
        **load_kwargs,
    )
    app = build_app(model=model)

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install service dependencies with: uv sync --python 3.12") from exc

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_interval=None)


if __name__ == "__main__":
    main()
