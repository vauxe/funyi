# coding=utf-8
"""
Profile runtime streaming transcription stages with CUDA-synchronized timers.

This is the streaming counterpart of tools/profile_transcribe.py. It profiles a
single audio window and attributes repeated streaming inference cost across:
processor, audio tower, prefill, decode, CUDA graph replay/capture/cache-copy,
batch_decode, and backend inference wrappers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_window import load_audio_window
from tools.streaming_regression_common import StreamingCaseSpec, run_streaming_case
from tools.runtime_helpers import _default_attn_implementation, _dispose_model, _resolve_dtype, _set_seed


def _sync() -> None:
    if not torch.cuda.is_available():
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    torch.cuda.synchronize()


def _now() -> float:
    _sync()
    return time.perf_counter()


class Stopwatch:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self, *, wall_sec: float) -> list[dict[str, Any]]:
        rows = []
        for name in sorted(self.totals, key=lambda key: self.totals[key], reverse=True):
            total = self.totals[name]
            count = self.counts[name]
            rows.append(
                {
                    "name": name,
                    "total_sec": round(total, 4),
                    "calls": count,
                    "avg_ms": round(total * 1000.0 / max(1, count), 3),
                    "wall_pct": round(100.0 * total / wall_sec, 2) if wall_sec > 0 else None,
                }
            )
        return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile streaming transcription stages.")
    parser.add_argument("--golden", default="local_goldens/streaming_regression.json")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--context", default="")
    parser.add_argument("--language", default=None)
    parser.add_argument("--step-ms", type=int, default=1000)
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--max-window-sec", type=float, default=30.0)
    parser.add_argument("--max-prefix-tokens", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-len-bucket", type=int, default=1)
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--fused-rmsnorm", action="store_true")
    parser.add_argument("--fused-linears", action="store_true")
    parser.add_argument("--quantized-linears", action="store_true")
    parser.add_argument(
        "--spec-decode",
        action="store_true",
        help="Enable speculative verification of the rollback prefix; validate quality with a local CER sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    load_kwargs = dict(golden["load_kwargs"])
    if args.dtype is not None:
        load_kwargs["dtype"] = args.dtype
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    if args.max_new_tokens is not None:
        load_kwargs["max_new_tokens"] = args.max_new_tokens

    attn_implementation = args.attn_implementation or load_kwargs.get("attn_implementation") or _default_attn_implementation(load_kwargs.get("device_map"))
    load_kwargs["attn_implementation"] = attn_implementation
    dtype_name = load_kwargs.pop("dtype")
    max_inference_batch_size = int(load_kwargs.pop("max_inference_batch_size"))
    max_new_tokens = int(load_kwargs.pop("max_new_tokens"))

    _set_seed(int(golden["seed"]))

    from qwen3_asr_runtime import Qwen3ASRModel
    from qwen3_asr_runtime import utils as runtime_utils

    init_kwargs = dict(
        backend="transformers",
        dtype=_resolve_dtype(dtype_name),
        device_map=load_kwargs.get("device_map"),
        attn_implementation=attn_implementation,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    if args.cuda_graph:
        init_kwargs["cuda_graph"] = True
        init_kwargs["cuda_graph_len_bucket"] = args.cuda_graph_len_bucket
    if args.flashinfer:
        init_kwargs["flashinfer"] = True
    if args.fused_rmsnorm:
        init_kwargs["fused_rmsnorm"] = True
    if args.fused_linears:
        init_kwargs["fused_linears"] = True
    if args.quantized_linears:
        init_kwargs["quantized_linears"] = True

    model = Qwen3ASRModel.from_pretrained(golden["model"], **init_kwargs)
    model.eval()

    sample_rate = int(runtime_utils.SAMPLE_RATE)
    audio_path = args.audio or golden["audio"]
    sliced_audio = load_audio_window(
        audio_path,
        sample_rate=sample_rate,
        start_sec=float(args.start_sec),
        duration_sec=float(args.duration_sec),
        normalize_audios=runtime_utils.normalize_audios,
    )
    case = StreamingCaseSpec(
        name="profile_streaming",
        start_sec=float(args.start_sec),
        duration_sec=float(args.duration_sec),
        context=str(args.context or ""),
        language=args.language,
    )

    backend = model.backend_runtime
    processor = backend.processor
    hf_model = backend.model
    thinker = hf_model.thinker
    graph_decoder = getattr(backend, "_cuda_graph_decoder", None)

    original_infer = backend.infer_with_prompts
    original_infer_draft = getattr(backend, "infer_streaming_with_draft", None)
    original_processor_call = processor.__class__.__call__
    original_batch_decode = processor.__class__.batch_decode
    original_thinker_forward = thinker.forward
    original_get_audio_features = thinker.get_audio_features

    try:
        for _ in range(max(0, int(args.warmup))):
            run_streaming_case(
                model=model,
                wav16k=sliced_audio,
                sample_rate=sample_rate,
                case=case,
                step_ms=int(args.step_ms),
                chunk_size_sec=float(args.chunk_size_sec),
                unfixed_chunk_num=int(args.unfixed_chunk_num),
                unfixed_token_num=int(args.unfixed_token_num),
                max_window_sec=args.max_window_sec,
                max_prefix_tokens=args.max_prefix_tokens,
                timed=False,
                spec_decode=bool(args.spec_decode),
            )

        sw = Stopwatch()
        call_idx = {"n": 0}

        def timed_infer(*infer_args, **infer_kwargs):
            call_idx["n"] = 0
            t0 = _now()
            try:
                return original_infer(*infer_args, **infer_kwargs)
            finally:
                sw.add("backend.infer_with_prompts", _now() - t0)

        def timed_infer_draft(*infer_args, **infer_kwargs):
            call_idx["n"] = 0
            t0 = _now()
            try:
                return original_infer_draft(*infer_args, **infer_kwargs)
            finally:
                sw.add("backend.infer_streaming_with_draft", _now() - t0)

        def timed_processor_call(self, *call_args, **call_kwargs):
            t0 = _now()
            try:
                return original_processor_call(self, *call_args, **call_kwargs)
            finally:
                sw.add("processor.__call__", _now() - t0)

        def timed_batch_decode(self, *call_args, **call_kwargs):
            t0 = _now()
            try:
                return original_batch_decode(self, *call_args, **call_kwargs)
            finally:
                sw.add("processor.batch_decode", _now() - t0)

        def timed_get_audio_features(*call_args, **call_kwargs):
            t0 = _now()
            try:
                return original_get_audio_features(*call_args, **call_kwargs)
            finally:
                sw.add("audio_tower.get_audio_features", _now() - t0)

        def timed_thinker_forward(*call_args, **call_kwargs):
            idx = call_idx["n"]
            call_idx["n"] += 1
            t0 = _now()
            try:
                return original_thinker_forward(*call_args, **call_kwargs)
            finally:
                name = "thinker.forward[prefill]" if idx == 0 else "thinker.forward[decode]"
                sw.add(name, _now() - t0)

        backend.infer_with_prompts = timed_infer
        if original_infer_draft is not None:
            backend.infer_streaming_with_draft = timed_infer_draft
        processor.__class__.__call__ = timed_processor_call
        processor.__class__.batch_decode = timed_batch_decode
        thinker.forward = timed_thinker_forward
        thinker.get_audio_features = timed_get_audio_features
        if graph_decoder is not None:
            graph_decoder.set_profile_callback(lambda name, seconds: sw.add(name, seconds))

        t0 = _now()
        payload = run_streaming_case(
            model=model,
            wav16k=sliced_audio,
            sample_rate=sample_rate,
            case=case,
            step_ms=int(args.step_ms),
            chunk_size_sec=float(args.chunk_size_sec),
            unfixed_chunk_num=int(args.unfixed_chunk_num),
            unfixed_token_num=int(args.unfixed_token_num),
            max_window_sec=args.max_window_sec,
            max_prefix_tokens=args.max_prefix_tokens,
            timed=True,
            spec_decode=bool(args.spec_decode),
            include_internal_stats=True,
        )
        wall_sec = _now() - t0

        rows = sw.summary(wall_sec=wall_sec)
        stage = {row["name"]: row for row in rows}
        prefill_sec = stage.get("thinker.forward[prefill]", {}).get("total_sec", 0.0)
        audio_sec = stage.get("audio_tower.get_audio_features", {}).get("total_sec", 0.0)
        decode_sec = stage.get("thinker.forward[decode]", {}).get("total_sec", 0.0)
        processor_sec = stage.get("processor.__call__", {}).get("total_sec", 0.0)
        batch_decode_sec = stage.get("processor.batch_decode", {}).get("total_sec", 0.0)
        infer_prompt_sec = stage.get("backend.infer_with_prompts", {}).get("total_sec", 0.0)
        infer_draft_sec = stage.get("backend.infer_streaming_with_draft", {}).get("total_sec", 0.0)
        infer_sec = float(infer_prompt_sec) + float(infer_draft_sec)
        cuda_graph_sec = sum(row["total_sec"] for row in rows if row["name"].startswith("cuda_graph."))
        cuda_graph_replay_sec = stage.get("cuda_graph.replay", {}).get("total_sec", 0.0)
        cuda_graph_replay_calls = stage.get("cuda_graph.replay", {}).get("calls", 0)
        cuda_graph_capture_sec = stage.get("cuda_graph.capture_total", {}).get("total_sec", 0.0)
        cuda_graph_cache_copy_sec = stage.get("cuda_graph.cache_copy_dynamic_to_static", {}).get("total_sec", 0.0)
        # capture_total wraps decode forwards during CUDA graph capture, so it is
        # reported separately but not subtracted here as a non-overlapping stage.
        cuda_graph_non_overlap_sec = float(cuda_graph_replay_sec) + float(cuda_graph_cache_copy_sec)
        prefill_ex_audio_sec = max(0.0, float(prefill_sec) - float(audio_sec))
        other_inside_infer_sec = max(
            0.0,
            float(infer_sec)
            - float(processor_sec)
            - float(prefill_sec)
            - float(decode_sec)
            - float(batch_decode_sec)
            - float(cuda_graph_non_overlap_sec),
        )

        active_walls = [
            event["wall_sec"]
            for event in payload.get("timing", {}).get("events", [])
            if event.get("decode_steps", 0) > 0
        ]
        print(
            "streaming_profile",
            {
                "audio": audio_path,
                "start_sec": float(args.start_sec),
                "duration_sec": float(args.duration_sec),
                "step_ms": int(args.step_ms),
                "chunk_size_sec": float(args.chunk_size_sec),
                "max_window_sec": args.max_window_sec,
                "wall_sec": round(wall_sec, 4),
                "model_decode_updates": payload["metrics"]["model_decode_updates"],
                "push_count": payload["metrics"]["push_count"],
                "final_chars": payload["final"]["text_chars"],
                "spec_decode_stats": payload["metrics"].get("spec_decode_stats", {}),
                "active_update_wall_median_sec": round(float(median(active_walls)), 4) if active_walls else None,
                "prefill_ex_audio_sec": round(prefill_ex_audio_sec, 4),
                "other_inside_infer_sec": round(other_inside_infer_sec, 4),
                "cuda_graph_sec": round(float(cuda_graph_sec), 4),
                "cuda_graph_replay_sec": round(float(cuda_graph_replay_sec), 4),
                "cuda_graph_replay_calls": int(cuda_graph_replay_calls),
                "cuda_graph_capture_sec": round(float(cuda_graph_capture_sec), 4),
                "cuda_graph_cache_copy_sec": round(float(cuda_graph_cache_copy_sec), 4),
                "cuda_graph": bool(args.cuda_graph),
                "cuda_graph_len_bucket": int(args.cuda_graph_len_bucket) if args.cuda_graph else None,
                "flashinfer": bool(args.flashinfer),
                "fused_rmsnorm": bool(args.fused_rmsnorm),
                "fused_linears": bool(args.fused_linears),
                "quantized_linears": bool(args.quantized_linears),
                "spec_decode": bool(args.spec_decode),
                "stages": rows,
            },
        )
    finally:
        if graph_decoder is not None:
            graph_decoder.set_profile_callback(None)
        backend.infer_with_prompts = original_infer
        if original_infer_draft is not None:
            backend.infer_streaming_with_draft = original_infer_draft
        processor.__class__.__call__ = original_processor_call
        processor.__class__.batch_decode = original_batch_decode
        thinker.forward = original_thinker_forward
        thinker.get_audio_features = original_get_audio_features
        del model
        _dispose_model()


if __name__ == "__main__":
    main()
