# coding=utf-8
"""
Lightweight timing profile for the runtime transformers backend.

Instruments the transcribe() hot path with CUDA-synchronized timers:
  - processor (feature extractor + tokenizer)
  - audio tower forward (inside the first thinker forward)
  - prefill (first thinker forward minus audio tower)
  - decode (sum + count of subsequent thinker forwards)
  - CUDA graph replay/capture/cache-copy internals when --cuda-graph is used
  - tokenizer batch_decode

No model behavior changes. Intended for 1-2 repeats on a single case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_window import load_audio_window
from tools.runtime_helpers import (
    _default_attn_implementation,
    _dispose_model,
    _resolve_dtype,
    _set_seed,
)


def _sync() -> None:
    if not torch.cuda.is_available():
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync()
    return time.perf_counter()


class Stopwatch:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> List[Dict[str, Any]]:
        rows = []
        for name in sorted(self.totals, key=lambda k: self.totals[k], reverse=True):
            total = self.totals[name]
            count = self.counts[name]
            rows.append(
                {
                    "name": name,
                    "total_sec": round(total, 4),
                    "calls": count,
                    "avg_ms": round(total * 1000.0 / max(1, count), 3),
                }
            )
        return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile transcribe() stages with sync timers."
    )
    parser.add_argument("--golden", default="local_goldens/offline_regression.json")
    parser.add_argument("--case", default="short_default_15s")
    parser.add_argument(
        "--audio", default=None, help="Override the audio source from the golden."
    )
    parser.add_argument(
        "--start-sec", type=float, default=None, help="Override case start_sec."
    )
    parser.add_argument(
        "--duration-sec", type=float, default=None, help="Override case duration_sec."
    )
    parser.add_argument("--context", default=None, help="Override case context.")
    parser.add_argument(
        "--language",
        default=None,
        help="Override case language. Use an empty string to disable forced language.",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup runs before timing."
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--dtype", default=None, choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--fused-rmsnorm", action="store_true")
    parser.add_argument("--fused-linears", action="store_true")
    parser.add_argument("--quantized-linears", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    matching = [case for case in golden["cases"] if case["name"] == args.case]
    if not matching:
        raise ValueError(f"Case not found: {args.case}")
    case = matching[0]
    audio_source = args.audio or golden["audio"]
    start_sec = float(case["start_sec"] if args.start_sec is None else args.start_sec)
    duration_sec = float(
        case["duration_sec"] if args.duration_sec is None else args.duration_sec
    )
    context = str(case["context"] if args.context is None else args.context)
    if args.language is None:
        language = case["language"]
    else:
        language = str(args.language).strip() or None

    load_kwargs = dict(golden["load_kwargs"])
    if args.dtype is not None:
        load_kwargs["dtype"] = args.dtype
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    if args.max_new_tokens is not None:
        load_kwargs["max_new_tokens"] = args.max_new_tokens

    attn_implementation = (
        args.attn_implementation
        or load_kwargs.get("attn_implementation")
        or _default_attn_implementation(load_kwargs.get("device_map"))
    )
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
    sliced_audio = load_audio_window(
        audio_source,
        sample_rate=sample_rate,
        start_sec=start_sec,
        duration_sec=duration_sec,
        normalize_audios=runtime_utils.normalize_audios,
    )

    backend = model.backend_runtime
    processor = backend.processor
    hf_model = backend.model
    thinker = hf_model.thinker
    graph_decoder = getattr(backend, "_cuda_graph_decoder", None)

    sw = Stopwatch()
    call_idx = {"n": 0, "audio_called": False}

    original_processor_call = processor.__class__.__call__
    original_batch_decode = processor.__class__.batch_decode
    original_thinker_forward = thinker.forward
    original_get_audio_features = thinker.get_audio_features

    def timed_processor_call(self, *args, **kwargs):
        t0 = _now()
        try:
            return original_processor_call(self, *args, **kwargs)
        finally:
            sw.add("processor.__call__", _now() - t0)

    def timed_batch_decode(self, *args, **kwargs):
        t0 = _now()
        try:
            return original_batch_decode(self, *args, **kwargs)
        finally:
            sw.add("processor.batch_decode", _now() - t0)

    def timed_get_audio_features(*args, **kwargs):
        t0 = _now()
        try:
            return original_get_audio_features(*args, **kwargs)
        finally:
            sw.add("audio_tower.get_audio_features", _now() - t0)
            call_idx["audio_called"] = True

    def timed_thinker_forward(*args, **kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        t0 = _now()
        try:
            return original_thinker_forward(*args, **kwargs)
        finally:
            dt = _now() - t0
            if idx == 0:
                # prefill includes audio tower; we also recorded it separately
                sw.add("thinker.forward[prefill]", dt)
            else:
                sw.add("thinker.forward[decode]", dt)

    processor.__class__.__call__ = timed_processor_call
    processor.__class__.batch_decode = timed_batch_decode
    thinker.forward = timed_thinker_forward
    thinker.get_audio_features = timed_get_audio_features

    try:
        for _ in range(int(args.warmup)):
            model.transcribe(
                audio=sliced_audio,
                context=context,
                language=language,
                return_time_stamps=False,
            )

        # reset counters after warmup
        sw = Stopwatch()

        # rebind (the inner closures reference sw via nonlocal capture of the outer sw - fix)
        def timed_processor_call2(self, *args, **kwargs):
            t0 = _now()
            try:
                return original_processor_call(self, *args, **kwargs)
            finally:
                sw.add("processor.__call__", _now() - t0)

        def timed_batch_decode2(self, *args, **kwargs):
            t0 = _now()
            try:
                return original_batch_decode(self, *args, **kwargs)
            finally:
                sw.add("processor.batch_decode", _now() - t0)

        def timed_get_audio_features2(*args, **kwargs):
            t0 = _now()
            try:
                return original_get_audio_features(*args, **kwargs)
            finally:
                sw.add("audio_tower.get_audio_features", _now() - t0)

        def timed_thinker_forward2(*args, **kwargs):
            idx = call_idx["n"]
            call_idx["n"] += 1
            t0 = _now()
            try:
                return original_thinker_forward(*args, **kwargs)
            finally:
                dt = _now() - t0
                if idx == 0:
                    sw.add("thinker.forward[prefill]", dt)
                else:
                    sw.add("thinker.forward[decode]", dt)

        processor.__class__.__call__ = timed_processor_call2
        processor.__class__.batch_decode = timed_batch_decode2
        thinker.forward = timed_thinker_forward2
        thinker.get_audio_features = timed_get_audio_features2
        if graph_decoder is not None:
            graph_decoder.set_profile_callback(
                lambda name, seconds: sw.add(name, seconds)
            )

        wall_total = 0.0
        for _ in range(int(args.repeats)):
            call_idx["n"] = 0
            t0 = _now()
            model.transcribe(
                audio=sliced_audio,
                context=context,
                language=language,
                return_time_stamps=False,
            )
            wall_total += _now() - t0

        wall_mean = wall_total / max(1, int(args.repeats))
        rows = sw.summary()
        cuda_graph_sec = sum(
            r["total_sec"] for r in rows if r["name"].startswith("cuda_graph.")
        )
        cuda_graph_replay_sec = next(
            (r["total_sec"] for r in rows if r["name"] == "cuda_graph.replay"), 0.0
        )
        cuda_graph_replay_calls = next(
            (r["calls"] for r in rows if r["name"] == "cuda_graph.replay"), 0
        )
        cuda_graph_capture_sec = next(
            (r["total_sec"] for r in rows if r["name"] == "cuda_graph.capture_total"),
            0.0,
        )
        cuda_graph_cache_copy_sec = next(
            (
                r["total_sec"]
                for r in rows
                if r["name"] == "cuda_graph.cache_copy_dynamic_to_static"
            ),
            0.0,
        )
        accounted_no_dup = 0.0
        for row in rows:
            name = row["name"]
            if name == "audio_tower.get_audio_features":
                # Audio tower is inside thinker.forward[prefill].
                continue
            if name == "cuda_graph.capture_total":
                # Capture wraps decode forwards, so report it separately but do
                # not include it in the non-overlapping accounted total.
                continue
            accounted_no_dup += float(row["total_sec"])

        print(
            "profile",
            {
                "case": args.case,
                "audio": str(audio_source),
                "start_sec": start_sec,
                "duration_sec": duration_sec,
                "language": language,
                "context_chars": len(context),
                "repeats": int(args.repeats),
                "wall_sec_per_run": round(wall_mean, 4),
                "wall_sec_total": round(wall_total, 4),
                "accounted_sec": round(accounted_no_dup, 4),
                "other_sec": round(wall_total - accounted_no_dup, 4),
                "cuda_graph_sec": round(cuda_graph_sec, 4),
                "cuda_graph_replay_sec": round(cuda_graph_replay_sec, 4),
                "cuda_graph_replay_calls": int(cuda_graph_replay_calls),
                "cuda_graph_capture_sec": round(cuda_graph_capture_sec, 4),
                "cuda_graph_cache_copy_sec": round(cuda_graph_cache_copy_sec, 4),
                "cuda_graph": bool(args.cuda_graph),
                "flashinfer": bool(args.flashinfer),
                "fused_rmsnorm": bool(args.fused_rmsnorm),
                "fused_linears": bool(args.fused_linears),
                "quantized_linears": bool(args.quantized_linears),
                "stages": rows,
            },
        )
    finally:
        if graph_decoder is not None:
            graph_decoder.set_profile_callback(None)
        processor.__class__.__call__ = original_processor_call
        processor.__class__.batch_decode = original_batch_decode
        thinker.forward = original_thinker_forward
        thinker.get_audio_features = original_get_audio_features
        del model
        _dispose_model()


if __name__ == "__main__":
    main()
