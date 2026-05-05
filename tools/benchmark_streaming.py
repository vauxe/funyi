# coding=utf-8
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_window import load_audio_window
from tools.streaming_regression_common import (
    DEFAULT_STEP_MS,
    StreamingCaseSpec,
    parse_int_list,
    parse_name_filter,
    run_streaming_case,
    summarize_seconds,
)
from tools.runtime_helpers import _default_attn_implementation, _dispose_model, _resolve_dtype, _set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime streaming transcription on local streaming goldens.")
    parser.add_argument("--golden", default="local_goldens/streaming_regression.json", help="Streaming regression golden JSON")
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--audio", default=None, help="Optional audio override")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"], help="Optional dtype override")
    parser.add_argument("--device-map", default=None, help="Optional device_map override")
    parser.add_argument("--attn-implementation", default=None, help="Optional attn_implementation override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-inference-batch-size", type=int, default=None)
    parser.add_argument("--cases", default=None, help="Comma separated case names to run. Default: all")
    parser.add_argument("--step-ms", default=None, help="Comma separated push step sizes. Default: golden config")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--check-final", action="store_true", help="Verify final text hash against the golden.")
    parser.add_argument("--max-window-sec", type=float, default=None, help="Override with a bounded live-audio model window.")
    parser.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=None,
        help="Override rolling text-prefix token cap. Defaults inside runtime when --max-window-sec is set.",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA graph decode loop.")
    parser.add_argument("--cuda-graph-len-bucket", type=int, default=1, help="Round CUDA graph/cache length up to this token bucket.")
    parser.add_argument("--flashinfer", action="store_true", help="Use FlashInfer decode attention.")
    parser.add_argument("--fused-rmsnorm", action="store_true", help="Patch RMSNorm modules to F.rms_norm.")
    parser.add_argument("--fused-linears", action="store_true", help="Fuse q/k/v and gate/up linears.")
    parser.add_argument("--quantized-linears", action="store_true", help="Use W8A16 for fused qkv/gate_up.")
    parser.add_argument(
        "--spec-decode",
        action="store_true",
        help="Speculative verification of the rollback prefix. Not byte-identical under bf16; quality-gated via CER.",
    )
    return parser.parse_args()


def _steps_by_ms(case_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(step["step_ms"]): step for step in case_payload["steps"]}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _sum_spec_decode_stats(payloads: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for payload in payloads:
        for name, value in payload["metrics"].get("spec_decode_stats", {}).items():
            stats[name] = stats.get(name, 0) + int(value)
    return stats


def main() -> None:
    args = _parse_args()
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    model_name = args.model or golden["model"]
    audio_path = args.audio or golden["audio"]
    seed = int(args.seed if args.seed is not None else golden["seed"])
    _set_seed(seed)

    load_kwargs = dict(golden["load_kwargs"])
    if args.dtype is not None:
        load_kwargs["dtype"] = args.dtype
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    if args.max_new_tokens is not None:
        load_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.max_inference_batch_size is not None:
        load_kwargs["max_inference_batch_size"] = args.max_inference_batch_size

    attn_implementation = args.attn_implementation
    if attn_implementation is None:
        attn_implementation = load_kwargs.get("attn_implementation") or _default_attn_implementation(load_kwargs.get("device_map"))
    load_kwargs["attn_implementation"] = attn_implementation

    dtype_name = load_kwargs.pop("dtype")
    max_inference_batch_size = int(load_kwargs.pop("max_inference_batch_size"))
    max_new_tokens = int(load_kwargs.pop("max_new_tokens"))

    from qwen3_asr_runtime import Qwen3ASRModel
    from qwen3_asr_runtime import utils as runtime_utils

    sample_rate = int(runtime_utils.SAMPLE_RATE)
    selected_names = parse_name_filter(args.cases)
    if selected_names is not None:
        available_names = {str(case["name"]) for case in golden["cases"]}
        unknown_names = sorted(selected_names.difference(available_names))
        if unknown_names:
            raise ValueError(f"Unknown streaming cases: {unknown_names}. Available: {sorted(available_names)}")
    default_step_ms = tuple(int(item) for item in golden.get("streaming_config", {}).get("step_ms", DEFAULT_STEP_MS))
    selected_steps = set(parse_int_list(args.step_ms, default=default_step_ms))
    streaming_config = golden["streaming_config"]
    max_window_sec = args.max_window_sec
    if max_window_sec is None:
        max_window_sec = streaming_config.get("max_window_sec")
    max_prefix_tokens = args.max_prefix_tokens
    if max_prefix_tokens is None:
        max_prefix_tokens = streaming_config.get("max_prefix_tokens")
    if args.check_final:
        golden_max_window_sec = streaming_config.get("max_window_sec")
        golden_max_prefix_tokens = streaming_config.get("max_prefix_tokens")
        if max_window_sec != golden_max_window_sec or max_prefix_tokens != golden_max_prefix_tokens:
            raise ValueError(
                "--check-final compares against the golden streaming semantics and cannot be used "
                "with different live-window settings. Regenerate a matching golden or drop --check-final. "
                f"golden max_window_sec={golden_max_window_sec}, requested max_window_sec={max_window_sec}; "
                f"golden max_prefix_tokens={golden_max_prefix_tokens}, requested max_prefix_tokens={max_prefix_tokens}"
            )

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
        init_kwargs["cuda_graph_len_bucket"] = int(args.cuda_graph_len_bucket)
    if args.flashinfer:
        init_kwargs["flashinfer"] = True
    if args.fused_rmsnorm:
        init_kwargs["fused_rmsnorm"] = True
    if args.fused_linears:
        init_kwargs["fused_linears"] = True
    if args.quantized_linears:
        init_kwargs["quantized_linears"] = True

    model = Qwen3ASRModel.from_pretrained(model_name, **init_kwargs)
    model.eval()
    results = []
    try:
        for case_payload in golden["cases"]:
            if selected_names is not None and case_payload["name"] not in selected_names:
                continue
            step_payloads = _steps_by_ms(case_payload)
            missing_steps = sorted(selected_steps.difference(step_payloads))
            if missing_steps:
                raise ValueError(f"Case {case_payload['name']} does not contain step_ms entries: {missing_steps}")

            sliced_audio = load_audio_window(
                audio_path,
                sample_rate=sample_rate,
                start_sec=case_payload["start_sec"],
                duration_sec=case_payload["duration_sec"],
                normalize_audios=runtime_utils.normalize_audios,
            )
            case = StreamingCaseSpec(
                name=case_payload["name"],
                start_sec=float(case_payload["start_sec"]),
                duration_sec=float(case_payload["duration_sec"]),
                context=str(case_payload["context"]),
                language=case_payload["language"],
            )

            for step_ms in sorted(selected_steps):
                expected = step_payloads[step_ms]
                for _ in range(max(0, int(args.warmup))):
                    run_streaming_case(
                        model=model,
                        wav16k=sliced_audio,
                        sample_rate=sample_rate,
                        case=case,
                        step_ms=step_ms,
                        chunk_size_sec=float(streaming_config["chunk_size_sec"]),
                        unfixed_chunk_num=int(streaming_config["unfixed_chunk_num"]),
                        unfixed_token_num=int(streaming_config["unfixed_token_num"]),
                        max_window_sec=float(max_window_sec) if max_window_sec is not None else None,
                        max_prefix_tokens=int(max_prefix_tokens) if max_prefix_tokens is not None else None,
                        timed=True,
                        spec_decode=bool(args.spec_decode),
                    )

                run_payloads = []
                for _ in range(max(1, int(args.repeats))):
                    payload = run_streaming_case(
                        model=model,
                        wav16k=sliced_audio,
                        sample_rate=sample_rate,
                        case=case,
                        step_ms=step_ms,
                        chunk_size_sec=float(streaming_config["chunk_size_sec"]),
                        unfixed_chunk_num=int(streaming_config["unfixed_chunk_num"]),
                        unfixed_token_num=int(streaming_config["unfixed_token_num"]),
                        max_window_sec=float(max_window_sec) if max_window_sec is not None else None,
                        max_prefix_tokens=int(max_prefix_tokens) if max_prefix_tokens is not None else None,
                        timed=True,
                        spec_decode=bool(args.spec_decode),
                        include_internal_stats=True,
                    )
                    if args.check_final and payload["final"]["text_sha256"] != expected["final"]["text_sha256"]:
                        raise AssertionError(
                            f"final text hash mismatch for {case.name}:{step_ms}: "
                            f"expected {expected['final']['text_sha256']}, actual {payload['final']['text_sha256']}"
                        )
                    run_payloads.append(payload)

                total_walls = [payload["timing"]["total_wall_sec"] for payload in run_payloads]
                active_update_walls = [
                    event["wall_sec"]
                    for payload in run_payloads
                    for event in payload["timing"]["events"]
                    if event["decode_steps"] > 0
                ]
                all_push_walls = [
                    event["wall_sec"]
                    for payload in run_payloads
                    for event in payload["timing"]["events"]
                    if event["event"] == "push"
                ]
                total_summary = summarize_seconds(total_walls)
                median_total = float(median(total_walls))
                duration_sec = float(case_payload["duration_sec"])
                results.append(
                    {
                        "name": case.name,
                        "step_ms": step_ms,
                        "duration_sec": duration_sec,
                        "push_count": run_payloads[-1]["metrics"]["push_count"],
                        "model_decode_updates": run_payloads[-1]["metrics"]["model_decode_updates"],
                        "first_text_audio_ms": run_payloads[-1]["metrics"]["first_text_audio_ms"],
                        "audio_sec_per_wall_sec": round(duration_sec / median_total, 4) if median_total > 0 else None,
                        "total_wall": total_summary,
                        "active_update_wall": summarize_seconds(active_update_walls),
                        "active_update_p95_sec": round(_percentile(active_update_walls, 0.95), 4),
                        "all_push_wall": summarize_seconds(all_push_walls),
                        "spec_decode_stats": _sum_spec_decode_stats(run_payloads),
                    }
                )
    finally:
        model = None
        _dispose_model()

    if not results:
        raise ValueError("No streaming benchmark cases were selected.")

    print(
        "Streaming benchmark:",
        {
            "golden": args.golden,
            "model": model_name,
            "audio": audio_path,
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "cuda_graph": bool(args.cuda_graph),
            "cuda_graph_len_bucket": int(args.cuda_graph_len_bucket) if args.cuda_graph else None,
            "flashinfer": bool(args.flashinfer),
            "fused_rmsnorm": bool(args.fused_rmsnorm),
            "fused_linears": bool(args.fused_linears),
            "quantized_linears": bool(args.quantized_linears),
            "max_window_sec": max_window_sec,
            "max_prefix_tokens": max_prefix_tokens,
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "checked_final": bool(args.check_final),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
