# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import median

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime offline transcription on local golden slices."
    )
    parser.add_argument(
        "--golden",
        default="local_goldens/offline_regression.json",
        help="Offline regression golden JSON",
    )
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--audio", default=None, help="Optional audio override")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Optional dtype override",
    )
    parser.add_argument(
        "--device-map", default=None, help="Optional device_map override"
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attn_implementation override",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-inference-batch-size", type=int, default=None)
    parser.add_argument(
        "--cases", default=None, help="Comma separated case names to run. Default: all"
    )
    parser.add_argument(
        "--warmup-cases",
        default="short_default_15s",
        help="Comma separated warmup case names",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="Benchmark repeats per case"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify text hash against golden while benchmarking",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use the hand-rolled CUDA graph decode loop.",
    )
    parser.add_argument(
        "--flashinfer",
        action="store_true",
        help="Dispatch decode attention through flashinfer single_decode_with_kv_cache.",
    )
    parser.add_argument(
        "--fused-rmsnorm",
        action="store_true",
        help="Replace hand-rolled RMSNorm with torch.nn.functional.rms_norm.",
    )
    parser.add_argument(
        "--fused-linears",
        action="store_true",
        help="Fuse q/k/v and gate/up linear projections into single matmuls per layer.",
    )
    parser.add_argument(
        "--quantized-linears",
        action="store_true",
        help="Use W8A16 for fused qkv/gate_up.",
    )
    return parser.parse_args()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min_sec": round(ordered[0], 4),
        "median_sec": round(float(median(ordered)), 4),
        "max_sec": round(ordered[-1], 4),
        "mean_sec": round(sum(ordered) / len(ordered), 4),
    }


def _sync_cuda() -> None:
    try:
        import torch
    except Exception:
        pass
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


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
        attn_implementation = load_kwargs.get(
            "attn_implementation"
        ) or _default_attn_implementation(load_kwargs.get("device_map"))
    load_kwargs["attn_implementation"] = attn_implementation

    dtype_name = load_kwargs.pop("dtype")
    max_inference_batch_size = int(load_kwargs.pop("max_inference_batch_size"))
    max_new_tokens = int(load_kwargs.pop("max_new_tokens"))

    from qwen3_asr_runtime import Qwen3ASRModel
    from qwen3_asr_runtime import utils as runtime_utils

    sample_rate = int(runtime_utils.SAMPLE_RATE)

    selected_names = None
    if args.cases:
        selected_names = {
            item.strip() for item in args.cases.split(",") if item.strip()
        }
    warmup_names = {
        item.strip() for item in args.warmup_cases.split(",") if item.strip()
    }

    cases = [
        case
        for case in golden["cases"]
        if selected_names is None or case["name"] in selected_names
    ]
    if not cases:
        raise ValueError("No cases selected.")

    init_kwargs = dict(
        backend="transformers",
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    init_kwargs["device_map"] = load_kwargs.get("device_map")
    init_kwargs["dtype"] = _resolve_dtype(dtype_name)
    init_kwargs["attn_implementation"] = attn_implementation
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

    model = Qwen3ASRModel.from_pretrained(
        model_name,
        **init_kwargs,
    )
    model.eval()
    try:
        for case in cases:
            if case["name"] not in warmup_names:
                continue
            sliced_audio = load_audio_window(
                audio_path,
                sample_rate=sample_rate,
                start_sec=case["start_sec"],
                duration_sec=case["duration_sec"],
                normalize_audios=runtime_utils.normalize_audios,
            )
            model.transcribe(
                audio=sliced_audio,
                context=case["context"],
                language=case["language"],
                return_time_stamps=False,
            )

        results = []
        for case in cases:
            sliced_audio = load_audio_window(
                audio_path,
                sample_rate=sample_rate,
                start_sec=case["start_sec"],
                duration_sec=case["duration_sec"],
                normalize_audios=runtime_utils.normalize_audios,
            )
            elapsed = []
            text = None
            for _ in range(int(args.repeats)):
                _sync_cuda()
                start = time.perf_counter()
                actual = model.transcribe(
                    audio=sliced_audio,
                    context=case["context"],
                    language=case["language"],
                    return_time_stamps=False,
                )
                _sync_cuda()
                elapsed.append(time.perf_counter() - start)
                text = actual[0].text

            assert text is not None
            if args.check:
                expected_hash = case["text_sha256"]
                actual_hash = _text_sha256(text)
                if expected_hash != actual_hash:
                    raise AssertionError(
                        f"text hash mismatch for {case['name']}: expected {expected_hash}, actual {actual_hash}"
                    )

            summary = _summarize(elapsed)
            median_sec = summary["median_sec"]
            results.append(
                {
                    "name": case["name"],
                    "duration_sec": case["duration_sec"],
                    "chars": len(text),
                    "audio_sec_per_wall_sec": round(
                        case["duration_sec"] / median_sec, 4
                    )
                    if median_sec > 0
                    else None,
                    "chars_per_wall_sec": round(len(text) / median_sec, 2)
                    if median_sec > 0
                    else None,
                    **summary,
                }
            )
    finally:
        model = None
        _dispose_model()

    print(
        "Offline benchmark:",
        {
            "golden": args.golden,
            "backend": "transformers",
            "model": model_name,
            "audio": audio_path,
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "cuda_graph": args.cuda_graph,
            "flashinfer": args.flashinfer,
            "fused_rmsnorm": args.fused_rmsnorm,
            "fused_linears": args.fused_linears,
            "quantized_linears": args.quantized_linears,
            "repeats": int(args.repeats),
            "checked": bool(args.check),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
