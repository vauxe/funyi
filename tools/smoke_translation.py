# coding=utf-8
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_MAX_NEW_TOKENS,
    DEFAULT_HYMT_MODEL,
    HYMTGenerationConfig,
    HYMTTranslator,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test HY-MT subtitle translation inference.")
    parser.add_argument("--model", default=DEFAULT_HYMT_MODEL, help="HY-MT model path or Hugging Face id.")
    parser.add_argument(
        "--text",
        default=(
            "今天的会议主要讨论实时字幕系统的端到端延迟、翻译质量和资源占用，"
            "我们需要在不中断语音识别的情况下，把已经稳定的中文转写结果快速翻译成英文，"
            "并且保证前端看到的双语字幕顺序一致、内容完整。"
        ),
    )
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--source-language", default="")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cpu, or auto.")
    parser.add_argument("--dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        default=DEFAULT_HYMT_ATTN_IMPLEMENTATION,
        help="Transformers attention implementation. Use 'none' to let transformers choose.",
    )
    parser.add_argument("--decode-backend", default=DEFAULT_HYMT_DECODE_BACKEND, choices=["fixed_mask", "generate"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_HYMT_MAX_NEW_TOKENS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for deterministic decode.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download model files during startup. Default is local-only.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print one JSON summary.")
    parser.add_argument("--profile", action="store_true", help="Include encode/generate/decode timing and token counts.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generation_config = HYMTGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=not args.greedy,
    )

    load_started = time.perf_counter()
    translator = HYMTTranslator(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        decode_backend=args.decode_backend,
        generation_config=generation_config,
    )
    load_wall_sec = time.perf_counter() - load_started

    for _ in range(max(0, int(args.warmup))):
        translator.translate(
            args.text,
            target_language=args.target_language,
            source_language=args.source_language,
            max_new_tokens=args.max_new_tokens,
        )
    if torch.cuda.is_available() and ("cuda" in str(args.device) or args.device == "auto"):
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    profiles: list[dict[str, float | int]] = []
    output = ""
    for _ in range(max(1, int(args.repeats))):
        _sync_cuda(args.device)
        started = time.perf_counter()
        if args.profile:
            result = translator.profile_translate(
                args.text,
                target_language=args.target_language,
                source_language=args.source_language,
                max_new_tokens=args.max_new_tokens,
                sync_cuda=True,
            )
            output = result.text
            profiles.append(
                {
                    "prompt_tokens": result.prompt_tokens,
                    "generated_tokens": result.generated_tokens,
                    "encode_wall_sec": round(result.encode_wall_sec, 4),
                    "generate_wall_sec": round(result.generate_wall_sec, 4),
                    "decode_wall_sec": round(result.decode_wall_sec, 4),
                    "total_wall_sec": round(result.total_wall_sec, 4),
                    "decode_tokens_per_sec": round(result.generated_tokens / result.generate_wall_sec, 2)
                    if result.generate_wall_sec > 0
                    else 0.0,
                }
            )
        else:
            output = translator.translate(
                args.text,
                target_language=args.target_language,
                source_language=args.source_language,
                max_new_tokens=args.max_new_tokens,
            )
        _sync_cuda(args.device)
        latencies.append(time.perf_counter() - started)

    payload = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype or ("bfloat16" if args.device.startswith("cuda") or args.device == "auto" else "auto"),
        "attn_implementation": translator.attn_implementation,
        "decode_backend": translator.decode_backend,
        "target_language": args.target_language,
        "source_chars": len(args.text),
        "translation_chars": len(output),
        "load_wall_sec": round(load_wall_sec, 3),
        "latency_sec": {
            "median": round(median(latencies), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "cuda_peak_allocated_mb": _cuda_peak_allocated_mb(args.device),
        "output": output,
    }
    if profiles:
        payload["profile"] = profiles[-1]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"model: {payload['model']}")
        print(
            f"device: {payload['device']} dtype: {payload['dtype']} "
            f"attn: {payload['attn_implementation']} decode: {payload['decode_backend']}"
        )
        print(f"load_wall_sec: {payload['load_wall_sec']}")
        print(f"latency_sec: {payload['latency_sec']}")
        print(f"cuda_peak_allocated_mb: {payload['cuda_peak_allocated_mb']}")
        print("output:")
        print(output)


def _sync_cuda(device: str) -> None:
    if torch.cuda.is_available() and ("cuda" in str(device) or device == "auto"):
        torch.cuda.synchronize()


def _cuda_peak_allocated_mb(device: str) -> float | None:
    if not torch.cuda.is_available() or ("cuda" not in str(device) and device != "auto"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


if __name__ == "__main__":
    main()
