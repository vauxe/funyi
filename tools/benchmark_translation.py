# coding=utf-8
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

from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_MAX_NEW_TOKENS,
    DEFAULT_HYMT_MODEL,
    HYMTGenerationConfig,
    HYMTTranslator,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HY-MT translation inference on a provided JSONL case file.")
    parser.add_argument("--dataset", required=True, help="JSONL case file path.")
    parser.add_argument("--cases", default=None, help="Comma-separated case ids or groups. Default: all.")
    parser.add_argument("--model", default=DEFAULT_HYMT_MODEL, help="HY-MT model path or Hugging Face id.")
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
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument("--sample", action="store_true", help="Enable sampling. Default is deterministic greedy decode.")
    decode_group.add_argument("--greedy", action="store_true", help="Use deterministic greedy decode. This is the default.")
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed for repeatable sampling benchmarks.")
    parser.add_argument(
        "--seed-mode",
        choices=["case", "run"],
        default="case",
        help="Use per-case seeds by default so output changes do not shift later case RNG.",
    )
    parser.add_argument(
        "--extra-generate-kwargs",
        default=None,
        help="Optional JSON object. Omit to use runtime defaults.",
    )
    parser.add_argument("--warmup", default="1", help="Warmup count, or 'all' to warm up every selected case.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download model files during startup. Default is local-only.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional path to write benchmark JSON.")
    parser.add_argument("--include-output", action="store_true", help="Include full translated text in each case row.")
    parser.add_argument("--include-dataset-path", action="store_true", help="Include the dataset path in benchmark JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _select_cases(_load_cases(Path(args.dataset)), args.cases)
    if not cases:
        raise ValueError("No translation benchmark cases selected")

    do_sample = bool(args.sample)
    torch.manual_seed(int(args.seed))
    config_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "do_sample": do_sample,
    }
    if args.extra_generate_kwargs is not None:
        config_kwargs["extra_generate_kwargs"] = _parse_json_object(args.extra_generate_kwargs)
    generation_config = HYMTGenerationConfig(**config_kwargs)
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

    warmup_count = _resolve_warmup_count(args.warmup, len(cases))
    warmup_started = time.perf_counter()
    for index in range(warmup_count):
        case = cases[index % len(cases)]
        translator.profile_translate(
            str(case["text"]),
            target_language=str(case["target_language"]),
            source_language=str(case.get("source_language") or ""),
            max_new_tokens=args.max_new_tokens,
            sync_cuda=True,
        )
    warmup_wall_sec = time.perf_counter() - warmup_started
    # Warmup consumes sampling RNG; measured outputs should not depend on warmup count.
    torch.manual_seed(int(args.seed))

    if torch.cuda.is_available() and ("cuda" in str(args.device) or args.device == "auto"):
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for index, case in enumerate(cases):
        seed = int(args.seed) + index if args.seed_mode == "case" else None
        rows.append(
            _run_case(
                translator,
                case,
                repeats=max(1, int(args.repeats)),
                max_new_tokens=args.max_new_tokens,
                include_output=args.include_output,
                seed=seed,
            )
        )
    payload = {
        "model": args.model,
        "case_count": len(rows),
        "device": args.device,
        "dtype": args.dtype or ("bfloat16" if args.device.startswith("cuda") or args.device == "auto" else "auto"),
        "attn_implementation": translator.attn_implementation,
        "decode_backend": translator.decode_backend,
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "do_sample": do_sample,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "extra_generate_kwargs": generation_config.extra_generate_kwargs,
            "seed": args.seed,
            "seed_mode": args.seed_mode,
        },
        "load_wall_sec": round(load_wall_sec, 3),
        "warmup_count": warmup_count,
        "warmup_wall_sec": round(warmup_wall_sec, 3),
        "summary": _summarize(rows),
        "cuda_peak_allocated_mb": _cuda_peak_allocated_mb(args.device),
        "cases": rows,
    }
    if args.include_dataset_path:
        payload["dataset"] = str(args.dataset)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: case must be a JSON object")
        for field in ("id", "target_language", "text"):
            if not str(item.get(field) or "").strip():
                raise ValueError(f"{path}:{line_number}: missing required field {field!r}")
        case_id = str(item["id"])
        if case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case id {case_id!r}")
        seen.add(case_id)
        cases.append(item)
    return cases


def _select_cases(cases: list[dict[str, Any]], spec: str | None) -> list[dict[str, Any]]:
    if not spec:
        return cases
    wanted = {item.strip() for item in spec.split(",") if item.strip()}
    return [
        case
        for case in cases
        if str(case.get("id") or "") in wanted or str(case.get("group") or "") in wanted
    ]


def _resolve_warmup_count(value: str, case_count: int) -> int:
    text = str(value).strip().lower()
    if text == "all":
        return case_count
    return max(0, int(text))


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _run_case(
    translator: HYMTTranslator,
    case: dict[str, Any],
    *,
    repeats: int,
    max_new_tokens: int,
    include_output: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    output = ""
    for _ in range(repeats):
        if seed is not None:
            torch.manual_seed(int(seed))
        result = translator.profile_translate(
            str(case["text"]),
            target_language=str(case["target_language"]),
            source_language=str(case.get("source_language") or ""),
            max_new_tokens=max_new_tokens,
            sync_cuda=True,
        )
        output = result.text
        samples.append(
            {
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "encode_wall_sec": result.encode_wall_sec,
                "generate_wall_sec": result.generate_wall_sec,
                "decode_wall_sec": result.decode_wall_sec,
                "total_wall_sec": result.total_wall_sec,
            }
        )
    total_values = [float(item["total_wall_sec"]) for item in samples]
    encode_values = [float(item["encode_wall_sec"]) for item in samples]
    generate_values = [float(item["generate_wall_sec"]) for item in samples]
    decode_values = [float(item["decode_wall_sec"]) for item in samples]
    generated_tokens = [int(item["generated_tokens"]) for item in samples]
    median_total = median(total_values)
    median_encode = median(encode_values)
    median_generate = median(generate_values)
    median_decode = median(decode_values)
    median_generated_tokens = median(generated_tokens)
    row = {
        "id": case["id"],
        "group": case.get("group", ""),
        "target_language": case["target_language"],
        "source_language": case.get("source_language", ""),
        "source_chars": len(str(case["text"])),
        "prompt_tokens": int(samples[-1]["prompt_tokens"]),
        "generated_tokens_median": median_generated_tokens,
        "total_wall_sec_median": round(median_total, 4),
        "encode_wall_sec_median": round(median_encode, 4),
        "generate_wall_sec_median": round(median_generate, 4),
        "decode_wall_sec_median": round(median_decode, 4),
        "decode_tokens_per_sec": round(median_generated_tokens / median_generate, 2) if median_generate > 0 else 0.0,
        "output_chars": len(output),
    }
    if include_output:
        row["output"] = output
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [float(row["total_wall_sec_median"]) for row in rows]
    encodes = [float(row["encode_wall_sec_median"]) for row in rows]
    generates = [float(row["generate_wall_sec_median"]) for row in rows]
    decodes = [float(row["decode_wall_sec_median"]) for row in rows]
    prompt_tokens = [float(row["prompt_tokens"]) for row in rows]
    generated_tokens = [float(row["generated_tokens_median"]) for row in rows]
    generated_tokens_sum = sum(generated_tokens)
    generate_wall_sec_sum = sum(generates)
    total_wall_sec_sum = sum(totals)
    return {
        "total_wall_sec_sum": round(total_wall_sec_sum, 4),
        "total_wall_sec_median": round(median(totals), 4),
        "encode_wall_sec_sum": round(sum(encodes), 4),
        "generate_wall_sec_sum": round(generate_wall_sec_sum, 4),
        "decode_wall_sec_sum": round(sum(decodes), 4),
        "generated_tokens_sum": round(generated_tokens_sum, 1),
        "decode_tokens_per_sec_total": _tokens_per_sec(generated_tokens_sum, generate_wall_sec_sum),
        "end_to_end_tokens_per_sec_total": _tokens_per_sec(generated_tokens_sum, total_wall_sec_sum),
        "decode_tokens_per_sec_median": round(median(float(row["decode_tokens_per_sec"]) for row in rows), 2),
        "prompt_tokens_median": round(median(prompt_tokens), 1),
        "generated_tokens_median": round(median(generated_tokens), 1),
        "correlation": {
            "prompt_tokens_vs_generate_sec": _pearson(prompt_tokens, generates),
            "generated_tokens_vs_generate_sec": _pearson(generated_tokens, generates),
            "generated_tokens_vs_total_sec": _pearson(generated_tokens, totals),
        },
    }


def _tokens_per_sec(tokens: float, wall_sec: float) -> float:
    return round(tokens / wall_sec, 2) if wall_sec > 0 else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_deltas = [x - x_mean for x in xs]
    y_deltas = [y - y_mean for y in ys]
    x_norm = sum(item * item for item in x_deltas) ** 0.5
    y_norm = sum(item * item for item in y_deltas) ** 0.5
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return round(sum(x * y for x, y in zip(x_deltas, y_deltas)) / (x_norm * y_norm), 4)


def _cuda_peak_allocated_mb(device: str) -> float | None:
    if not torch.cuda.is_available() or ("cuda" not in str(device) and device != "auto"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


if __name__ == "__main__":
    main()
