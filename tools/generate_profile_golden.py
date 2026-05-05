# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_window import load_audio_window
from tools.runtime_helpers import _default_attn_implementation, _dispose_model, _resolve_dtype, _set_seed


@dataclass(frozen=True)
class CaseSpec:
    name: str
    start_sec: float
    duration_sec: float
    context: str
    language: Optional[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a runtime offline regression golden for a selected case set."
    )
    parser.add_argument("--source-golden", default="local_goldens/offline_regression.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-inference-batch-size", type=int, default=None)
    parser.add_argument("--cases", default="")
    parser.add_argument(
        "--extra-window",
        action="append",
        default=[],
        help="Optional start_sec:duration_sec case to append. Repeat the flag for multiple windows.",
    )
    return parser.parse_args()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_cases() -> list[dict[str, Optional[str]]]:
    return [
        {"name": "default", "context": "", "language": None},
        {"name": "context_only", "context": "交易 停滞", "language": None},
        {"name": "forced_language", "context": "", "language": "English"},
    ]


def _parse_extra_window(spec: str) -> CaseSpec:
    start_sec, duration_sec = spec.split(":")
    start = float(start_sec)
    duration = float(duration_sec)
    return CaseSpec(
        name=f"window_{start:g}_{duration:g}",
        start_sec=start,
        duration_sec=duration,
        context="",
        language=None,
    )


def main() -> None:
    args = _parse_args()
    source_golden = json.loads(Path(args.source_golden).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else source_golden["seed"])
    _set_seed(seed)

    model_name = args.model or source_golden["model"]
    audio_path = args.audio or source_golden["audio"]
    attn_implementation = args.attn_implementation or _default_attn_implementation(args.device_map)
    source_load_kwargs = dict(source_golden.get("load_kwargs", {}))
    if args.max_new_tokens is not None:
        max_new_tokens = int(args.max_new_tokens)
    elif "max_new_tokens" in source_load_kwargs:
        max_new_tokens = int(source_load_kwargs["max_new_tokens"])
    else:
        raise ValueError(
            "max_new_tokens is missing from the source golden's load_kwargs; "
            "pass --max-new-tokens explicitly to avoid silently truncating long outputs."
        )
    if args.max_inference_batch_size is not None:
        max_inference_batch_size = int(args.max_inference_batch_size)
    elif "max_inference_batch_size" in source_load_kwargs:
        max_inference_batch_size = int(source_load_kwargs["max_inference_batch_size"])
    else:
        raise ValueError(
            "max_inference_batch_size is missing from the source golden's load_kwargs; "
            "pass --max-inference-batch-size explicitly."
        )

    requested_case_names = {item.strip() for item in args.cases.split(",") if item.strip()}
    selected_cases = []
    if requested_case_names:
        selected_cases.extend(
            CaseSpec(
                name=case["name"],
                start_sec=float(case["start_sec"]),
                duration_sec=float(case["duration_sec"]),
                context=str(case["context"]),
                language=case["language"],
            )
            for case in source_golden["cases"]
            if case["name"] in requested_case_names
        )
    selected_cases.extend(_parse_extra_window(item) for item in args.extra_window)
    if not selected_cases:
        raise ValueError("No cases selected. Use --cases and/or --extra-window.")

    from qwen3_asr_runtime import Qwen3ASRModel
    from qwen3_asr_runtime import utils as runtime_utils

    sample_rate = int(runtime_utils.SAMPLE_RATE)
    dtype = _resolve_dtype(args.dtype)

    model = Qwen3ASRModel.from_pretrained(
        model_name,
        backend="transformers",
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=attn_implementation,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    model.eval()
    try:
        prompts = {
            case["name"]: model._build_text_prompt(
                context=str(case["context"]),
                force_language=case["language"],
            )
            for case in _prompt_cases()
        }

        cases_payload = []
        for case in selected_cases:
            sliced_audio = load_audio_window(
                audio_path,
                sample_rate=sample_rate,
                start_sec=case.start_sec,
                duration_sec=case.duration_sec,
                normalize_audios=runtime_utils.normalize_audios,
            )
            result = [
                {
                    "language": item.language,
                    "text": item.text,
                    "time_stamps": item.time_stamps,
                }
                for item in model.transcribe(
                    audio=sliced_audio,
                    context=case.context,
                    language=case.language,
                    return_time_stamps=False,
                )
            ]
            text = result[0]["text"]
            cases_payload.append(
                {
                    "name": case.name,
                    "start_sec": case.start_sec,
                    "duration_sec": case.duration_sec,
                    "context": case.context,
                    "language": case.language,
                    "result": result,
                    "text_sha256": _text_sha256(text),
                }
            )

        golden = {
            "schema_version": 1,
            "generator": "tools/generate_profile_golden.py",
            "model": model_name,
            "audio": audio_path,
            "load_kwargs": {
                "dtype": args.dtype,
                "device_map": args.device_map,
                "attn_implementation": attn_implementation,
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
            },
            "seed": seed,
            "supported_languages": model.get_supported_languages(),
            "prompts": prompts,
            "cases": cases_payload,
        }
    finally:
        model = None
        _dispose_model()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "Generated profile golden.",
        {
            "output": str(output_path),
            "case_count": len(cases_payload),
            "model": model_name,
        },
    )


if __name__ == "__main__":
    main()
