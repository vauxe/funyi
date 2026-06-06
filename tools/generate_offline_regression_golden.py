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
from tools.runtime_helpers import (
    _cer,
    _default_attn_implementation,
    _dispose_model,
    _extract_srt_text_range,
    _resolve_dtype,
    _serialize_transcriptions,
    _set_seed,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    start_sec: float
    duration_sec: float
    context: str
    language: Optional[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline regression goldens from the runtime default path."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF repo id or local model path, e.g. Qwen/Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--audio", required=True, help="Audio path used to build regression slices"
    )
    parser.add_argument(
        "--reference-srt",
        default=None,
        help="Optional SRT path used for reference CER metadata",
    )
    parser.add_argument(
        "--output",
        default="local_goldens/offline_regression.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    return parser.parse_args()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="short_default_15s",
            start_sec=0.0,
            duration_sec=15.0,
            context="",
            language=None,
        ),
        CaseSpec(
            name="short_context_15s",
            start_sec=0.0,
            duration_sec=15.0,
            context="交易 停滞",
            language=None,
        ),
        CaseSpec(
            name="short_forced_language_15s",
            start_sec=0.0,
            duration_sec=15.0,
            context="",
            language="English",
        ),
        CaseSpec(
            name="segment_0s",
            start_sec=0.0,
            duration_sec=1200.0,
            context="",
            language=None,
        ),
        CaseSpec(
            name="segment_3600s",
            start_sec=3600.0,
            duration_sec=1200.0,
            context="",
            language=None,
        ),
        CaseSpec(
            name="segment_7200s",
            start_sec=7200.0,
            duration_sec=1200.0,
            context="",
            language=None,
        ),
    ]


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))
    dtype = _resolve_dtype(args.dtype)

    from qwen3_asr_runtime import Qwen3ASRModel
    from qwen3_asr_runtime import utils as runtime_utils

    sample_rate = int(runtime_utils.SAMPLE_RATE)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device_map,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    attn_implementation = args.attn_implementation or _default_attn_implementation(
        args.device_map
    )
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3ASRModel.from_pretrained(args.model, **load_kwargs)
    model.eval()
    try:
        prompt_cases = [
            {"name": "default", "context": "", "language": None},
            {"name": "context_only", "context": "交易 停滞", "language": None},
            {"name": "forced_language", "context": "", "language": "English"},
        ]
        prompts = {
            case["name"]: model._build_text_prompt(
                context=case["context"], force_language=case["language"]
            )
            for case in prompt_cases
        }

        cases_payload = []
        for case in _build_cases():
            sliced_audio = load_audio_window(
                args.audio,
                sample_rate=sample_rate,
                start_sec=case.start_sec,
                duration_sec=case.duration_sec,
                normalize_audios=runtime_utils.normalize_audios,
            )
            result = _serialize_transcriptions(
                model.transcribe(
                    audio=sliced_audio,
                    context=case.context,
                    language=case.language,
                    return_time_stamps=False,
                )
            )
            text = result[0]["text"]
            payload = {
                "name": case.name,
                "start_sec": case.start_sec,
                "duration_sec": case.duration_sec,
                "context": case.context,
                "language": case.language,
                "result": result,
                "text_sha256": _text_sha256(text),
            }
            if args.reference_srt and case.context == "" and case.language is None:
                reference_text = _extract_srt_text_range(
                    args.reference_srt, case.start_sec, case.duration_sec
                )
                payload["reference"] = {
                    "reference_chars": len(reference_text),
                    "cer_vs_reference": round(_cer(text, reference_text), 6),
                    "reference_text_sha256": _text_sha256(reference_text),
                }
            cases_payload.append(payload)

        golden = {
            "schema_version": 1,
            "generator": "tools/generate_offline_regression_golden.py",
            "source_backend": "runtime_transformers_default",
            "model": args.model,
            "audio": args.audio,
            "reference_srt": args.reference_srt,
            "load_kwargs": {
                "dtype": args.dtype,
                "device_map": args.device_map,
                "attn_implementation": attn_implementation,
                "max_inference_batch_size": args.max_inference_batch_size,
                "max_new_tokens": args.max_new_tokens,
            },
            "seed": int(args.seed),
            "supported_languages": model.get_supported_languages(),
            "prompts": prompts,
            "cases": cases_payload,
        }

        output_path.write_text(
            json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        model = None
        _dispose_model()

    print(
        "Generated offline regression golden:",
        {
            "output": str(output_path),
            "case_count": len(golden["cases"]),
            "model": args.model,
        },
    )


if __name__ == "__main__":
    main()
