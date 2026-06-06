# coding=utf-8
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_window import load_audio_window
from tools.streaming_regression_common import (
    DEFAULT_STEP_MS,
    parse_int_list,
    parse_name_filter,
    run_streaming_case,
    selected_default_cases,
    text_sha256,
)
from tools.runtime_helpers import (
    _cer,
    _default_attn_implementation,
    _dispose_model,
    _extract_srt_text_range,
    _resolve_dtype,
    _set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate runtime streaming regression goldens."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF repo id or local model path, e.g. Qwen/Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--audio", required=True, help="Audio path used to build streaming slices"
    )
    parser.add_argument(
        "--reference-srt",
        default=None,
        help="Optional SRT path used for final-text CER metadata",
    )
    parser.add_argument(
        "--output",
        default="local_goldens/streaming_regression.json",
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
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma separated case names. Default: all built-in streaming cases",
    )
    parser.add_argument(
        "--step-ms",
        default=None,
        help="Comma separated push step sizes. Default: 500,2000",
    )
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument(
        "--max-window-sec",
        type=float,
        default=None,
        help="Optional bounded live-audio model window.",
    )
    parser.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=None,
        help="Optional rolling text-prefix token cap. Defaults inside runtime when --max-window-sec is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    selected_names = parse_name_filter(args.cases)
    cases = selected_default_cases(selected_names)
    step_ms_list = parse_int_list(args.step_ms, default=DEFAULT_STEP_MS)

    from qwen3_asr_runtime import Qwen3ASRModel
    from qwen3_asr_runtime import utils as runtime_utils

    sample_rate = int(runtime_utils.SAMPLE_RATE)
    attn_implementation = args.attn_implementation or _default_attn_implementation(
        args.device_map
    )
    init_kwargs: dict[str, Any] = {
        "backend": "transformers",
        "dtype": _resolve_dtype(args.dtype),
        "device_map": args.device_map,
        "max_inference_batch_size": int(args.max_inference_batch_size),
        "max_new_tokens": int(args.max_new_tokens),
    }
    if attn_implementation:
        init_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3ASRModel.from_pretrained(args.model, **init_kwargs)
    model.eval()
    try:
        prompt_specs = {
            "default": {"context": "", "language": None},
            "context_only": {"context": "交易 停滞", "language": None},
            "forced_language": {"context": "", "language": "English"},
        }
        prompts = {
            name: model._build_text_prompt(
                context=spec["context"], force_language=spec["language"]
            )
            for name, spec in prompt_specs.items()
        }

        cases_payload = []
        for case in cases:
            sliced_audio = load_audio_window(
                args.audio,
                sample_rate=sample_rate,
                start_sec=case.start_sec,
                duration_sec=case.duration_sec,
                normalize_audios=runtime_utils.normalize_audios,
            )
            steps = []
            for step_ms in step_ms_list:
                step_payload = run_streaming_case(
                    model=model,
                    wav16k=sliced_audio,
                    sample_rate=sample_rate,
                    case=case,
                    step_ms=step_ms,
                    chunk_size_sec=float(args.chunk_size_sec),
                    unfixed_chunk_num=int(args.unfixed_chunk_num),
                    unfixed_token_num=int(args.unfixed_token_num),
                    max_window_sec=args.max_window_sec,
                    max_prefix_tokens=args.max_prefix_tokens,
                    timed=False,
                )
                if args.reference_srt and case.context == "" and case.language is None:
                    reference_text = _extract_srt_text_range(
                        args.reference_srt, case.start_sec, case.duration_sec
                    )
                    step_payload["reference"] = {
                        "reference_chars": len(reference_text),
                        "cer_vs_reference": round(
                            _cer(step_payload["final"]["text"], reference_text), 6
                        ),
                        "reference_text_sha256": text_sha256(reference_text),
                    }
                steps.append(step_payload)

            cases_payload.append(
                {
                    "name": case.name,
                    "start_sec": case.start_sec,
                    "duration_sec": case.duration_sec,
                    "context": case.context,
                    "language": case.language,
                    "steps": steps,
                }
            )

        golden = {
            "schema_version": 1,
            "generator": "tools/generate_streaming_regression_golden.py",
            "source_backend": "runtime_transformers_default",
            "model": args.model,
            "audio": args.audio,
            "reference_srt": args.reference_srt,
            "load_kwargs": {
                "dtype": args.dtype,
                "device_map": args.device_map,
                "attn_implementation": attn_implementation,
                "max_inference_batch_size": int(args.max_inference_batch_size),
                "max_new_tokens": int(args.max_new_tokens),
            },
            "seed": int(args.seed),
            "streaming_config": {
                "step_ms": step_ms_list,
                "chunk_size_sec": float(args.chunk_size_sec),
                "unfixed_chunk_num": int(args.unfixed_chunk_num),
                "unfixed_token_num": int(args.unfixed_token_num),
            },
            "supported_languages": model.get_supported_languages(),
            "prompts": prompts,
            "cases": cases_payload,
        }
        if args.max_window_sec is not None:
            golden["streaming_config"]["max_window_sec"] = float(args.max_window_sec)
        if args.max_prefix_tokens is not None:
            golden["streaming_config"]["max_prefix_tokens"] = int(
                args.max_prefix_tokens
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        model = None
        _dispose_model()

    print(
        "Generated streaming regression golden:",
        {
            "output": str(output_path),
            "case_count": len(cases_payload),
            "step_ms": step_ms_list,
            "model": args.model,
        },
    )


if __name__ == "__main__":
    main()
