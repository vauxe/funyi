# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_window import load_audio_window
from tools.runtime_helpers import _default_attn_implementation, _dispose_model, _resolve_dtype, _set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runtime offline regression against a local golden.")
    parser.add_argument("--golden", default="local_goldens/offline_regression.json", help="Offline regression golden JSON")
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--audio", default=None, help="Optional audio override")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"], help="Optional dtype override")
    parser.add_argument("--device-map", default=None, help="Optional device_map override")
    parser.add_argument("--attn-implementation", default=None, help="Optional attn_implementation override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional max_new_tokens override")
    parser.add_argument("--max-inference-batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--cases", default=None, help="Comma separated case names to run. Default: all")
    return parser.parse_args()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_equal(name: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise AssertionError(f"{name} mismatch:\nEXPECTED: {expected!r}\nACTUAL  : {actual!r}")


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

    selected_names = None
    if args.cases:
        selected_names = {item.strip() for item in args.cases.split(",") if item.strip()}

    init_kwargs = dict(
        backend="transformers",
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    init_kwargs["device_map"] = load_kwargs.get("device_map")
    init_kwargs["dtype"] = _resolve_dtype(dtype_name)
    init_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3ASRModel.from_pretrained(
        model_name,
        **init_kwargs,
    )
    model.eval()
    try:
        _assert_equal("supported_languages", golden["supported_languages"], model.get_supported_languages())
        prompt_specs = {
            "default": {"context": "", "language": None},
            "context_only": {"context": "交易 停滞", "language": None},
            "forced_language": {"context": "", "language": "English"},
        }
        for name, spec in prompt_specs.items():
            _assert_equal(
                f"prompt[{name}]",
                golden["prompts"][name],
                model._build_text_prompt(context=spec["context"], force_language=spec["language"]),
            )

        ran_cases = []
        for case in golden["cases"]:
            if selected_names is not None and case["name"] not in selected_names:
                continue
            sliced_audio = load_audio_window(
                audio_path,
                sample_rate=sample_rate,
                start_sec=case["start_sec"],
                duration_sec=case["duration_sec"],
                normalize_audios=runtime_utils.normalize_audios,
            )
            actual = [
                {
                    "language": item.language,
                    "text": item.text,
                    "time_stamps": item.time_stamps,
                }
                for item in model.transcribe(
                    audio=sliced_audio,
                    context=case["context"],
                    language=case["language"],
                    return_time_stamps=False,
                )
            ]
            _assert_equal(f"case[{case['name']}].result", case["result"], actual)
            _assert_equal(
                f"case[{case['name']}].text_sha256",
                case["text_sha256"],
                _text_sha256(actual[0]["text"]),
            )
            ran_cases.append(case["name"])
    finally:
        model = None
        _dispose_model()

    print(
        "Offline regression passed.",
        {
            "golden": args.golden,
            "model": model_name,
            "audio": audio_path,
            "cases": ran_cases,
        },
    )


if __name__ == "__main__":
    main()
