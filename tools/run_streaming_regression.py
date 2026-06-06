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
    StreamingCaseSpec,
    first_diff_index,
    parse_int_list,
    parse_name_filter,
    run_streaming_case,
)
from tools.runtime_helpers import (
    _default_attn_implementation,
    _dispose_model,
    _resolve_dtype,
    _set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime streaming regression against a local golden."
    )
    parser.add_argument(
        "--golden",
        default="local_goldens/streaming_regression.json",
        help="Streaming regression golden JSON",
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
        "--step-ms",
        default=None,
        help="Comma separated push step sizes. Default: golden config",
    )
    parser.add_argument(
        "--cuda-graph", action="store_true", help="Use CUDA graph decode loop."
    )
    parser.add_argument(
        "--cuda-graph-len-bucket",
        type=int,
        default=1,
        help="Round CUDA graph/cache length up to this token bucket.",
    )
    parser.add_argument(
        "--flashinfer", action="store_true", help="Use FlashInfer decode attention."
    )
    parser.add_argument(
        "--fused-rmsnorm",
        action="store_true",
        help="Patch RMSNorm modules to F.rms_norm.",
    )
    parser.add_argument(
        "--fused-linears", action="store_true", help="Fuse q/k/v and gate/up linears."
    )
    parser.add_argument(
        "--quantized-linears",
        action="store_true",
        help="Use W8A16 for fused qkv/gate_up.",
    )
    return parser.parse_args()


def _assert_equal(name: str, expected: Any, actual: Any) -> None:
    if expected == actual:
        return
    if isinstance(expected, list) and isinstance(actual, list):
        idx = first_diff_index(expected, actual)
        exp_item = (
            expected[idx] if idx is not None and idx < len(expected) else "<missing>"
        )
        act_item = actual[idx] if idx is not None and idx < len(actual) else "<missing>"
        raise AssertionError(
            f"{name} mismatch at index {idx}:\nEXPECTED: {exp_item!r}\nACTUAL  : {act_item!r}\n"
            f"EXPECTED_LEN: {len(expected)} ACTUAL_LEN: {len(actual)}"
        )
    raise AssertionError(
        f"{name} mismatch:\nEXPECTED: {expected!r}\nACTUAL  : {actual!r}"
    )


def _steps_by_ms(case_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(step["step_ms"]): step for step in case_payload["steps"]}


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
    selected_names = parse_name_filter(args.cases)
    if selected_names is not None:
        available_names = {str(case["name"]) for case in golden["cases"]}
        unknown_names = sorted(selected_names.difference(available_names))
        if unknown_names:
            raise ValueError(
                f"Unknown streaming cases: {unknown_names}. Available: {sorted(available_names)}"
            )
    default_step_ms = tuple(
        int(item)
        for item in golden.get("streaming_config", {}).get("step_ms", DEFAULT_STEP_MS)
    )
    selected_steps = set(parse_int_list(args.step_ms, default=default_step_ms))
    streaming_config = golden["streaming_config"]
    max_window_sec = streaming_config.get("max_window_sec")
    max_prefix_tokens = streaming_config.get("max_prefix_tokens")

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
    ran: list[str] = []
    try:
        _assert_equal(
            "supported_languages",
            golden["supported_languages"],
            model.get_supported_languages(),
        )
        prompt_specs = {
            "default": {"context": "", "language": None},
            "context_only": {"context": "交易 停滞", "language": None},
            "forced_language": {"context": "", "language": "English"},
        }
        for name, spec in prompt_specs.items():
            _assert_equal(
                f"prompt[{name}]",
                golden["prompts"][name],
                model._build_text_prompt(
                    context=spec["context"], force_language=spec["language"]
                ),
            )

        for case_payload in golden["cases"]:
            if (
                selected_names is not None
                and case_payload["name"] not in selected_names
            ):
                continue
            step_payloads = _steps_by_ms(case_payload)
            missing_steps = sorted(selected_steps.difference(step_payloads))
            if missing_steps:
                raise ValueError(
                    f"Case {case_payload['name']} does not contain step_ms entries: {missing_steps}"
                )

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
                actual = run_streaming_case(
                    model=model,
                    wav16k=sliced_audio,
                    sample_rate=sample_rate,
                    case=case,
                    step_ms=step_ms,
                    chunk_size_sec=float(streaming_config["chunk_size_sec"]),
                    unfixed_chunk_num=int(streaming_config["unfixed_chunk_num"]),
                    unfixed_token_num=int(streaming_config["unfixed_token_num"]),
                    max_window_sec=float(max_window_sec)
                    if max_window_sec is not None
                    else None,
                    max_prefix_tokens=int(max_prefix_tokens)
                    if max_prefix_tokens is not None
                    else None,
                    timed=False,
                )
                expected = step_payloads[step_ms]
                _assert_equal(
                    f"case[{case.name}][{step_ms}].snapshots",
                    expected["snapshots"],
                    actual["snapshots"],
                )
                _assert_equal(
                    f"case[{case.name}][{step_ms}].final",
                    expected["final"],
                    actual["final"],
                )
                _assert_equal(
                    f"case[{case.name}][{step_ms}].metrics",
                    expected["metrics"],
                    actual["metrics"],
                )
                ran.append(f"{case.name}:{step_ms}")
    finally:
        model = None
        _dispose_model()

    if not ran:
        raise ValueError("No streaming regression cases were selected.")

    print(
        "Streaming regression passed.",
        {
            "golden": args.golden,
            "model": model_name,
            "audio": audio_path,
            "cases": ran,
        },
    )


if __name__ == "__main__":
    main()
