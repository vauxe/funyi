# coding=utf-8
"""
Streaming CER sweep across audio windows.

This is the streaming counterpart of tools/sweep_cer_vs_srt.py. It feeds each
window through Qwen3ASRModel.streaming_transcribe(), evaluates the final text
against the SRT reference, and records latency/stability metrics for the partial
updates observed along the way.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime import Qwen3ASRModel
from qwen3_asr_runtime.utils import normalize_language_name, validate_language
from tools.streaming_regression_common import StreamingCaseSpec, run_streaming_case
from tools.sweep_cer_vs_srt import _cer, _normalize_for_cer, _pct, load_srt, srt_text_in_window
from tools.runtime_helpers import _dispose_model

_RESUME_ARG_KEYS = (
    "audio",
    "srt",
    "model",
    "dtype",
    "device_map",
    "attn_implementation",
    "window_sec",
    "num_windows",
    "stride_sec",
    "start_sec",
    "step_ms",
    "chunk_size_sec",
    "unfixed_chunk_num",
    "unfixed_token_num",
    "max_window_sec",
    "max_prefix_tokens",
    "max_new_tokens",
    "strip_ruby",
    "language",
    "flashinfer",
    "cuda_graph_len_bucket",
    "fused_rmsnorm",
    "fused_linears",
    "quantized_linears",
    "spec_decode",
    "timed",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming CER sweep vs SRT reference.")
    parser.add_argument("--audio", required=True, help="Local audio file to evaluate.")
    parser.add_argument("--srt", required=True, help="Reference SRT file for the same audio.")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--num-windows", type=int, default=200)
    parser.add_argument("--stride-sec", type=float, default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--step-ms", default="2000", help="Comma separated streaming push step sizes.")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--max-window-sec", type=float, default=None, help="Optional bounded live-audio model window.")
    parser.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=None,
        help="Optional rolling text-prefix token cap. Defaults inside runtime when --max-window-sec is set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--paths", default="opt_nograph", help="Comma subset of: base, opt_nograph, graph.")
    parser.add_argument("--output", default="artifacts/streaming_cer_sweep.json")
    parser.add_argument("--resume", action="store_true", help="Skip windows whose results already exist in --output.")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush JSON after this many completed path rows.")
    parser.add_argument(
        "--strip-ruby",
        action="store_true",
        help="Remove Japanese-style furigana annotations like （さいがい） from the SRT reference.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a known ASR language, e.g. Chinese. Empty string keeps auto language detection.",
    )
    parser.add_argument("--flashinfer", action="store_true", help="Use FlashInfer decode attention for optimized paths.")
    parser.add_argument("--cuda-graph-len-bucket", type=int, default=1, help="Round CUDA graph/cache length up to this token bucket for graph paths.")
    parser.add_argument("--fused-rmsnorm", action="store_true", help="Use F.rms_norm for optimized paths.")
    parser.add_argument("--fused-linears", action="store_true", help="Use fused q/k/v and gate/up linears for optimized paths.")
    parser.add_argument("--quantized-linears", action="store_true", help="Use W8A16 for fused qkv/gate_up.")
    parser.add_argument(
        "--spec-decode",
        action="store_true",
        help="Speculative verification of the rollback prefix. Validate quality with a local CER sweep.",
    )
    parser.add_argument(
        "--timed",
        action="store_true",
        help="Synchronize CUDA and record per-push latency. Slower but needed for performance gates.",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not steps:
        raise ValueError("At least one --step-ms value is required.")
    for step in steps:
        if step <= 0:
            raise ValueError(f"--step-ms values must be > 0, got {step}")
    return steps


def _parse_paths(value: str) -> list[str]:
    paths = [item.strip() for item in str(value).split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one path is required.")
    unknown = sorted(set(paths).difference({"base", "opt_nograph", "graph"}))
    if unknown:
        raise ValueError(f"Unknown paths: {unknown}. Supported: base, opt_nograph, graph")
    return paths


def _flush(path: Optional[str], data: Dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resume_config(args_or_dict: argparse.Namespace | dict[str, Any]) -> dict[str, Any]:
    if isinstance(args_or_dict, dict):
        values = {key: args_or_dict.get(key) for key in _RESUME_ARG_KEYS}
    else:
        values = {key: getattr(args_or_dict, key) for key in _RESUME_ARG_KEYS}
    if values.get("language") is not None:
        language = str(values["language"]).strip()
        values["language"] = normalize_language_name(language) if language else None
    return values


def _validate_resume_args(previous_args: dict[str, Any], current_args: argparse.Namespace, output: str) -> None:
    previous = _resume_config(previous_args)
    current = _resume_config(current_args)
    mismatches = {
        key: {"previous": previous.get(key), "current": current.get(key)}
        for key in _RESUME_ARG_KEYS
        if previous.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            f"--resume output {output} was generated with different sweep args. "
            f"Use a new --output path or delete the old file. Mismatches: {mismatches}"
        )


def _load_audio_16k(path: str, sample_rate: int) -> np.ndarray:
    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if int(file_sr) == int(sample_rate):
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=file_sr, target_sr=sample_rate).astype(np.float32)


def _safe_empty_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cuda_oom_error(ex: BaseException) -> bool:
    text = f"{type(ex).__name__}: {ex}".lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cuda error: out of memory" in text
    )


def _build_windows(audio_sec: float, *, start_sec: float, window_sec: float, stride_sec: float, num_windows: int) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for idx in range(int(num_windows)):
        start = float(start_sec) + idx * float(stride_sec)
        if start + float(window_sec) > float(audio_sec):
            break
        windows.append((start, float(window_sec)))
    return windows


def _extract_active_update_walls(payload: dict[str, Any]) -> list[float]:
    timing = payload.get("timing") or {}
    return [
        float(event["wall_sec"])
        for event in timing.get("events", [])
        if int(event.get("decode_steps") or 0) > 0
    ]


def _summarize_partial_stability(payload: dict[str, Any]) -> dict[str, Any]:
    snapshots = payload["snapshots"]
    text_hashes = [item["text_sha256"] for item in snapshots if item["event"] in {"push", "finish"}]
    non_empty = [item for item in snapshots if item.get("text_chars", 0) > 0]
    text_changes = 0
    prev_hash = None
    for item in snapshots:
        if item["event"] not in {"push", "finish"}:
            continue
        cur_hash = item["text_sha256"]
        if prev_hash is not None and cur_hash != prev_hash:
            text_changes += 1
        prev_hash = cur_hash
    unique_texts = len(set(text_hashes))
    return {
        "snapshot_count": len(snapshots),
        "non_empty_snapshots": len(non_empty),
        "unique_partial_texts": unique_texts,
        "partial_text_changes": text_changes,
        "first_text_audio_ms": payload["metrics"]["first_text_audio_ms"],
    }


def _load_model(args: argparse.Namespace, path_name: str) -> Qwen3ASRModel:
    init_kwargs: dict[str, Any] = {
        "backend": "transformers",
        "dtype": _dtype(args.dtype),
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "max_inference_batch_size": 32,
        "max_new_tokens": int(args.max_new_tokens),
    }
    if path_name in {"opt_nograph", "graph"}:
        if args.flashinfer:
            init_kwargs["flashinfer"] = True
        if args.fused_rmsnorm:
            init_kwargs["fused_rmsnorm"] = True
        if args.fused_linears:
            init_kwargs["fused_linears"] = True
        if args.quantized_linears:
            init_kwargs["quantized_linears"] = True
    if path_name == "graph":
        init_kwargs["cuda_graph"] = True
        init_kwargs["cuda_graph_len_bucket"] = int(args.cuda_graph_len_bucket)
    model = Qwen3ASRModel.from_pretrained(args.model, **init_kwargs)
    model.eval()
    return model


def main() -> None:
    args = _parse_args()
    selected_paths = _parse_paths(args.paths)
    step_ms_list = _parse_steps(args.step_ms)
    stride_sec = args.stride_sec if args.stride_sec is not None else args.window_sec
    sample_rate = 16000
    force_language = None
    if args.language is not None and str(args.language).strip():
        force_language = normalize_language_name(str(args.language))
        validate_language(force_language)

    info = sf.info(args.audio)
    windows = _build_windows(
        info.duration,
        start_sec=args.start_sec,
        window_sec=args.window_sec,
        stride_sec=stride_sec,
        num_windows=args.num_windows,
    )
    srt_entries = load_srt(args.srt, strip_ruby=args.strip_ruby)
    wav_full = _load_audio_16k(args.audio, sample_rate)

    print(
        f"streaming CER sweep: windows={len(windows)} window_sec={args.window_sec} "
        f"steps={step_ms_list} paths={selected_paths} timed={bool(args.timed)} "
        f"max_window_sec={args.max_window_sec} max_prefix_tokens={args.max_prefix_tokens} "
        f"language={force_language}"
    )

    existing: dict[tuple[int, int], dict[str, Any]] = {}
    if args.resume and Path(args.output).exists():
        prev = json.loads(Path(args.output).read_text(encoding="utf-8"))
        previous_args = prev.get("args")
        if not isinstance(previous_args, dict):
            raise ValueError(f"--resume output {args.output} does not contain args metadata; use a new output path.")
        _validate_resume_args(previous_args, args, args.output)
        for row in prev.get("windows", []):
            existing[(int(row["idx"]), int(row["step_ms"]))] = row
        print(f"resume: {len(existing)} window/step rows already present in {args.output}")

    row_by_key: dict[tuple[int, int], dict[str, Any]] = dict(existing)
    completed = 0
    had_error = False
    fatal_error: str | None = None

    def get_row(idx: int, start_sec: float, dur_sec: float, step_ms: int, ref_chars: int) -> dict[str, Any]:
        key = (idx, step_ms)
        row = row_by_key.get(key)
        if row is not None:
            return row
        row = {
            "idx": idx,
            "start_sec": start_sec,
            "duration_sec": dur_sec,
            "step_ms": step_ms,
            "ref_chars": ref_chars,
            "ref_empty": ref_chars == 0,
            "paths": {},
        }
        row_by_key[key] = row
        return row

    for path in selected_paths:
        if had_error:
            break
        print(f"loading path={path}")
        model = _load_model(args, path)
        try:
            for idx, (start_sec, dur_sec) in enumerate(windows):
                if had_error:
                    break
                s = int(round(start_sec * sample_rate))
                e = s + int(round(dur_sec * sample_rate))
                wav = wav_full[s:e]
                ref_text = srt_text_in_window(srt_entries, start_sec, dur_sec)
                ref_chars = len(_normalize_for_cer(ref_text))
                case = StreamingCaseSpec(
                    name=f"window_{idx}",
                    start_sec=float(start_sec),
                    duration_sec=float(dur_sec),
                    context="",
                    language=force_language,
                )

                for step_ms in step_ms_list:
                    row = get_row(idx, start_sec, dur_sec, step_ms, ref_chars)
                    if path in row["paths"]:
                        print(
                            f"  [{idx+1:3d}/{len(windows)}] start={start_sec:7.1f}s "
                            f"step={step_ms:4d}ms path={path} (resumed)"
                        )
                        continue

                    _safe_empty_cache()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    try:
                        payload = run_streaming_case(
                            model=model,
                            wav16k=wav,
                            sample_rate=sample_rate,
                            case=case,
                            step_ms=step_ms,
                            chunk_size_sec=float(args.chunk_size_sec),
                            unfixed_chunk_num=int(args.unfixed_chunk_num),
                            unfixed_token_num=int(args.unfixed_token_num),
                            max_window_sec=args.max_window_sec,
                            max_prefix_tokens=args.max_prefix_tokens,
                            timed=bool(args.timed),
                            spec_decode=bool(args.spec_decode),
                        )
                        if not args.timed and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        wall_sec = time.perf_counter() - t0
                        text = payload["final"]["text"]
                        cer = _cer(text, ref_text)
                        active_update_walls = _extract_active_update_walls(payload)
                        path_row = {
                            "tokens": None,
                            "chars": len(_normalize_for_cer(text)),
                            "cer": round(cer, 6),
                            "wall_sec": round(wall_sec, 3),
                            "error": None,
                            "model_decode_updates": payload["metrics"]["model_decode_updates"],
                            "push_count": payload["metrics"]["push_count"],
                            "final_text_sha256": payload["final"]["text_sha256"],
                            "stability": _summarize_partial_stability(payload),
                        }
                        if args.timed:
                            path_row["active_update_wall_mean_sec"] = round(mean(active_update_walls), 4) if active_update_walls else None
                            path_row["active_update_wall_p50_sec"] = round(median(active_update_walls), 4) if active_update_walls else None
                            path_row["active_update_wall_p95_sec"] = round(_pct(active_update_walls, 0.95), 4) if active_update_walls else None
                    except Exception as ex:
                        if not _is_cuda_oom_error(ex):
                            raise
                        _safe_empty_cache()
                        gc.collect()
                        path_row = {
                            "tokens": None,
                            "chars": 0,
                            "cer": None,
                            "wall_sec": 0.0,
                            "error": "OOM",
                            "model_decode_updates": 0,
                            "push_count": 0,
                            "final_text_sha256": "",
                            "stability": {},
                        }
                        had_error = True
                        fatal_error = f"{path} window={idx} step_ms={step_ms}: {type(ex).__name__}: {ex}"
                    row["paths"][path] = path_row
                    completed += 1

                    line = f"  [{idx+1:3d}/{len(windows)}] start={start_sec:7.1f}s step={step_ms:4d}ms ref_chars={ref_chars:4d} "
                    pp = row["paths"][path]
                    cer_text = "ERR" if pp["cer"] is None else f"{pp['cer']:.3f}"
                    line += f"{path}:cer={cer_text} wall={pp['wall_sec']:.2f}s"
                    print(line)

                    flush_every = max(1, int(args.flush_every))
                    if args.output and completed % flush_every == 0:
                        partial_rows = sorted(row_by_key.values(), key=lambda item: (int(item["idx"]), int(item["step_ms"])))
                        _flush(args.output, {"args": vars(args), "windows": partial_rows})
                    if had_error:
                        break
        finally:
            model = None
            _dispose_model()

    rows = sorted(row_by_key.values(), key=lambda item: (int(item["idx"]), int(item["step_ms"])))
    for row in rows:
        if "base" not in row["paths"]:
            continue
        base_cer = row["paths"]["base"]["cer"]
        for path in selected_paths:
            if path == "base" or path not in row["paths"] or base_cer is None or row["paths"][path]["cer"] is None:
                continue
            row["paths"][path]["delta_vs_base"] = round(row["paths"][path]["cer"] - base_cer, 6)

    non_empty = [row for row in rows if not row.get("ref_empty")]
    print()
    print(f"Aggregate over {len(non_empty)} non-empty window/step rows:")
    summary: dict[str, Any] = {}
    for path in selected_paths:
        path_rows = [row for row in non_empty if path in row["paths"]]
        cer_vals = [row["paths"][path]["cer"] for row in path_rows if row["paths"][path]["cer"] is not None]
        wall_vals = [row["paths"][path]["wall_sec"] for row in path_rows if row["paths"][path].get("wall_sec", 0) > 0]
        update_vals = [
            row["paths"][path].get("active_update_wall_p95_sec")
            for row in path_rows
            if row["paths"][path].get("active_update_wall_p95_sec") is not None
        ]
        first_text_vals = [
            row["paths"][path]["stability"].get("first_text_audio_ms")
            for row in path_rows
            if row["paths"][path].get("stability") and row["paths"][path]["stability"].get("first_text_audio_ms") is not None
        ]
        entry: dict[str, Any] = {
            "n_windows": len({int(row["idx"]) for row in path_rows if row["paths"][path].get("cer") is not None}),
            "n_rows": len(cer_vals),
            "cer_mean": round(mean(cer_vals), 4) if cer_vals else None,
            "cer_p50": round(median(cer_vals), 4) if cer_vals else None,
            "cer_p90": round(_pct(cer_vals, 0.9), 4) if cer_vals else None,
            "cer_max": round(max(cer_vals), 4) if cer_vals else None,
            "wall_mean_sec": round(mean(wall_vals), 3) if wall_vals else None,
            "wall_p50_sec": round(median(wall_vals), 3) if wall_vals else None,
            "first_text_audio_ms_mean": round(mean(first_text_vals), 1) if first_text_vals else None,
        }
        if update_vals:
            entry["active_update_p95_mean_sec"] = round(mean(update_vals), 4)
            entry["active_update_p95_max_sec"] = round(max(update_vals), 4)
        if path != "base" and "base" in selected_paths:
            deltas = [
                row["paths"][path]["delta_vs_base"]
                for row in path_rows
                if "delta_vs_base" in row["paths"][path]
            ]
            abs_deltas = [abs(value) for value in deltas]
            entry["delta_mean"] = round(mean(deltas), 5) if deltas else None
            entry["delta_abs_mean"] = round(mean(abs_deltas), 5) if abs_deltas else None
            entry["delta_abs_max"] = round(max(abs_deltas), 5) if abs_deltas else None
        summary[path] = entry
        print(f"  {path}: {entry}")

    final = {"args": vars(args), "summary": summary, "windows": rows}
    if fatal_error is not None:
        final["fatal_error"] = fatal_error
    _flush(args.output, final)
    if args.output:
        print(f"\nwrote {args.output}")
    if fatal_error is not None:
        raise RuntimeError(f"Streaming CER sweep stopped after CUDA OOM: {fatal_error}")


if __name__ == "__main__":
    main()
