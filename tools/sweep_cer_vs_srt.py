# coding=utf-8
"""
Sweep a long audio into windows, transcribe each window with one of two
runtime decode paths, and compare to the SRT reference using
punctuation-stripped CER:

  base  - runtime default path (model.generate with DynamicCache)
  graph - CudaGraphDecoder (StaticCache + captured graph)

This lets us compare CER vectors from multiple runtime paths on the same
audio/SRT windows. tools/merge_cer_sweeps.py merges the per-path JSONs.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import numpy as np
from rapidfuzz.distance import Levenshtein
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime import Qwen3ASRModel
from qwen3_asr_runtime.decode_runtime import CudaGraphDecoder
from qwen3_asr_runtime.utils import (
    normalize_language_name,
    parse_asr_output,
    validate_language,
)

_RESUME_ARG_KEYS = (
    "audio",
    "srt",
    "dtype",
    "attn_implementation",
    "window_sec",
    "num_windows",
    "stride_sec",
    "start_sec",
    "max_new_tokens",
    "paths",
    "strip_ruby",
    "language",
    "flashinfer",
    "fused_rmsnorm",
    "fused_linears",
    "quantized_linears",
)


# ---------------------------------------------------------------------------
# CER utilities
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[\s　]+|[\w]*﻿[\w]*", flags=re.UNICODE)


def _normalize_for_cer(text: str) -> str:
    """Lowercase, strip all punctuation and whitespace."""
    text = text or ""
    text = text.strip()
    out = []
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            # punctuation or symbol
            continue
        out.append(ch.lower())
    return "".join(out)


def _cer(hyp: str, ref: str) -> float:
    hyp_n = _normalize_for_cer(hyp)
    ref_n = _normalize_for_cer(ref)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return Levenshtein.distance(hyp_n, ref_n) / len(ref_n)


# ---------------------------------------------------------------------------
# SRT loading
# ---------------------------------------------------------------------------


def _parse_srt_time(raw: str) -> float:
    text = raw.strip()
    if "," in text:
        hhmmss, millis = text.split(",", 1)
    elif "." in text:
        hhmmss, millis = text.split(".", 1)
    else:
        hhmmss, millis = text, "0"
    h, m, s = hhmmss.strip().split(":")
    millis = (millis.strip() or "0").ljust(3, "0")[:3]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis) / 1000.0


_RUBY_RE = re.compile(r"[（(][^）)]{1,8}[）)]")


def strip_ruby_annotations(text: str) -> str:
    """Remove furigana-style parenthesized annotations, e.g. 災害（さいがい） -> 災害."""
    prev = None
    while prev != text:
        prev = text
        text = _RUBY_RE.sub("", text)
    return text


def load_srt(path: str, *, strip_ruby: bool = False) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    blocks = Path(path).read_text(encoding="utf-8").split("\n\n")
    for block in blocks:
        raw = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(raw) < 2:
            continue
        if "-->" not in raw[1]:
            continue
        start_raw, end_raw = [s.strip() for s in raw[1].split("-->", 1)]
        start_sec = _parse_srt_time(start_raw)
        end_sec = _parse_srt_time(end_raw)
        text = "".join(raw[2:])
        if strip_ruby:
            text = strip_ruby_annotations(text)
        entries.append({"start": start_sec, "end": end_sec, "text": text})
    return entries


def srt_text_in_window(
    entries: List[Dict[str, Any]], start: float, duration: float
) -> str:
    end = start + duration
    out = []
    for e in entries:
        # include an entry if any overlap with [start, end]
        if e["end"] <= start or e["start"] >= end:
            continue
        out.append(e["text"])
    return "".join(out)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CER sweep across windows and runtime paths vs SRT reference."
    )
    p.add_argument("--audio", required=True, help="Local audio file to evaluate.")
    p.add_argument(
        "--srt", required=True, help="Reference SRT file for the same audio."
    )
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--num-windows", type=int, default=200)
    p.add_argument("--stride-sec", type=float, default=None)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--paths",
        default="base",
        help="Comma subset of: base, graph. Run one path per process to avoid cross-path allocator issues.",
    )
    p.add_argument("--output", default="artifacts/cer_sweep.json")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip windows whose results already exist in --output.",
    )
    p.add_argument(
        "--strip-ruby",
        action="store_true",
        help="Remove Japanese-style furigana annotations like （さいがい） from the SRT reference.",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Force a known ASR language, e.g. Chinese. Empty string keeps auto language detection.",
    )
    p.add_argument(
        "--flashinfer",
        action="store_true",
        help="Use flashinfer's single_decode attention kernel.",
    )
    p.add_argument(
        "--fused-rmsnorm",
        action="store_true",
        help="Replace hand-rolled RMSNorm with torch.nn.functional.rms_norm.",
    )
    p.add_argument(
        "--fused-linears",
        action="store_true",
        help="Fuse q/k/v and gate/up linear projections per layer.",
    )
    p.add_argument(
        "--quantized-linears",
        action="store_true",
        help="Use W8A16 for fused qkv/gate_up.",
    )
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _decode_tokens(tokenizer, ids: List[int], *, user_language: Optional[str]) -> str:
    if not ids:
        return ""
    eos = {151645, 151643}
    while ids and ids[-1] in eos:
        ids = ids[:-1]
    raw = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    _, text = parse_asr_output(raw, user_language=user_language)
    return text


def _flush(path: Optional[str], data: Dict[str, Any]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _parse_paths(value: str) -> List[str]:
    paths = [p.strip() for p in str(value).split(",") if p.strip()]
    for path in paths:
        if path not in {"base", "graph"}:
            raise ValueError(f"unknown path: {path}")
    if not paths:
        raise ValueError("At least one path is required.")
    return paths


def _resume_config(args_or_dict: argparse.Namespace | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(args_or_dict, dict):
        values = {key: args_or_dict.get(key) for key in _RESUME_ARG_KEYS}
    else:
        values = {key: getattr(args_or_dict, key) for key in _RESUME_ARG_KEYS}
    if values.get("paths") is not None:
        values["paths"] = ",".join(_parse_paths(str(values["paths"])))
    if values.get("language") is not None:
        language = str(values["language"]).strip()
        values["language"] = normalize_language_name(language) if language else None
    return values


def _validate_resume_args(
    previous_args: Dict[str, Any], current_args: argparse.Namespace, output: str
) -> None:
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


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> None:
    args = _parse_args()
    dtype = _dtype(args.dtype)
    stride_sec = args.stride_sec if args.stride_sec is not None else args.window_sec
    sr = 16000
    selected_paths = _parse_paths(args.paths)
    force_language = None
    if args.language is not None and str(args.language).strip():
        force_language = normalize_language_name(str(args.language))
        validate_language(force_language)

    info = sf.info(args.audio)
    total_sec = info.duration
    windows = []
    for i in range(args.num_windows):
        start = args.start_sec + i * stride_sec
        if start + args.window_sec > total_sec:
            break
        windows.append((start, args.window_sec))

    srt_entries = load_srt(args.srt, strip_ruby=args.strip_ruby)
    print(
        f"loaded SRT: {len(srt_entries)} entries; sweeping {len(windows)} windows of "
        f"{args.window_sec}s  dtype={args.dtype}  attn={args.attn_implementation}  paths={selected_paths}"
    )

    existing: Dict[int, Dict[str, Any]] = {}
    if args.resume and Path(args.output).exists():
        prev = json.loads(Path(args.output).read_text(encoding="utf-8"))
        previous_args = prev.get("args")
        if not isinstance(previous_args, dict):
            raise ValueError(
                f"--resume output {args.output} does not contain args metadata; use a new output path."
            )
        _validate_resume_args(previous_args, args, args.output)
        for row in prev.get("windows", []):
            existing[int(row["idx"])] = row
        print(f"resume: {len(existing)} windows already present in {args.output}")

    kwargs = dict(
        dtype=dtype,
        device_map="cuda:0",
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=32,
    )
    if getattr(args, "flashinfer", False):
        kwargs["flashinfer"] = True
    if getattr(args, "fused_rmsnorm", False):
        kwargs["fused_rmsnorm"] = True
    if getattr(args, "fused_linears", False):
        kwargs["fused_linears"] = True
    if getattr(args, "quantized_linears", False):
        kwargs["quantized_linears"] = True
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        **kwargs,
    )
    model.eval()
    hf_model = model.backend_runtime.model
    thinker = hf_model.thinker
    processor = model.backend_runtime.processor
    tokenizer = processor.tokenizer
    prompt = model._build_text_prompt(context="", force_language=force_language)

    def fresh_decoder() -> CudaGraphDecoder:
        return CudaGraphDecoder(thinker)

    wav_full, file_sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav_full.ndim > 1:
        wav_full = wav_full.mean(axis=1)
    wav_full = wav_full.astype(np.float32)
    if file_sr != sr:
        import librosa

        wav_full = librosa.resample(wav_full, orig_sr=file_sr, target_sr=sr).astype(
            np.float32
        )
        print(f"resampled {file_sr}Hz -> {sr}Hz, {wav_full.shape[0]} samples")

    rows: List[Dict[str, Any]] = []

    def run_path(name: str, inputs: Dict[str, Any], prompt_len: int) -> Dict[str, Any]:
        thinker.rope_deltas = None
        _sync_cuda()
        t0 = time.perf_counter()
        if name == "base":
            seq = hf_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            ).sequences
        elif name == "graph":
            dec = fresh_decoder()
            seq = dec.generate(
                input_ids=inputs["input_ids"],
                input_features=inputs.get("input_features"),
                attention_mask=inputs["attention_mask"],
                feature_attention_mask=inputs.get("feature_attention_mask"),
                max_new_tokens=args.max_new_tokens,
            )
            del dec
        else:
            raise ValueError(name)
        _sync_cuda()
        wall = time.perf_counter() - t0
        ids = seq[0, prompt_len:].tolist()
        del seq
        torch.cuda.empty_cache()
        gc.collect()
        return {"ids": ids, "wall": wall}

    for idx, (start_sec, dur_sec) in enumerate(windows):
        if idx in existing:
            rows.append(existing[idx])
            print(f"  [{idx + 1:3d}/{len(windows)}] start={start_sec:7.1f}s  (resumed)")
            continue

        s = int(round(start_sec * sr))
        e = s + int(round(dur_sec * sr))
        wav = wav_full[s:e]
        ref_text = srt_text_in_window(srt_entries, start_sec, dur_sec)

        inputs = (
            processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
            .to("cuda:0")
            .to(dtype)
        )
        prompt_len = int(inputs["input_ids"].shape[1])

        out: Dict[str, Dict[str, Any]] = {}
        for p in selected_paths:
            try:
                out[p] = run_path(p, inputs, prompt_len)
            except torch.cuda.OutOfMemoryError as ex:
                torch.cuda.empty_cache()
                gc.collect()
                out[p] = {"ids": [], "wall": 0.0, "error": "OOM"}
                print(f"    path={p} OOM")

        texts = {
            p: _decode_tokens(tokenizer, out[p]["ids"], user_language=force_language)
            for p in selected_paths
        }
        cers = {p: _cer(texts[p], ref_text) for p in selected_paths}

        row = {
            "idx": idx,
            "start_sec": start_sec,
            "duration_sec": dur_sec,
            "ref_chars": len(_normalize_for_cer(ref_text)),
            "ref_empty": len(_normalize_for_cer(ref_text)) == 0,
            "paths": {
                p: {
                    "tokens": len(out[p]["ids"]),
                    "chars": len(_normalize_for_cer(texts[p])),
                    "cer": round(cers[p], 6),
                    "wall_sec": round(out[p]["wall"], 3),
                    "error": out[p].get("error"),
                }
                for p in selected_paths
            },
        }
        if "base" in selected_paths:
            for p in selected_paths:
                if p != "base":
                    row["paths"][p]["delta_vs_base"] = round(cers[p] - cers["base"], 6)

        rows.append(row)

        line = f"  [{idx + 1:3d}/{len(windows)}] start={start_sec:7.1f}s  ref_chars={row['ref_chars']:4d}  "
        for p in selected_paths:
            pp = row["paths"][p]
            line += f"{p}:cer={pp['cer']:.3f} "
        print(line)

        # flush every 5 windows
        if args.output and idx % 5 == 0:
            _flush(args.output, {"args": vars(args), "windows": rows})

    # ---- aggregate
    def _collect(field: str, filter_fn=None) -> List[float]:
        vals = []
        for r in rows:
            if filter_fn and not filter_fn(r):
                continue
            for p in selected_paths:
                if p == field.split("_", 1)[0]:
                    val = r["paths"][p].get(field.split("_", 1)[1])
                    if val is not None:
                        vals.append(val)
        return vals

    non_empty = [r for r in rows if not r.get("ref_empty")]
    print()
    print(f"Aggregate over {len(non_empty)} windows with non-empty SRT reference:")
    summary: Dict[str, Any] = {}
    for p in selected_paths:
        cer_vals = [
            r["paths"][p]["cer"] for r in non_empty if r["paths"][p]["cer"] is not None
        ]
        wall_vals = [
            r["paths"][p]["wall_sec"]
            for r in non_empty
            if r["paths"][p].get("wall_sec", 0) > 0
        ]
        entry = {
            "cer_mean": round(mean(cer_vals), 4) if cer_vals else None,
            "cer_p50": round(median(cer_vals), 4) if cer_vals else None,
            "cer_p90": round(_pct(cer_vals, 0.9), 4) if cer_vals else None,
            "cer_max": round(max(cer_vals), 4) if cer_vals else None,
            "wall_mean_sec": round(mean(wall_vals), 3) if wall_vals else None,
            "wall_p50_sec": round(median(wall_vals), 3) if wall_vals else None,
        }
        if p != "base" and "base" in selected_paths:
            deltas = [
                r["paths"][p]["delta_vs_base"]
                for r in non_empty
                if "delta_vs_base" in r["paths"][p]
            ]
            abs_deltas = [abs(value) for value in deltas]
            entry["delta_mean"] = round(mean(deltas), 5) if deltas else None
            entry["delta_p50"] = round(median(deltas), 5) if deltas else None
            entry["delta_p90"] = round(_pct(deltas, 0.9), 5) if deltas else None
            entry["delta_min"] = round(min(deltas), 5) if deltas else None
            entry["delta_max"] = round(max(deltas), 5) if deltas else None
            entry["delta_abs_mean"] = round(mean(abs_deltas), 5) if abs_deltas else None
            entry["delta_abs_max"] = round(max(abs_deltas), 5) if abs_deltas else None
        summary[p] = entry
        print(f"  {p}: {entry}")

    final = {"args": vars(args), "summary": summary, "windows": rows}
    _flush(args.output, final)
    if args.output:
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
