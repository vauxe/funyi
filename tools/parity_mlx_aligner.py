# coding=utf-8
"""Timestamp parity check for the MLX forced aligner against the official torch
aligner (per AGENTS.md, alignment quality is gated against the official-code
reference, never by byte equality).

Modes:
  * --reference-json: compare MLX timestamps against stored reference items
    ([{text,start_time,end_time}, ...]); offline gate, no torch.
  * live (--reference-model): run the torch Qwen3ForcedAlignerBackend on CPU and
    compare item-by-item (word count + max start/end drift).
  * --no-gate: run the MLX aligner only, print items, and check monotonicity.

Gate: identical word count AND max(|Δstart|, |Δend|) <= --max-drift seconds.

Example:
    uv run python tools/parity_mlx_aligner.py \
        --model mlx-community/Qwen3-ForcedAligner-0.6B-4bit \
        --reference-model <bf16 forced-aligner> \
        --wav local_data/clip.wav --text "..." --language Chinese
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16000


def _load_clip(path: str, seconds: float) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if seconds and seconds > 0:
        wav = wav[: int(seconds * SAMPLE_RATE)]
    return np.asarray(wav, dtype=np.float32)


def _items_to_list(result) -> list[dict]:
    return [
        {"text": it.text, "start_time": float(it.start_time), "end_time": float(it.end_time)}
        for it in result.items
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-ForcedAligner-0.6B-4bit", help="MLX (4bit) aligner checkpoint")
    ap.add_argument("--dtype", default="bfloat16", help="MLX compute dtype")
    ap.add_argument("--wav", required=True, help="audio clip")
    ap.add_argument("--text", required=True, help="transcript to align")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--seconds", type=float, default=0.0, help="trim clip to N seconds (0 = full)")
    ap.add_argument("--reference-model", default=None, help="bf16 torch forced-aligner for the live reference")
    ap.add_argument("--reference-json", default=None, help="JSON list of reference items to gate against")
    ap.add_argument("--max-drift", type=float, default=0.25, help="gate: max |Δstart|/|Δend| in seconds")
    ap.add_argument("--no-gate", action="store_true", help="run MLX aligner only; print items + monotonicity")
    args = ap.parse_args()

    from qwen3_asr_runtime.mlx_forced_aligner import MLXForcedAlignerBackend

    wav = _load_clip(args.wav, args.seconds)
    aligner = MLXForcedAlignerBackend.from_pretrained(args.model, dtype=args.dtype)
    mlx_items = _items_to_list(aligner.align((wav, SAMPLE_RATE), args.text, args.language)[0])

    print(f"\n=== MLX aligner: {len(mlx_items)} item(s) ===")
    for it in mlx_items:
        print(f"  [{it['start_time']:7.3f}, {it['end_time']:7.3f}]  {it['text']!r}")
    monotonic = all(
        mlx_items[i]["start_time"] <= mlx_items[i]["end_time"]
        and (i == 0 or mlx_items[i - 1]["start_time"] <= mlx_items[i]["start_time"])
        for i in range(len(mlx_items))
    )
    print(f"  monotonic = {monotonic}")

    if args.no_gate or (args.reference_model is None and args.reference_json is None):
        print("\ngate skipped (no reference)" if not args.no_gate else "\ngate skipped (--no-gate)")
        return 0 if monotonic else 1

    if args.reference_json:
        ref_items = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
    else:
        import torch  # noqa: F401
        from qwen3_asr_runtime.forced_aligner import Qwen3ForcedAlignerBackend

        print(f"\n[live reference] loading torch {args.reference_model} on CPU (float32) ...")
        ref = Qwen3ForcedAlignerBackend.from_pretrained(
            args.reference_model, device_map="cpu", dtype=torch.float32
        )
        ref_items = _items_to_list(ref.align((wav, SAMPLE_RATE), args.text, args.language)[0])

    if len(mlx_items) != len(ref_items):
        print(f"\nWORD COUNT MISMATCH: MLX={len(mlx_items)} vs ref={len(ref_items)}")
        print("GATE: FAIL")
        return 1

    max_drift = 0.0
    for m, r in zip(mlx_items, ref_items):
        d = max(abs(m["start_time"] - float(r["start_time"])), abs(m["end_time"] - float(r["end_time"])))
        max_drift = max(max_drift, d)
    print(f"\nmax timestamp drift = {max_drift:.3f}s over {len(mlx_items)} item(s)")
    ok = monotonic and max_drift <= args.max_drift
    print("GATE:", "PASS" if ok else "FAIL", f"(word count match, drift threshold {args.max_drift}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
