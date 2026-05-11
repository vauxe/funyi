# coding=utf-8
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime.forced_aligner import Qwen3ForcedAlignerBackend
from qwen3_asr_runtime.utils import SAMPLE_RATE, normalize_audios
from tools.run_forced_aligner_baseline import _dtype, _load_srt, _window_groups


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Qwen3 forced-aligner stages on local validation data.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-audio-sec", type=float, default=600.0)
    parser.add_argument("--window-sec", type=float, default=90.0)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _elapsed(fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync()
    start = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - start


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "sum": round(sum(values), 6),
        "avg": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


class _ModuleTimer:
    def __init__(self, module: Any, attr: str) -> None:
        self.module = module
        self.attr = attr
        self.original = getattr(module, attr)
        self.times: list[float] = []

    def __enter__(self) -> "_ModuleTimer":
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out, elapsed = _elapsed(lambda: self.original(*args, **kwargs))
            self.times.append(elapsed)
            return out

        setattr(self.module, self.attr, wrapped)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        setattr(self.module, self.attr, self.original)


def _profile_window(
    aligner: Qwen3ForcedAlignerBackend,
    audio: np.ndarray,
    text: str,
    language: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    words, prompt = aligner.aligner_processor.encode_timestamp(text, language)
    row["word_count"] = len(words)

    inputs, row["processor_sec"] = _elapsed(
        lambda: aligner.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
    )
    row["seq_len"] = int(inputs["input_ids"].shape[1])
    row["timestamp_positions"] = int((inputs["input_ids"] == aligner.timestamp_token_id).sum().item())
    row["input_features_shape"] = list(inputs["input_features"].shape)
    aligner._drop_single_full_attention_mask(inputs)
    aligner._drop_single_full_feature_mask(inputs)

    inputs, row["move_inputs_sec"] = _elapsed(lambda: aligner._move_inputs_like_official(inputs))
    timestamp_mask = inputs["input_ids"][0] == aligner.timestamp_token_id

    with _ModuleTimer(aligner.model.thinker, "get_audio_features") as audio_timer:
        with _ModuleTimer(aligner.model.thinker, "get_rope_index") as rope_timer:
            with _ModuleTimer(aligner.model.thinker.model, "forward") as text_timer:
                with _ModuleTimer(aligner.model.thinker.lm_head, "forward") as lm_head_timer:
                    outputs, row["thinker_sec"] = _elapsed(lambda: aligner.model.thinker(**inputs))

    row["audio_encoder_sec"] = round(sum(audio_timer.times), 6)
    row["rope_index_sec"] = round(sum(rope_timer.times), 6)
    row["text_decoder_sec"] = round(sum(text_timer.times), 6)
    row["lm_head_sec"] = round(sum(lm_head_timer.times), 6)
    logits = outputs.logits
    row["logits_shape"] = list(logits.shape)

    timestamp_logits = logits[0, timestamp_mask, :]
    output_ids, row["timestamp_argmax_sec"] = _elapsed(lambda: timestamp_logits.argmax(dim=-1))
    timestamp_ms = (output_ids * aligner.timestamp_segment_time).to("cpu").numpy()
    _, row["parse_sec"] = _elapsed(lambda: aligner.aligner_processor.parse_timestamp(words, timestamp_ms))
    return row


def main() -> None:
    args = _parse_args()
    aligner = Qwen3ForcedAlignerBackend.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    wav = normalize_audios(args.audio)[0]
    if args.max_audio_sec > 0:
        wav = wav[: int(args.max_audio_sec * SAMPLE_RATE)]
    entries = _load_srt(args.srt, max_audio_sec=args.max_audio_sec)

    rows = []
    groups = list(_window_groups(entries, window_sec=args.window_sec))
    if args.limit_windows > 0:
        groups = groups[: args.limit_windows]
    for index, group in enumerate(groups):
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        audio = wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        text = " ".join(str(entry["text"]) for entry in group)
        row = _profile_window(aligner, audio, text, args.language)
        row.update({"window": index, "start": start, "end": end, "duration": round(end - start, 3)})
        rows.append(row)

    keys = [
        "processor_sec",
        "move_inputs_sec",
        "thinker_sec",
        "audio_encoder_sec",
        "rope_index_sec",
        "text_decoder_sec",
        "lm_head_sec",
        "timestamp_argmax_sec",
        "parse_sec",
    ]
    payload = {
        "windows": len(rows),
        "summary": {key: _stats([float(row[key]) for row in rows]) for key in keys},
        "rows": rows,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
