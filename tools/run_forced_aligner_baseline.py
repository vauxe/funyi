# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime.utils import SAMPLE_RATE, normalize_audios


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible Qwen3 forced-aligner baseline.")
    parser.add_argument("--implementation", required=True, choices=["local", "official"])
    parser.add_argument("--official-repo", default="Qwen3-ASR", help="Path to the upstream Qwen3-ASR checkout.")
    parser.add_argument("--model", required=True, help="Forced aligner checkpoint path or repo id.")
    parser.add_argument("--audio", required=True, help="Local audio file.")
    parser.add_argument("--srt", required=True, help="Reference SRT used to build alignment windows.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-audio-sec", type=float, default=600.0)
    parser.add_argument("--window-sec", type=float, default=90.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--compare-to", default=None, help="Optional official/local JSON to compare against.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(path: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out.strip()


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


def _load_srt(path: str, *, max_audio_sec: float) -> List[dict[str, Any]]:
    entries: List[dict[str, Any]] = []
    for block in Path(path).read_text(encoding="utf-8").split("\n\n"):
        raw = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(raw) < 2 or "-->" not in raw[1]:
            continue
        start_raw, end_raw = [s.strip() for s in raw[1].split("-->", 1)]
        start = _parse_srt_time(start_raw)
        end = _parse_srt_time(end_raw)
        if start >= max_audio_sec:
            continue
        entries.append({"start": start, "end": min(end, max_audio_sec), "text": "".join(raw[2:])})
    return entries


def _window_groups(entries: List[dict[str, Any]], *, window_sec: float) -> Iterable[List[dict[str, Any]]]:
    group: List[dict[str, Any]] = []
    group_start = 0.0
    for entry in entries:
        if not group:
            group = [entry]
            group_start = float(entry["start"])
            continue
        if float(entry["end"]) - group_start > window_sec:
            yield group
            group = [entry]
            group_start = float(entry["start"])
        else:
            group.append(entry)
    if group:
        yield group


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_aligner(args: argparse.Namespace) -> Any:
    kwargs = {
        "device_map": args.device_map,
        "dtype": _dtype(args.dtype),
        "attn_implementation": args.attn_implementation,
    }
    if args.implementation == "local":
        from qwen3_asr_runtime.forced_aligner import Qwen3ForcedAlignerBackend

        return Qwen3ForcedAlignerBackend.from_pretrained(args.model, **kwargs)

    official_repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(official_repo))
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    return Qwen3ForcedAligner.from_pretrained(args.model, **kwargs)


def _result_items(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "text": item.text,
            "start_time": float(item.start_time),
            "end_time": float(item.end_time),
        }
        for item in result.items
    ]


def _token_count(aligner: Any, text: str, language: str) -> int:
    return len(aligner.aligner_processor.encode_timestamp(text, language)[0])


def _run_once(aligner: Any, wav: np.ndarray, entries: List[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    windows = []
    sentences: list[dict[str, Any]] = []
    for index, group in enumerate(_window_groups(entries, window_sec=args.window_sec)):
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        crop = wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        text = " ".join(str(entry["text"]) for entry in group)
        result = aligner.align(audio=(crop, SAMPLE_RATE), text=text, language=args.language)[0]
        items = _result_items(result)
        windows.append({"index": index, "start": start, "end": end, "text": text, "items": items})

        cursor = 0
        for entry in group:
            count = _token_count(aligner, str(entry["text"]), args.language)
            aligned = items[cursor : cursor + count]
            if aligned:
                sentences.append(
                    {
                        "text": entry["text"],
                        "reference_start": entry["start"],
                        "reference_end": entry["end"],
                        "start_time": round(start + aligned[0]["start_time"], 3),
                        "end_time": round(start + aligned[-1]["end_time"], 3),
                    }
                )
            else:
                sentences.append(
                    {
                        "text": entry["text"],
                        "reference_start": entry["start"],
                        "reference_end": entry["end"],
                        "start_time": None,
                        "end_time": None,
                    }
                )
            cursor += count
    return {"windows": windows, "sentences": sentences}


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    official_repo = Path(args.official_repo).resolve()
    return {
        "implementation": args.implementation,
        "seed": args.seed,
        "repo": {
            "path": str(ROOT),
            "commit": _git(ROOT, "rev-parse", "HEAD"),
            "dirty": _git(ROOT, "status", "--short"),
        },
        "official_repo": {
            "path": str(official_repo),
            "commit": _git(official_repo, "rev-parse", "HEAD"),
            "dirty": _git(official_repo, "status", "--short"),
        },
        "model": args.model,
        "audio": args.audio,
        "audio_sha256": _sha256(args.audio),
        "srt": args.srt,
        "srt_sha256": _sha256(args.srt),
        "language": args.language,
        "max_audio_sec": args.max_audio_sec,
        "window_sec": args.window_sec,
        "repeat": args.repeat,
        "load_kwargs": {
            "device_map": args.device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def _assert_same(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    for key in ("windows", "sentences"):
        if expected[key] != actual[key]:
            for idx, (left, right) in enumerate(zip(expected[key], actual[key])):
                if left != right:
                    raise AssertionError(
                        f"{key} mismatch at index {idx}:\n"
                        f"EXPECTED: {_short_repr(left)}\n"
                        f"ACTUAL  : {_short_repr(right)}"
                    )
            raise AssertionError(f"{key} length mismatch: expected={len(expected[key])} actual={len(actual[key])}")
    return {
        "status": "matched",
        "windows": len(actual["windows"]),
        "sentences": len(actual["sentences"]),
    }


def _short_repr(value: Any, *, limit: int = 2000) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "... <truncated>"


def main() -> None:
    args = _parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    _set_seed(args.seed)
    load_start = time.perf_counter()
    aligner = _load_aligner(args)
    load_sec = time.perf_counter() - load_start
    wav = normalize_audios(args.audio)[0]
    if args.max_audio_sec > 0:
        wav = wav[: int(args.max_audio_sec * SAMPLE_RATE)]
    entries = _load_srt(args.srt, max_audio_sec=args.max_audio_sec)

    times = []
    result: dict[str, Any] = {}
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = _run_once(aligner, wav, entries, args)
        times.append(time.perf_counter() - start)

    payload = {
        **_metadata(args),
        "entries": len(entries),
        "load_sec": round(load_sec, 3),
        "time_sec": {
            "values": [round(t, 3) for t in times],
            "min": round(min(times), 3),
            "avg": round(statistics.mean(times), 3),
            "median": round(statistics.median(times), 3),
        },
        **result,
    }
    if args.compare_to:
        expected = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        payload["comparison"] = _assert_same(expected, result)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("implementation", "entries", "load_sec", "time_sec")}, ensure_ascii=False, indent=2))
    if "comparison" in payload:
        print(json.dumps({"comparison": payload["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
