# coding=utf-8
"""Materialize a large-scale HY-MT translation eval set from opus-100 test.

Writes a JSONL in the same case schema consumed by ``tools/gate_translation.py``
so the translation quality gate can run a ~1200-sentence paired regression
instead of only the 42 in-domain cases.

opus-100 is public CC text (not audio-derived), but the generated file is kept
in ``local_data/`` (git-ignored) per the project convention. The download needs
``huggingface_hub`` and ``pyarrow``; the resulting JSONL is then read offline by
the gate forever, so these are only needed when (re)generating the set.

Coverage: opus-100 is English-centric, so this yields en<->zh and en<->ja
(four directions). zh<->ja is not available here and stays on the in-domain set.

Caveat: opus-100 references are loose subtitle alignments (noisy). Absolute chrF
is unreliable; trust the paired base-vs-candidate delta, which cancels ref noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# opus-100 config -> (source short, target short) for the two English-centric pairs.
_PAIRS = (("en-zh", "en", "zh"), ("en-ja", "en", "ja"))
_LANG_NAME = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
# Pin an immutable commit so regenerating the set is reproducible and an upstream
# repo change cannot silently swap sentences under the same positional case ids.
_OPUS_REVISION = "805090dc28bf78897da9641cdf08b61287580df9"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an opus-100 paired translation eval JSONL.")
    parser.add_argument("--output", default=str(ROOT / "local_data" / "opus_mt_eval.jsonl"))
    parser.add_argument(
        "--per-dir",
        type=int,
        default=300,
        help="Sentences per language pair; each pair yields 2 directions, 2 pairs => 4 directions "
        "(e.g. 300 -> 300 cases in each of the 4 directions, 1200 total).",
    )
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=400)
    return parser.parse_args()


def _stride(items: list, count: int) -> list:
    """Evenly sample ``count`` items across the list (deterministic, no RNG)."""
    if count >= len(items):
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def _load_pair(config: str, a: str, b: str, *, min_chars: int, max_chars: int) -> list[dict[str, str]]:
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - fetch-time only dependency
        raise SystemExit(
            f"fetch needs huggingface_hub and pyarrow ({exc}); install them in a throwaway env, "
            "e.g. `uv venv /tmp/ev && uv pip install --python /tmp/ev huggingface_hub pyarrow`, "
            "then run this script with /tmp/ev/bin/python."
        ) from exc
    path = hf_hub_download(
        "Helsinki-NLP/opus-100",
        f"{config}/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=_OPUS_REVISION,
    )
    table = pq.read_table(path).column("translation").to_pylist()
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in table:
        src = (item.get(a) or "").strip()
        ref = (item.get(b) or "").strip()
        if not src or not ref or not (min_chars <= len(src) <= max_chars):
            continue
        if src in seen:
            continue
        seen.add(src)
        pairs.append({"src": src, "ref": ref})
    return pairs


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, object]] = []
    for config, a, b in _PAIRS:
        pairs = _load_pair(config, a, b, min_chars=args.min_chars, max_chars=args.max_chars)
        sampled = _stride(pairs, args.per_dir)
        # Each pair yields both directions: src->ref and ref->src.
        for index, pair in enumerate(sampled):
            rows.append(_case(f"opus_{a}-{b}_{index:04d}", a, b, pair["src"], pair["ref"]))
            rows.append(_case(f"opus_{b}-{a}_{index:04d}", b, a, pair["ref"], pair["src"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        # Provenance header; the gate's case loader skips lines starting with '#'.
        handle.write(
            f"# opus-100 test @ {_OPUS_REVISION} | per_pair={args.per_dir} "
            f"| chars {args.min_chars}-{args.max_chars} | {len(rows)} cases\n"
        )
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    directions = sorted({f"{row['source_language']}->{row['target_language']}" for row in rows})
    print(f"wrote {len(rows)} cases to {output} across {len(directions)} directions: {directions}", file=sys.stderr)


def _case(case_id: str, src_short: str, tgt_short: str, text: str, reference: str) -> dict[str, object]:
    return {
        "id": case_id,
        "group": "opus_mt",
        "use_case": "opus100_test",
        "split": "test",
        "source_language": _LANG_NAME[src_short],
        "target_language": _LANG_NAME[tgt_short],
        "text": text,
        "reference": reference,
    }


if __name__ == "__main__":
    main()
