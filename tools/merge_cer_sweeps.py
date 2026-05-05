# coding=utf-8
"""
Merge multiple single-path cer sweep outputs into one combined report.

Usage:
  python tools/merge_cer_sweeps.py \
      --input artifacts/cer_base.json=base \
              artifacts/cer_custom.json=custom \
              artifacts/cer_graph.json=graph \
      --output artifacts/cer_merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _same_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left == right
    return abs(float(left) - float(right)) <= 1e-6


def _validate_compatible_row(existing: Dict[str, Any], row: Dict[str, Any], *, label: str, file: str) -> None:
    mismatches = {}
    for key in ("start_sec", "duration_sec"):
        if not _same_float(existing.get(key), row.get(key)):
            mismatches[key] = {"existing": existing.get(key), "incoming": row.get(key)}
    incoming_ref_empty = row.get("ref_empty", row.get("ref_chars") == 0)
    for key, incoming in (("ref_chars", row.get("ref_chars")), ("ref_empty", incoming_ref_empty)):
        if existing.get(key) != incoming:
            mismatches[key] = {"existing": existing.get(key), "incoming": incoming}
    if mismatches:
        raise ValueError(
            f"Incompatible CER row for idx={row.get('idx')} step_ms={row.get('step_ms')} "
            f"from label={label} file={file}: {mismatches}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True,
                   help="PATH=LABEL pairs, e.g. artifacts/cer_base.json=base")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    path_map: Dict[str, str] = {}   # label -> file
    for spec in args.input:
        file, label = spec.rsplit("=", 1)
        path_map[label] = file

    per_window: Dict[tuple[int, int | None], Dict[str, Any]] = {}
    labels = list(path_map.keys())

    for label, file in path_map.items():
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        for row in data.get("windows", []):
            idx = int(row["idx"])
            step_ms = int(row["step_ms"]) if row.get("step_ms") is not None else None
            key = (idx, step_ms)
            if key not in per_window:
                merged_row = {
                    "idx": idx,
                    "start_sec": row["start_sec"],
                    "duration_sec": row.get("duration_sec"),
                    "ref_chars": row["ref_chars"],
                    "ref_empty": row.get("ref_empty", row["ref_chars"] == 0),
                    "paths": {},
                }
                if step_ms is not None:
                    merged_row["step_ms"] = step_ms
                per_window[key] = merged_row
            else:
                _validate_compatible_row(per_window[key], row, label=label, file=file)
            # row["paths"] may contain multiple paths (old single-process runs).
            # Pull only the label we care about, but also accept if the file
            # only wrote one path under its own key.
            if label in row.get("paths", {}):
                per_window[key]["paths"][label] = row["paths"][label]
            elif len(row.get("paths", {})) == 1:
                # single-path output with a different key name
                (only_key,) = list(row["paths"].keys())
                per_window[key]["paths"][label] = row["paths"][only_key]

    rows = [per_window[k] for k in sorted(per_window, key=lambda item: (item[0], -1 if item[1] is None else item[1]))]

    non_empty = [r for r in rows if not r.get("ref_empty")]
    # Fill delta_vs_base where possible
    if "base" in labels:
        for r in non_empty:
            base_cer = r["paths"].get("base", {}).get("cer")
            if base_cer is None:
                continue
            for label in labels:
                if label == "base":
                    continue
                entry = r["paths"].get(label)
                if entry is None:
                    continue
                if "cer" in entry and entry["cer"] is not None:
                    entry["delta_vs_base"] = round(entry["cer"] - base_cer, 6)

    summary: Dict[str, Any] = {}
    for label in labels:
        cer_vals = [r["paths"][label]["cer"] for r in non_empty
                    if label in r["paths"] and r["paths"][label].get("cer") is not None]
        wall_vals = [r["paths"][label]["wall_sec"] for r in non_empty
                     if label in r["paths"] and r["paths"][label].get("wall_sec", 0) > 0]
        entry = {
            "n_windows": len({int(r["idx"]) for r in non_empty if label in r["paths"] and r["paths"][label].get("cer") is not None}),
            "n_rows": len(cer_vals),
            "cer_mean": round(mean(cer_vals), 4) if cer_vals else None,
            "cer_p50": round(median(cer_vals), 4) if cer_vals else None,
            "cer_p90": round(_pct(cer_vals, 0.9), 4) if cer_vals else None,
            "cer_max": round(max(cer_vals), 4) if cer_vals else None,
            "wall_mean_sec": round(mean(wall_vals), 3) if wall_vals else None,
            "wall_p50_sec": round(median(wall_vals), 3) if wall_vals else None,
        }
        if label != "base" and "base" in labels:
            deltas = []
            for r in non_empty:
                d = r["paths"].get(label, {}).get("delta_vs_base")
                if d is not None:
                    deltas.append(d)
            if deltas:
                entry["delta_mean"] = round(mean(deltas), 5)
                entry["delta_p50"] = round(median(deltas), 5)
                entry["delta_p90"] = round(_pct(deltas, 0.9), 5)
                entry["delta_min"] = round(min(deltas), 5)
                entry["delta_max"] = round(max(deltas), 5)
                entry["delta_abs_max"] = round(max(abs(x) for x in deltas), 5)
                entry["delta_abs_mean"] = round(mean(abs(x) for x in deltas), 5)
        summary[label] = entry
        print(f"{label}: {entry}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"summary": summary, "windows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
