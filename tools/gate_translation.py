# coding=utf-8
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_MAX_NEW_TOKENS,
    DEFAULT_HYMT_MODEL,
    HYMTGenerationConfig,
    HYMTTranslator,
)


@dataclass(frozen=True)
class TranslationIssue:
    severity: str
    code: str
    message: str


_REFERENCE_SIMILARITY_DROP_ERROR = 0.15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HY-MT translation E2E quality and performance gates.")
    parser.add_argument("--dataset", required=True, help="JSONL case file path.")
    parser.add_argument("--cases", default=None, help="Comma-separated case ids or groups. Default: all.")
    parser.add_argument("--model", default=DEFAULT_HYMT_MODEL, help="HY-MT model path or Hugging Face id.")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cpu, or auto.")
    parser.add_argument("--dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        default=DEFAULT_HYMT_ATTN_IMPLEMENTATION,
        help="Transformers attention implementation. Use 'none' to let transformers choose.",
    )
    parser.add_argument("--decode-backend", default=DEFAULT_HYMT_DECODE_BACKEND, choices=["fixed_mask", "generate"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_HYMT_MAX_NEW_TOKENS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for deterministic decode.")
    parser.add_argument(
        "--extra-generate-kwargs",
        default=None,
        help="Optional JSON object. Omit to use runtime defaults.",
    )
    parser.add_argument("--warmup", default="1", help="Warmup count, or 'all' to warm up every selected case.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each case and gate the last output.")
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed for repeatable sampling gates.")
    parser.add_argument(
        "--seed-mode",
        choices=["case", "run"],
        default="case",
        help="Use per-case seeds by default so output changes do not shift later case RNG.",
    )
    parser.add_argument("--baseline-json", default=None, help="Optional benchmark/gate JSON to compare speed against.")
    parser.add_argument(
        "--quality-baseline-json",
        default=None,
        help="Optional gate JSON used to fail only on new quality issues.",
    )
    parser.add_argument("--min-speedup", type=float, default=None, help="Minimum total-wall speedup vs baseline JSON.")
    parser.add_argument("--max-total-wall-sec", type=float, default=None)
    parser.add_argument("--max-median-wall-sec", type=float, default=None)
    parser.add_argument("--min-decode-tokens-per-sec", type=float, default=None)
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download model files during startup. Default is local-only.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional path to write gate JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _select_cases(_load_cases(Path(args.dataset)), args.cases)
    if not cases:
        raise ValueError("No translation gate cases selected")

    torch.manual_seed(int(args.seed))
    config_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "do_sample": not args.greedy,
    }
    if args.extra_generate_kwargs is not None:
        config_kwargs["extra_generate_kwargs"] = _parse_json_object(args.extra_generate_kwargs)
    generation_config = HYMTGenerationConfig(**config_kwargs)
    load_started = time.perf_counter()
    translator = HYMTTranslator(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        decode_backend=args.decode_backend,
        generation_config=generation_config,
    )
    load_wall_sec = time.perf_counter() - load_started

    warmup_count = _resolve_warmup_count(args.warmup, len(cases))
    warmup_started = time.perf_counter()
    for index in range(warmup_count):
        case = cases[index % len(cases)]
        translator.profile_translate(
            str(case["text"]),
            target_language=str(case["target_language"]),
            source_language=str(case.get("source_language") or ""),
            max_new_tokens=args.max_new_tokens,
            sync_cuda=True,
        )
    warmup_wall_sec = time.perf_counter() - warmup_started
    # Warmup consumes sampling RNG; measured outputs should not depend on warmup count.
    torch.manual_seed(int(args.seed))

    if torch.cuda.is_available() and ("cuda" in str(args.device) or args.device == "auto"):
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for index, case in enumerate(cases):
        seed = int(args.seed) + index if args.seed_mode == "case" else None
        rows.append(
            _run_case(
                translator,
                case,
                repeats=max(1, int(args.repeats)),
                max_new_tokens=args.max_new_tokens,
                seed=seed,
            )
        )
    summary = _summarize(rows)
    gate_issues = _performance_issues(
        summary,
        baseline_json=Path(args.baseline_json) if args.baseline_json else None,
        min_speedup=args.min_speedup,
        max_total_wall_sec=args.max_total_wall_sec,
        max_median_wall_sec=args.max_median_wall_sec,
        min_decode_tokens_per_sec=args.min_decode_tokens_per_sec,
    )
    case_error_count = sum(1 for row in rows if row["errors"])
    case_warning_count = sum(1 for row in rows if row["warnings"])
    quality_gate = _quality_gate(
        rows,
        quality_baseline_json=Path(args.quality_baseline_json) if args.quality_baseline_json else None,
        fail_on_warnings=args.fail_on_warnings,
    )
    gate_issues.extend(quality_gate["issues"])
    passed = not gate_issues
    payload = {
        "passed": passed,
        "case_count": len(rows),
        "device": args.device,
        "dtype": args.dtype or ("bfloat16" if args.device.startswith("cuda") or args.device == "auto" else "auto"),
        "attn_implementation": translator.attn_implementation,
        "decode_backend": translator.decode_backend,
        "max_new_tokens": args.max_new_tokens,
        "generation": {
            "do_sample": not args.greedy,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "extra_generate_kwargs": generation_config.extra_generate_kwargs,
            "seed": args.seed,
            "seed_mode": args.seed_mode,
        },
        "load_wall_sec": round(load_wall_sec, 3),
        "warmup_count": warmup_count,
        "warmup_wall_sec": round(warmup_wall_sec, 3),
        "summary": summary,
        "gate_issues": [issue.__dict__ for issue in gate_issues],
        "case_error_count": case_error_count,
        "case_warning_count": case_warning_count,
        "quality_gate": {
            key: value
            for key, value in quality_gate.items()
            if key != "issues"
        },
        "cuda_peak_allocated_mb": _cuda_peak_allocated_mb(args.device),
        "cases": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not passed:
        raise SystemExit(1)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: case must be a JSON object")
        for field in ("id", "target_language", "text"):
            if not str(item.get(field) or "").strip():
                raise ValueError(f"{path}:{line_number}: missing required field {field!r}")
        case_id = str(item["id"])
        if case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case id {case_id!r}")
        seen.add(case_id)
        cases.append(item)
    return cases


def _select_cases(cases: list[dict[str, Any]], spec: str | None) -> list[dict[str, Any]]:
    if not spec:
        return cases
    wanted = {item.strip() for item in spec.split(",") if item.strip()}
    return [
        case
        for case in cases
        if str(case.get("id") or "") in wanted or str(case.get("group") or "") in wanted
    ]


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _resolve_warmup_count(value: str, case_count: int) -> int:
    text = str(value).strip().lower()
    if text == "all":
        return case_count
    return max(0, int(text))


def _run_case(
    translator: HYMTTranslator,
    case: dict[str, Any],
    *,
    repeats: int,
    max_new_tokens: int,
    seed: int | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    output = ""
    for _ in range(repeats):
        if seed is not None:
            torch.manual_seed(int(seed))
        result = translator.profile_translate(
            str(case["text"]),
            target_language=str(case["target_language"]),
            source_language=str(case.get("source_language") or ""),
            max_new_tokens=max_new_tokens,
            sync_cuda=True,
        )
        output = result.text
        samples.append(
            {
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "encode_wall_sec": result.encode_wall_sec,
                "generate_wall_sec": result.generate_wall_sec,
                "decode_wall_sec": result.decode_wall_sec,
                "total_wall_sec": result.total_wall_sec,
            }
        )
    issues = _evaluate_quality(case, output, generated_tokens=int(samples[-1]["generated_tokens"]), max_new_tokens=max_new_tokens)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    reference_similarity = _reference_similarity(case, output)
    total_values = [float(item["total_wall_sec"]) for item in samples]
    generate_values = [float(item["generate_wall_sec"]) for item in samples]
    generated_tokens = [int(item["generated_tokens"]) for item in samples]
    median_total = median(total_values)
    median_generate = median(generate_values)
    median_generated_tokens = median(generated_tokens)
    return {
        "id": case["id"],
        "group": case.get("group", ""),
        "target_language": case["target_language"],
        "source_language": case.get("source_language", ""),
        "source_chars": len(str(case["text"])),
        "output_chars": len(output),
        "prompt_tokens": int(samples[-1]["prompt_tokens"]),
        "generated_tokens_median": median_generated_tokens,
        "total_wall_sec_median": round(median_total, 4),
        "generate_wall_sec_median": round(median_generate, 4),
        "decode_tokens_per_sec": round(median_generated_tokens / median_generate, 2) if median_generate > 0 else 0.0,
        "errors": [issue.__dict__ for issue in errors],
        "warnings": [issue.__dict__ for issue in warnings],
        **({"reference_similarity": reference_similarity} if reference_similarity is not None else {}),
    }


def _evaluate_quality(
    case: dict[str, Any],
    output: str,
    *,
    generated_tokens: int,
    max_new_tokens: int,
) -> list[TranslationIssue]:
    source = str(case["text"])
    target_language = str(case["target_language"])
    issues: list[TranslationIssue] = []
    if not output.strip():
        issues.append(TranslationIssue("error", "empty_output", "translation output is empty"))
        return issues
    if generated_tokens >= max_new_tokens:
        issues.append(TranslationIssue("error", "max_new_tokens_hit", "generation reached max_new_tokens"))
    issues.extend(_target_language_issues(output, target_language))
    issues.extend(_length_issues(source, output))
    issues.extend(_format_issues(source, output))
    issues.extend(_must_preserve_issues(case, output))
    issues.extend(_required_output_issues(case, output))
    if _has_repetition_loop(output):
        issues.append(TranslationIssue("warning", "repetition_loop", "output contains repeated phrase pattern"))
    return issues


def _target_language_issues(output: str, target_language: str) -> list[TranslationIssue]:
    language = target_language.strip().lower()
    text = output.strip()
    if language in {"english", "en"}:
        if _latin_count(text) < max(3, int(len(text) * 0.12)):
            return [TranslationIssue("error", "target_language_mismatch", "expected English-like output")]
    if language in {"chinese", "zh", "中文"}:
        if _han_count(text) < 2:
            return [TranslationIssue("error", "target_language_mismatch", "expected Chinese-like output")]
    if language in {"japanese", "ja", "日本語"}:
        if _kana_count(text) < 1:
            return [TranslationIssue("error", "target_language_mismatch", "expected Japanese-like output")]
    return []


def _length_issues(source: str, output: str) -> list[TranslationIssue]:
    if len(source.strip()) < 12:
        return []
    ratio = len(output.strip()) / max(1, len(source.strip()))
    if ratio < 0.08:
        return [TranslationIssue("error", "too_short", "output is too short relative to source")]
    if ratio > 8.0:
        return [TranslationIssue("error", "too_long", "output is too long relative to source")]
    if ratio < 0.2 or ratio > 5.0:
        return [TranslationIssue("warning", "suspicious_length_ratio", "output/source length ratio is unusual")]
    return []


def _format_issues(source: str, output: str) -> list[TranslationIssue]:
    issues: list[TranslationIssue] = []
    source_markers = _extract_format_markers(source)
    output_markers = _extract_format_markers(output)
    for key in ("urls", "srt_timestamps", "html_tags", "code_fences"):
        missing = _missing_required(source_markers[key], output_markers[key])
        if missing:
            issues.append(TranslationIssue("error", f"missing_{key}", f"missing {len(missing)} structural marker(s)"))
    for key in ("inline_code", "placeholders"):
        missing = _missing_required(source_markers[key], output_markers[key])
        if missing:
            issues.append(TranslationIssue("warning", f"missing_{key}", f"missing {len(missing)} inline marker(s)"))
    source_table_columns = source_markers["table_columns"]
    output_table_columns = output_markers["table_columns"]
    if len(source_table_columns) >= 2:
        if len(output_table_columns) < 2:
            issues.append(TranslationIssue("error", "missing_markdown_table", "markdown table structure was not preserved"))
        elif len(set(source_table_columns)) == 1 and any(count != source_table_columns[0] for count in output_table_columns):
            issues.append(TranslationIssue("error", "malformed_markdown_table", "markdown table column count changed"))
    return issues


def _extract_format_markers(text: str) -> dict[str, Any]:
    table_columns = _markdown_table_columns(text)
    return {
        "urls": re.findall(r"https?://[^\s)>\]]+", text),
        "srt_timestamps": re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", text),
        "html_tags": [item.lower() for item in re.findall(r"</?[a-zA-Z][^>]*>", text)],
        "code_fences": re.findall(r"```", text),
        "inline_code": re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text),
        "placeholders": re.findall(r"(\{[^{}\n]+\}|%[sdif]|\$[A-Za-z_][A-Za-z0-9_]*)", text),
        "table_pipe_lines": len(table_columns),
        "table_columns": table_columns,
    }


def _markdown_table_columns(text: str) -> list[int]:
    columns: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            columns.append(len(stripped.strip("|").split("|")))
    return columns


def _required_output_issues(case: dict[str, Any], output: str) -> list[TranslationIssue]:
    missing = [
        str(item)
        for item in case.get("required_output_substrings", [])
        if str(item) and str(item) not in output
    ]
    if not missing:
        return []
    return [TranslationIssue("error", "missing_required_output", f"missing {len(missing)} required output item(s)")]


def _must_preserve_issues(case: dict[str, Any], output: str) -> list[TranslationIssue]:
    missing = [
        str(item)
        for item in case.get("must_preserve", [])
        if str(item) and not _contains_preserved_item(output, str(item))
    ]
    if not missing:
        return []
    return [TranslationIssue("warning", "missing_must_preserve", f"missing {len(missing)} preserved item(s)")]


def _contains_preserved_item(output: str, item: str) -> bool:
    if item.isdigit():
        return re.search(rf"(?<![\d.]){re.escape(item)}(?![\d.])", output) is not None
    if item in output:
        return True
    if item.isascii():
        return item.lower() in output.lower()
    return False


def _missing_required(source_items: list[str], output_items: list[str]) -> list[str]:
    missing: list[str] = []
    remaining = list(output_items)
    for item in source_items:
        if item in remaining:
            remaining.remove(item)
        else:
            missing.append(item)
    return missing


def _has_repetition_loop(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return False
    words = compact.split(" ")
    for width in (1, 2, 3, 4):
        if len(words) < width * 4:
            continue
        for index in range(0, len(words) - width * 4 + 1):
            chunk = words[index : index + width]
            if all(words[index + width * n : index + width * (n + 1)] == chunk for n in range(1, 4)):
                return True
    return False


def _reference_similarity(case: dict[str, Any], output: str) -> float | None:
    reference = str(case.get("reference") or "").strip()
    if not reference:
        return None
    return _char_ngram_f1(reference, output)


def _char_ngram_f1(reference: str, output: str) -> float:
    ref = _normalize_reference_text(reference)
    hyp = _normalize_reference_text(output)
    if not ref or not hyp:
        return 0.0
    width = 1 if min(len(ref), len(hyp)) < 4 else 2
    ref_counts = Counter(ref[index : index + width] for index in range(len(ref) - width + 1))
    hyp_counts = Counter(hyp[index : index + width] for index in range(len(hyp) - width + 1))
    overlap = sum((ref_counts & hyp_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())
    return round(2 * precision * recall / (precision + recall), 4)


def _normalize_reference_text(text: str) -> str:
    return re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff]+", "", text.lower())


def _han_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _kana_count(text: str) -> int:
    return sum(1 for char in text if "\u3040" <= char <= "\u30ff")


def _latin_count(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [float(row["total_wall_sec_median"]) for row in rows]
    generates = [float(row["generate_wall_sec_median"]) for row in rows]
    generated_tokens = [float(row["generated_tokens_median"]) for row in rows]
    decode_tps = [float(row["decode_tokens_per_sec"]) for row in rows]
    generated_tokens_sum = sum(generated_tokens)
    generate_wall_sec_sum = sum(generates)
    total_wall_sec_sum = sum(totals)
    summary: dict[str, Any] = {
        "total_wall_sec_sum": round(total_wall_sec_sum, 4),
        "total_wall_sec_median": round(median(totals), 4),
        "generate_wall_sec_sum": round(generate_wall_sec_sum, 4),
        "generated_tokens_sum": round(generated_tokens_sum, 1),
        "decode_tokens_per_sec_total": _tokens_per_sec(generated_tokens_sum, generate_wall_sec_sum),
        "end_to_end_tokens_per_sec_total": _tokens_per_sec(generated_tokens_sum, total_wall_sec_sum),
        "decode_tokens_per_sec_median": round(median(decode_tps), 2),
        "generated_tokens_median": round(median(generated_tokens), 1),
    }
    similarities = [float(row["reference_similarity"]) for row in rows if row.get("reference_similarity") is not None]
    if similarities:
        summary["reference_similarity_median"] = round(median(similarities), 4)
        summary["reference_similarity_min"] = round(min(similarities), 4)
    return summary


def _tokens_per_sec(tokens: float, wall_sec: float) -> float:
    return round(tokens / wall_sec, 2) if wall_sec > 0 else 0.0


def _performance_issues(
    summary: dict[str, Any],
    *,
    baseline_json: Path | None,
    min_speedup: float | None,
    max_total_wall_sec: float | None,
    max_median_wall_sec: float | None,
    min_decode_tokens_per_sec: float | None,
) -> list[TranslationIssue]:
    issues: list[TranslationIssue] = []
    if baseline_json is not None and min_speedup is not None:
        baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
        baseline_total = float(baseline["summary"]["total_wall_sec_sum"])
        speedup = baseline_total / float(summary["total_wall_sec_sum"])
        summary["speedup_vs_baseline"] = round(speedup, 4)
        if speedup < min_speedup:
            issues.append(TranslationIssue("error", "speedup_below_threshold", "speedup vs baseline is too low"))
    if max_total_wall_sec is not None and float(summary["total_wall_sec_sum"]) > max_total_wall_sec:
        issues.append(TranslationIssue("error", "total_wall_sec_above_threshold", "total wall time is too high"))
    if max_median_wall_sec is not None and float(summary["total_wall_sec_median"]) > max_median_wall_sec:
        issues.append(TranslationIssue("error", "median_wall_sec_above_threshold", "median wall time is too high"))
    if min_decode_tokens_per_sec is not None and float(summary["decode_tokens_per_sec_median"]) < min_decode_tokens_per_sec:
        issues.append(TranslationIssue("error", "tokens_per_sec_below_threshold", "decode tokens/sec is too low"))
    return issues


def _quality_gate(
    rows: list[dict[str, Any]],
    *,
    quality_baseline_json: Path | None,
    fail_on_warnings: bool,
) -> dict[str, Any]:
    if quality_baseline_json is None:
        case_error_count = sum(1 for row in rows if row["errors"])
        case_warning_count = sum(1 for row in rows if row["warnings"])
        issues: list[TranslationIssue] = []
        if case_error_count:
            issues.append(TranslationIssue("error", "quality_errors", "translation quality errors found"))
        if fail_on_warnings and case_warning_count:
            issues.append(TranslationIssue("error", "quality_warnings", "translation quality warnings found"))
        return {
            "mode": "absolute",
            "baseline_json": None,
            "new_error_count": case_error_count,
            "new_warning_count": case_warning_count,
            "reference_similarity_drop_count": 0,
            "baseline_missing_case_count": 0,
            "issues": issues,
        }

    baseline_by_id = _load_quality_baseline(quality_baseline_json)
    new_error_count = 0
    new_warning_count = 0
    reference_similarity_drop_count = 0
    missing_case_count = 0
    for row in rows:
        baseline = baseline_by_id.get(str(row["id"]))
        if baseline is None:
            missing_case_count += 1
            row["new_errors"] = row["errors"]
            row["new_warnings"] = row["warnings"]
        else:
            row["new_errors"] = [
                issue
                for issue in row["errors"]
                if str(issue.get("code") or "") not in baseline["errors"]
            ]
            row["new_warnings"] = [
                issue
                for issue in row["warnings"]
                if str(issue.get("code") or "") not in baseline["warnings"]
            ]
            reference_drop = _reference_similarity_drop(row, baseline.get("reference_similarity"))
            if reference_drop is not None:
                row["reference_similarity_drop"] = reference_drop
                if reference_drop > _REFERENCE_SIMILARITY_DROP_ERROR:
                    reference_similarity_drop_count += 1
                    row["new_errors"].append(
                        {
                            "severity": "error",
                            "code": "reference_similarity_drop",
                            "message": "reference similarity dropped from quality baseline",
                        }
                    )
        new_error_count += len(row["new_errors"])
        new_warning_count += len(row["new_warnings"])

    issues = []
    if missing_case_count:
        issues.append(TranslationIssue("error", "quality_baseline_missing_cases", "quality baseline is missing cases"))
    if new_error_count:
        issues.append(TranslationIssue("error", "new_quality_errors", "new translation quality errors found"))
    if fail_on_warnings and new_warning_count:
        issues.append(TranslationIssue("error", "new_quality_warnings", "new translation quality warnings found"))
    return {
        "mode": "baseline_regression",
        "baseline_json": str(quality_baseline_json),
        "new_error_count": new_error_count,
        "new_warning_count": new_warning_count,
        "reference_similarity_drop_count": reference_similarity_drop_count,
        "baseline_missing_case_count": missing_case_count,
        "issues": issues,
    }


def _reference_similarity_drop(row: dict[str, Any], baseline_similarity: float | None) -> float | None:
    current = row.get("reference_similarity")
    if current is None or baseline_similarity is None:
        return None
    return round(float(baseline_similarity) - float(current), 4)


def _load_quality_baseline(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline: dict[str, dict[str, Any]] = {}
    for row in payload.get("cases", []):
        case_id = str(row.get("id") or "")
        if not case_id:
            continue
        baseline[case_id] = {
            "errors": {str(issue.get("code") or "") for issue in row.get("errors", [])},
            "warnings": {str(issue.get("code") or "") for issue in row.get("warnings", [])},
            "reference_similarity": _optional_float(row.get("reference_similarity")),
        }
    return baseline


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _cuda_peak_allocated_mb(device: str) -> float | None:
    if not torch.cuda.is_available() or ("cuda" not in str(device) and device != "auto"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


if __name__ == "__main__":
    main()
