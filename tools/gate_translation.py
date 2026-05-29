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


_REFERENCE_METRIC = "chrf2"
# Per-case chrF drop (0-100) used only to count/rank notably-changed cases for
# human review. A single-reference chrF swings widely on short sentences and
# conflates valid rephrasing with real loss, so a per-case drop never fails the
# gate on its own; the trustworthy signal is the per-direction mean drop.
_CASE_CHRF_DROP_FLAG = 10.0


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
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument("--sample", action="store_true", help="Enable sampling. Default is deterministic greedy decode.")
    decode_group.add_argument("--greedy", action="store_true", help="Use deterministic greedy decode. This is the default.")
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
    parser.add_argument(
        "--w8a16",
        action="store_true",
        help="Weight-only int8 on gate/up linears (decode speedup, CER-gated). Default off.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write gate JSON.")
    parser.add_argument(
        "--max-mean-chrf-drop",
        type=float,
        default=None,
        help="With --quality-baseline-json, fail if any single direction's mean chrF drop exceeds this "
        "(points, 0-100). Gating per direction (not a pooled mean) keeps a regression confined to one "
        "direction from being diluted. Per-case drops are flagged but never fail the gate. Use only on the "
        "large eval set (>=~200 cases/direction); on the 42-case set the per-direction mean is too noisy.",
    )
    parser.add_argument(
        "--worst-output",
        default=None,
        help="With --quality-baseline-json, write the worst chrF-drop changed cases here (source/reference/"
        "baseline/candidate side by side) for human review.",
    )
    parser.add_argument("--worst-n", type=int, default=15, help="How many worst cases to write to --worst-output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _select_cases(_load_cases(Path(args.dataset)), args.cases)
    if not cases:
        raise ValueError("No translation gate cases selected")

    do_sample = bool(args.sample)
    torch.manual_seed(int(args.seed))
    config_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "do_sample": do_sample,
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
        w8a16=args.w8a16,
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
        max_mean_chrf_drop=args.max_mean_chrf_drop,
    )
    gate_issues.extend(quality_gate["issues"])
    if args.worst_output and args.quality_baseline_json:
        _write_worst(
            rows,
            cases,
            Path(args.quality_baseline_json),
            Path(args.worst_output),
            max(1, int(args.worst_n)),
        )
    resolved_dtype = args.dtype or ("bfloat16" if args.device.startswith("cuda") or args.device == "auto" else "auto")
    generation_block = {
        "do_sample": do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "extra_generate_kwargs": generation_config.extra_generate_kwargs,
    }
    run_config_diff: dict[str, Any] = {}
    if args.quality_baseline_json:
        run_config_diff = _run_config_diff(
            Path(args.quality_baseline_json),
            {
                "dtype": resolved_dtype,
                "attn_implementation": translator.attn_implementation,
                "decode_backend": translator.decode_backend,
                "max_new_tokens": args.max_new_tokens,
                "generation": generation_block,
            },
        )
        if run_config_diff:
            # Non-failing: when the change under test IS the backend/dtype, the
            # operator wants this comparison; we only flag that the delta may
            # reflect run config, not only the model change.
            gate_issues.append(
                TranslationIssue(
                    "warning",
                    "run_config_differs_from_baseline",
                    "baseline and candidate run config differ; the paired chrF delta may reflect config, not only the model",
                )
            )
    passed = not gate_issues
    payload = {
        "passed": passed,
        "case_count": len(rows),
        "device": args.device,
        "dtype": resolved_dtype,
        "attn_implementation": translator.attn_implementation,
        "decode_backend": translator.decode_backend,
        "max_new_tokens": args.max_new_tokens,
        "reference_metric": _REFERENCE_METRIC,
        "run_config_diff": run_config_diff,
        "generation": {
            **generation_block,
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
        "output": output,
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
    return _chrf(reference, output)


def _chrf(reference: str, output: str, *, beta: float = 2.0, max_order: int = 6) -> float:
    """Character n-gram F-score (chrF2), 0-100.

    Language-agnostic and well suited to Chinese/Japanese, where word
    tokenization is ambiguous and word-BLEU is unreliable. Whitespace is stripped
    before forming character n-grams (so injected spaces do not inflate
    Latin-script targets), and the per-order F-scores are averaged. An order with
    no matching n-grams contributes 0 rather than being skipped, so a sentence is
    not flattered for matching only short orders.

    This is a self-contained variant and is NOT byte-identical to sacreBLEU's
    chrF2, so the absolute value should not be cross-referenced against published
    chrF. The gate only ever uses the PAIRED delta between baseline and candidate
    -- both scored by this same function -- where the convention cancels exactly.
    A single reference also means a per-sentence score cannot separate valid
    rephrasing from real loss; read it in aggregate (mean drop), never per case.
    """
    ref = re.sub(r"\s+", "", reference.lower())
    hyp = re.sub(r"\s+", "", output.lower())
    if not ref or not hyp:
        return 0.0
    beta_sq = beta * beta
    f_scores: list[float] = []
    for order in range(1, max_order + 1):
        ref_counts = Counter(ref[index : index + order] for index in range(len(ref) - order + 1))
        hyp_counts = Counter(hyp[index : index + order] for index in range(len(hyp) - order + 1))
        if not ref_counts or not hyp_counts:
            continue
        overlap = sum((ref_counts & hyp_counts).values())
        if overlap == 0:
            f_scores.append(0.0)
            continue
        precision = overlap / sum(hyp_counts.values())
        recall = overlap / sum(ref_counts.values())
        f_scores.append((1 + beta_sq) * precision * recall / (beta_sq * precision + recall))
    if not f_scores:
        return 0.0
    return round(100.0 * sum(f_scores) / len(f_scores), 2)


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
    # These hold chrF2 (0-100), not the old 0-1 similarity; named accordingly.
    chrf_values = [float(row["reference_similarity"]) for row in rows if row.get("reference_similarity") is not None]
    if chrf_values:
        summary["chrf_median"] = round(median(chrf_values), 4)
        summary["chrf_min"] = round(min(chrf_values), 4)
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
    max_mean_chrf_drop: float | None = None,
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
            "reference_metric": _REFERENCE_METRIC,
            "new_error_count": case_error_count,
            "new_warning_count": case_warning_count,
            "case_chrf_drop_flag_count": 0,
            "mean_chrf_drop": None,
            "baseline_missing_case_count": 0,
            "issues": issues,
        }

    baseline_by_id, baseline_metric = _load_quality_baseline(quality_baseline_json)
    metric_comparable = baseline_metric == _REFERENCE_METRIC
    new_error_count = 0
    new_warning_count = 0
    case_chrf_drop_flag_count = 0
    missing_case_count = 0
    chrf_drops: list[float] = []
    drops_by_direction: dict[str, list[float]] = {}
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
            # Per-case chrF drop is recorded and flagged for human review, but is
            # deliberately NOT promoted to an error: it is too noisy on single
            # sentences. Systematic loss is judged per direction below.
            reference_drop = (
                _reference_similarity_drop(row, baseline.get("reference_similarity")) if metric_comparable else None
            )
            if reference_drop is not None:
                row["reference_similarity_drop"] = reference_drop
                chrf_drops.append(reference_drop)
                direction = f"{row.get('source_language', '')}->{row.get('target_language', '')}"
                drops_by_direction.setdefault(direction, []).append(reference_drop)
                if reference_drop > _CASE_CHRF_DROP_FLAG:
                    case_chrf_drop_flag_count += 1
        new_error_count += len(row["new_errors"])
        new_warning_count += len(row["new_warnings"])

    mean_chrf_drop = round(sum(chrf_drops) / len(chrf_drops), 4) if chrf_drops else None
    mean_chrf_drop_by_direction = {
        direction: round(sum(values) / len(values), 4) for direction, values in sorted(drops_by_direction.items())
    }
    worst_direction, worst_direction_drop = (None, None)
    if mean_chrf_drop_by_direction:
        worst_direction, worst_direction_drop = max(mean_chrf_drop_by_direction.items(), key=lambda item: item[1])

    issues = []
    if missing_case_count:
        issues.append(TranslationIssue("error", "quality_baseline_missing_cases", "quality baseline is missing cases"))
    if not metric_comparable:
        issues.append(
            TranslationIssue(
                "error",
                "reference_metric_mismatch",
                f"baseline reference metric {baseline_metric!r} != {_REFERENCE_METRIC!r}; regenerate the baseline",
            )
        )
    if new_error_count:
        issues.append(TranslationIssue("error", "new_quality_errors", "new translation quality errors found"))
    if fail_on_warnings and new_warning_count:
        issues.append(TranslationIssue("error", "new_quality_warnings", "new translation quality warnings found"))
    if max_mean_chrf_drop is not None:
        if metric_comparable and not chrf_drops:
            # The aggregate gate is the only chrF teeth; never let it pass vacuously.
            issues.append(
                TranslationIssue(
                    "error", "no_comparable_chrf_cases", "--max-mean-chrf-drop set but no case had a comparable chrF score"
                )
            )
        elif worst_direction_drop is not None and worst_direction_drop > max_mean_chrf_drop:
            # Gate each direction independently: a regression confined to one
            # direction is hidden by a pooled mean, so the worst direction is the
            # sensitive signal.
            issues.append(
                TranslationIssue(
                    "error",
                    "mean_chrf_drop_above_threshold",
                    f"chrF dropped from baseline in {worst_direction} (mean drop {worst_direction_drop})",
                )
            )
    return {
        "mode": "baseline_regression",
        "baseline_json": str(quality_baseline_json),
        "reference_metric": _REFERENCE_METRIC,
        "baseline_reference_metric": baseline_metric,
        "metric_comparable": metric_comparable,
        "new_error_count": new_error_count,
        "new_warning_count": new_warning_count,
        "case_chrf_drop_flag_count": case_chrf_drop_flag_count,
        "compared_case_count": len(chrf_drops),
        "mean_chrf_drop": mean_chrf_drop,
        "mean_chrf_drop_by_direction": mean_chrf_drop_by_direction,
        "worst_direction": worst_direction,
        "worst_direction_drop": worst_direction_drop,
        "max_mean_chrf_drop": max_mean_chrf_drop,
        "baseline_missing_case_count": missing_case_count,
        "issues": issues,
    }


def _reference_similarity_drop(row: dict[str, Any], baseline_similarity: float | None) -> float | None:
    current = row.get("reference_similarity")
    if current is None or baseline_similarity is None:
        return None
    return round(float(baseline_similarity) - float(current), 4)


def _load_quality_baseline(path: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Top-level tag added by this tool; older baselines (char-bigram F1) lack it
    # and are reported as not comparable so chrF drops are not computed wrongly.
    metric = payload.get("reference_metric")
    baseline: dict[str, dict[str, Any]] = {}
    for row in payload.get("cases", []):
        case_id = str(row.get("id") or "")
        if not case_id:
            continue
        baseline[case_id] = {
            "errors": {str(issue.get("code") or "") for issue in row.get("errors", [])},
            "warnings": {str(issue.get("code") or "") for issue in row.get("warnings", [])},
            "reference_similarity": _optional_float(row.get("reference_similarity")),
            "output": str(row.get("output") or ""),
        }
    return baseline, metric


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_worst(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    baseline_path: Path,
    out_path: Path,
    count: int,
) -> None:
    """Dump the largest chrF-drop cases with baseline/candidate text side by side.

    The aggregate gate answers "did quality regress on average?"; this answers
    "show me the cases that moved most so a human can judge them" -- the step the
    metric alone cannot do. Baseline output text comes from the baseline JSON
    (present only if it was produced by this tool's --output-json).
    """
    baseline_by_id, _ = _load_quality_baseline(baseline_path)
    source_by_id = {str(case["id"]): case for case in cases}
    candidates = [row for row in rows if row.get("reference_similarity_drop") is not None]
    candidates.sort(key=lambda row: float(row["reference_similarity_drop"]), reverse=True)
    worst = []
    for row in candidates[:count]:
        case_id = str(row["id"])
        case = source_by_id.get(case_id, {})
        baseline = baseline_by_id.get(case_id, {})
        worst.append(
            {
                "id": case_id,
                "direction": f"{row.get('source_language', '')}->{row.get('target_language', '')}",
                "chrf_drop": row["reference_similarity_drop"],
                "chrf_baseline": round(float(baseline.get("reference_similarity") or 0.0), 2),
                "chrf_candidate": row.get("reference_similarity"),
                "source": str(case.get("text") or ""),
                "reference": str(case.get("reference") or ""),
                "baseline_output": baseline.get("output", ""),
                "candidate_output": row.get("output", ""),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(worst, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_config_diff(baseline_path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """Report run-config keys that differ between candidate and the baseline JSON.

    Surfaced as a non-failing warning: when the change under test IS the decode
    backend / dtype / generation settings, the operator wants the comparison, so
    this must not block. It only flags that a paired delta may reflect config
    rather than only the model change. Only keys present in ``candidate`` are
    compared, so extra baseline keys (e.g. seed) do not trigger spurious diffs.
    """
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    diff: dict[str, Any] = {}
    for key, value in candidate.items():
        if key not in baseline:
            continue
        baseline_value = baseline.get(key)
        if key == "generation" and isinstance(value, dict) and isinstance(baseline_value, dict):
            sub = {
                sub_key: {"baseline": baseline_value.get(sub_key), "candidate": sub_value}
                for sub_key, sub_value in value.items()
                if baseline_value.get(sub_key) != sub_value
            }
            if sub:
                diff[key] = sub
        elif baseline_value != value:
            diff[key] = {"baseline": baseline_value, "candidate": value}
    return diff


def _cuda_peak_allocated_mb(device: str) -> float | None:
    if not torch.cuda.is_available() or ("cuda" not in str(device) and device != "auto"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


if __name__ == "__main__":
    main()
