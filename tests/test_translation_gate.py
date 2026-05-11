# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from tools.gate_translation import (
    _evaluate_quality,
    _extract_format_markers,
    _parse_json_object,
    _parse_args,
    _performance_issues,
    _quality_gate,
    _reference_similarity,
    _summarize,
)


class TranslationGateQualityTest(unittest.TestCase):
    def test_decode_mode_defaults_to_greedy_and_sample_is_explicit(self) -> None:
        with mock.patch("sys.argv", ["gate_translation.py", "--dataset", "cases.jsonl"]):
            args = _parse_args()
        self.assertFalse(args.sample)
        self.assertFalse(args.greedy)

        with mock.patch("sys.argv", ["gate_translation.py", "--dataset", "cases.jsonl", "--sample"]):
            args = _parse_args()
        self.assertTrue(args.sample)
        self.assertFalse(args.greedy)

    def test_parse_json_object_rejects_non_object(self) -> None:
        self.assertEqual(_parse_json_object('{"logits_to_keep": 1}'), {"logits_to_keep": 1})

        with self.assertRaisesRegex(ValueError, "JSON object"):
            _parse_json_object("[]")

    def test_target_language_mismatch_is_error(self) -> None:
        issues = _evaluate_quality(
            {"text": "hello world", "target_language": "Chinese"},
            "hello world",
            generated_tokens=2,
            max_new_tokens=16,
        )

        self.assertIn("target_language_mismatch", {issue.code for issue in issues})

    def test_preserves_required_structural_markers(self) -> None:
        issues = _evaluate_quality(
            {
                "text": "open https://example.com and keep `user_id` in the table\n| a | b |\n| 1 | 2 |",
                "target_language": "Chinese",
            },
            "打开 https://example.com 并保留 `user_id`\n| a | b |\n| 1 | 2 |",
            generated_tokens=12,
            max_new_tokens=32,
        )

        self.assertFalse([issue for issue in issues if issue.severity == "error"])

    def test_missing_required_url_is_error(self) -> None:
        issues = _evaluate_quality(
            {"text": "open https://example.com now", "target_language": "Chinese"},
            "现在打开这个网站",
            generated_tokens=8,
            max_new_tokens=32,
        )

        self.assertIn("missing_urls", {issue.code for issue in issues})

    def test_extract_format_markers_finds_srt_and_html(self) -> None:
        markers = _extract_format_markers('00:00:01,000 <span data-x="1">hello</span> $USER')

        self.assertEqual(markers["srt_timestamps"], ["00:00:01,000"])
        self.assertEqual(markers["html_tags"], ['<span data-x="1">', "</span>"])
        self.assertEqual(markers["placeholders"], ["$USER"])

    def test_malformed_markdown_table_is_error(self) -> None:
        issues = _evaluate_quality(
            {
                "text": "| A | B | C |\n|---|---:|---:|\n| 1 | 2 | 3 |",
                "target_language": "Chinese",
            },
            "| 甲 | 乙 | 丙 |\n|---|---|---:|---|\n| 1 | 2 | 3 |",
            generated_tokens=16,
            max_new_tokens=32,
        )

        self.assertIn("malformed_markdown_table", {issue.code for issue in issues})

    def test_required_output_missing_item_is_error(self) -> None:
        issues = _evaluate_quality(
            {
                "text": "open the dashboard",
                "target_language": "Chinese",
                "required_output_substrings": ["控制台"],
            },
            "打开应用",
            generated_tokens=8,
            max_new_tokens=32,
        )

        self.assertIn("missing_required_output", {issue.code for issue in issues})

    def test_missing_must_preserve_item_is_warning(self) -> None:
        issues = _evaluate_quality(
            {
                "text": 'show "Another realtime session is active"',
                "target_language": "Chinese",
                "must_preserve": ["Another realtime session is active"],
            },
            "显示“还有另一个实时会话处于活动状态”",
            generated_tokens=12,
            max_new_tokens=32,
        )

        self.assertIn("missing_must_preserve", {issue.code for issue in issues})
        self.assertEqual(
            [issue.severity for issue in issues if issue.code == "missing_must_preserve"],
            ["warning"],
        )

    def test_digit_must_preserve_requires_standalone_match(self) -> None:
        issues = _evaluate_quality(
            {
                "text": "1\n00:00:01,000 --> 00:00:03,500\nhello",
                "target_language": "Chinese",
                "must_preserve": ["1"],
            },
            "00:00:01,000 --> 00:00:03,500\n你好",
            generated_tokens=12,
            max_new_tokens=32,
        )

        self.assertIn("missing_must_preserve", {issue.code for issue in issues})

    def test_reference_similarity_tracks_content_overlap(self) -> None:
        strong = _reference_similarity(
            {"reference": "The meeting has started. Please turn on captions."},
            "The meeting has started. Please turn on the subtitles.",
        )
        weak = _reference_similarity(
            {"reference": "The meeting has started. Please turn on captions."},
            "Payment failed. Please contact support.",
        )

        self.assertIsNotNone(strong)
        self.assertIsNotNone(weak)
        self.assertGreater(strong, weak)


class TranslationGatePerformanceTest(unittest.TestCase):
    def test_summarize_reports_wall_time_and_tps(self) -> None:
        rows = [
            {
                "total_wall_sec_median": 0.2,
                "generate_wall_sec_median": 0.18,
                "generated_tokens_median": 18,
                "decode_tokens_per_sec": 100,
                "reference_similarity": 0.85,
            },
            {
                "total_wall_sec_median": 0.4,
                "generate_wall_sec_median": 0.35,
                "generated_tokens_median": 35,
                "decode_tokens_per_sec": 100,
                "reference_similarity": 0.75,
            },
        ]

        summary = _summarize(rows)

        self.assertEqual(summary["total_wall_sec_sum"], 0.6)
        self.assertEqual(summary["generated_tokens_sum"], 53.0)
        self.assertEqual(summary["decode_tokens_per_sec_total"], 100.0)
        self.assertEqual(summary["end_to_end_tokens_per_sec_total"], 88.33)
        self.assertEqual(summary["decode_tokens_per_sec_median"], 100.0)
        self.assertEqual(summary["reference_similarity_median"], 0.8)

    def test_performance_issue_uses_baseline_speedup(self) -> None:
        with TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps({"summary": {"total_wall_sec_sum": 10.0}}),
                encoding="utf-8",
            )
            summary = {
                "total_wall_sec_sum": 8.0,
                "total_wall_sec_median": 0.2,
                "decode_tokens_per_sec_median": 120.0,
            }

            issues = _performance_issues(
                summary,
                baseline_json=baseline_path,
                min_speedup=1.5,
                max_total_wall_sec=None,
                max_median_wall_sec=None,
                min_decode_tokens_per_sec=None,
            )

        self.assertEqual(summary["speedup_vs_baseline"], 1.25)
        self.assertIn("speedup_below_threshold", {issue.code for issue in issues})


class TranslationGateRegressionTest(unittest.TestCase):
    def test_quality_baseline_allows_existing_errors(self) -> None:
        with TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline_gate.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case-1",
                                "errors": [{"code": "missing_code_fences"}],
                                "warnings": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "id": "case-1",
                    "errors": [{"code": "missing_code_fences"}],
                    "warnings": [],
                }
            ]

            result = _quality_gate(rows, quality_baseline_json=baseline_path, fail_on_warnings=False)

        self.assertEqual(result["new_error_count"], 0)
        self.assertFalse(result["issues"])
        self.assertEqual(rows[0]["new_errors"], [])

    def test_quality_baseline_reports_new_errors(self) -> None:
        with TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline_gate.json"
            baseline_path.write_text(
                json.dumps({"cases": [{"id": "case-1", "errors": [], "warnings": []}]}),
                encoding="utf-8",
            )
            rows = [
                {
                    "id": "case-1",
                    "errors": [{"code": "target_language_mismatch"}],
                    "warnings": [],
                }
            ]

            result = _quality_gate(rows, quality_baseline_json=baseline_path, fail_on_warnings=False)

        self.assertEqual(result["new_error_count"], 1)
        self.assertIn("new_quality_errors", {issue.code for issue in result["issues"]})
        self.assertEqual(rows[0]["new_errors"], [{"code": "target_language_mismatch"}])

    def test_quality_baseline_reports_reference_similarity_drop(self) -> None:
        with TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline_gate.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case-1",
                                "errors": [],
                                "warnings": [],
                                "reference_similarity": 0.92,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "id": "case-1",
                    "errors": [],
                    "warnings": [],
                    "reference_similarity": 0.7,
                }
            ]

            result = _quality_gate(rows, quality_baseline_json=baseline_path, fail_on_warnings=False)

        self.assertEqual(result["new_error_count"], 1)
        self.assertEqual(result["reference_similarity_drop_count"], 1)
        self.assertEqual(rows[0]["reference_similarity_drop"], 0.22)
        self.assertIn("reference_similarity_drop", {issue["code"] for issue in rows[0]["new_errors"]})


if __name__ == "__main__":
    unittest.main()
