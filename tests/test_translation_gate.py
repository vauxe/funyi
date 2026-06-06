# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qwen3_asr_runtime.translation import DEFAULT_HYMT_FUSED_RMSNORM, DEFAULT_HYMT_W8A16
import tools.gate_translation as gate_module
from tools.gate_translation import (
    _evaluate_quality,
    _extract_format_markers,
    _optimization_patch_issues,
    _parse_json_object,
    _parse_args,
    _performance_issues,
    _quality_gate,
    _reference_similarity,
    _run_config_diff,
    _summarize,
)


class TestTranslationGateQuality:
    @pytest.mark.parametrize(
        ("extra_args", "expected_sample"),
        [
            ([], False),
            (["--sample"], True),
        ],
    )
    def test_decode_mode_defaults_to_greedy_and_sample_is_explicit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extra_args: list[str],
        expected_sample: bool,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["gate_translation.py", "--dataset", "cases.jsonl", *extra_args],
        )

        args = _parse_args()

        assert args.sample is expected_sample
        assert not args.greedy

    def test_runtime_optimization_flags_default_to_translation_profile(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["gate_translation.py", "--dataset", "cases.jsonl"]
        )

        args = _parse_args()

        assert args.w8a16 is DEFAULT_HYMT_W8A16
        assert args.fused_rmsnorm is DEFAULT_HYMT_FUSED_RMSNORM

    def test_runtime_optimization_flags_can_be_disabled_for_ablations(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gate_translation.py",
                "--dataset",
                "cases.jsonl",
                "--no-w8a16",
                "--no-fused-rmsnorm",
            ],
        )

        args = _parse_args()

        assert not args.w8a16
        assert not args.fused_rmsnorm

    def test_main_forwards_and_records_runtime_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text(
            json.dumps({"id": "case-1", "text": "hello", "target_language": "Chinese"})
            + "\n",
            encoding="utf-8",
        )
        output = tmp_path / "gate.json"
        constructed: dict[str, object] = {}

        class FakeTranslator:
            attn_implementation = "sdpa"
            decode_backend = "generate"
            w8a16 = False
            fused_rmsnorm = False
            w8a16_patch_count = 0
            fused_rmsnorm_patch_count = 0
            resolved_model_commit = "abc123"

            def __init__(self, model: str, **kwargs: object) -> None:
                constructed["model"] = model
                constructed.update(kwargs)

            def profile_translate(
                self, *args: object, **kwargs: object
            ) -> SimpleNamespace:
                del args, kwargs
                return SimpleNamespace(text="你好")

        monkeypatch.setattr(gate_module, "HYMTTranslator", FakeTranslator)
        monkeypatch.setattr(
            gate_module,
            "_run_case",
            lambda *args, **kwargs: {
                "id": "case-1",
                "errors": [],
                "warnings": [],
                "generated_tokens_median": 1,
                "total_wall_sec_median": 0.01,
                "generate_wall_sec_median": 0.01,
                "decode_tokens_per_sec": 100.0,
            },
        )
        monkeypatch.setattr(
            gate_module, "_summarize", lambda rows: {"total_wall_sec_sum": 0.01}
        )
        monkeypatch.setattr(
            gate_module, "_quality_gate", lambda *args, **kwargs: {"issues": []}
        )
        monkeypatch.setattr(gate_module, "_cuda_peak_allocated_mb", lambda device: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gate_translation.py",
                "--dataset",
                str(dataset),
                "--model",
                "org/model",
                "--model-revision",
                "abc123",
                "--decode-backend",
                "generate",
                "--no-w8a16",
                "--no-fused-rmsnorm",
                "--output-json",
                str(output),
            ],
        )

        gate_module.main()
        capsys.readouterr()
        payload = json.loads(output.read_text(encoding="utf-8"))

        assert constructed["model"] == "org/model"
        assert constructed["model_revision"] == "abc123"
        assert constructed["w8a16"] is False
        assert constructed["fused_rmsnorm"] is False
        assert payload["model_revision"] == "abc123"
        assert payload["resolved_model_commit"] == "abc123"
        assert payload["w8a16"] is False
        assert payload["w8a16_patch_count"] == 0
        assert payload["fused_rmsnorm"] is False
        assert payload["fused_rmsnorm_patch_count"] == 0

    def test_optimization_patch_count_zero_is_error(self) -> None:
        translator = SimpleNamespace(
            w8a16=True,
            w8a16_patch_count=0,
            fused_rmsnorm=True,
            fused_rmsnorm_patch_count=0,
        )

        issues = _optimization_patch_issues(translator)

        assert {issue.code for issue in issues} == {
            "w8a16_patch_count_zero",
            "fused_rmsnorm_patch_count_zero",
        }
        assert {issue.severity for issue in issues} == {"error"}

    def test_main_fails_before_warmup_when_requested_optimization_did_not_patch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text(
            json.dumps({"id": "case-1", "text": "hello", "target_language": "Chinese"})
            + "\n",
            encoding="utf-8",
        )

        class FakeTranslator:
            attn_implementation = "sdpa"
            decode_backend = "fixed_mask"
            w8a16 = True
            w8a16_patch_count = 0
            fused_rmsnorm = False
            fused_rmsnorm_patch_count = 0
            model_revision = None
            resolved_model_commit = None

            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

            def profile_translate(
                self, *args: object, **kwargs: object
            ) -> SimpleNamespace:
                del args, kwargs
                raise AssertionError("gate should fail before warmup")

        monkeypatch.setattr(gate_module, "HYMTTranslator", FakeTranslator)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gate_translation.py",
                "--dataset",
                str(dataset),
                "--no-fused-rmsnorm",
            ],
        )

        with pytest.raises(SystemExit):
            gate_module.main()
        captured = capsys.readouterr()

        assert "w8a16_patch_count_zero" in captured.err

    def test_parse_json_object_rejects_non_object(self) -> None:
        assert _parse_json_object('{"logits_to_keep": 1}') == {"logits_to_keep": 1}

        with pytest.raises(ValueError, match="JSON object"):
            _parse_json_object("[]")

    def test_target_language_mismatch_is_error(self) -> None:
        issues = _evaluate_quality(
            {"text": "hello world", "target_language": "Chinese"},
            "hello world",
            generated_tokens=2,
            max_new_tokens=16,
        )

        assert "target_language_mismatch" in {issue.code for issue in issues}

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

        assert not [issue for issue in issues if issue.severity == "error"]

    def test_missing_required_url_is_error(self) -> None:
        issues = _evaluate_quality(
            {"text": "open https://example.com now", "target_language": "Chinese"},
            "现在打开这个网站",
            generated_tokens=8,
            max_new_tokens=32,
        )

        assert "missing_urls" in {issue.code for issue in issues}

    def test_extract_format_markers_finds_srt_and_html(self) -> None:
        markers = _extract_format_markers(
            '00:00:01,000 <span data-x="1">hello</span> $USER'
        )

        assert markers["srt_timestamps"] == ["00:00:01,000"]
        assert markers["html_tags"] == ['<span data-x="1">', "</span>"]
        assert markers["placeholders"] == ["$USER"]

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

        assert "malformed_markdown_table" in {issue.code for issue in issues}

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

        assert "missing_required_output" in {issue.code for issue in issues}

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

        assert "missing_must_preserve" in {issue.code for issue in issues}
        assert [
            issue.severity for issue in issues if issue.code == "missing_must_preserve"
        ] == ["warning"]

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

        assert "missing_must_preserve" in {issue.code for issue in issues}

    def test_reference_similarity_tracks_content_overlap(self) -> None:
        strong = _reference_similarity(
            {"reference": "The meeting has started. Please turn on captions."},
            "The meeting has started. Please turn on the subtitles.",
        )
        weak = _reference_similarity(
            {"reference": "The meeting has started. Please turn on captions."},
            "Payment failed. Please contact support.",
        )

        assert strong is not None
        assert weak is not None
        assert strong > weak


class TestTranslationGatePerformance:
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

        assert summary["total_wall_sec_sum"] == 0.6
        assert summary["generated_tokens_sum"] == 53.0
        assert summary["decode_tokens_per_sec_total"] == 100.0
        assert summary["end_to_end_tokens_per_sec_total"] == 88.33
        assert summary["decode_tokens_per_sec_median"] == 100.0
        # Field renamed: reference_similarity holds chrF2 (0-100), surfaced as chrf_median.
        assert summary["chrf_median"] == 0.8

    def test_performance_issue_uses_baseline_speedup(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
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

        assert summary["speedup_vs_baseline"] == 1.25
        assert "speedup_below_threshold" in {issue.code for issue in issues}


class TestTranslationGateRegression:
    def test_run_config_diff_compares_legacy_stock_golden_generation(
        self, tmp_path: Path
    ) -> None:
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "generator": "stock_transformers",
                    "decode": "greedy",
                    "repetition_penalty": 1.05,
                    "max_new_tokens": 512,
                }
            ),
            encoding="utf-8",
        )

        diff = _run_config_diff(
            baseline_path,
            {
                "max_new_tokens": 512,
                "generation": {
                    "do_sample": False,
                    "top_k": 20,
                    "repetition_penalty": 1.2,
                    "extra_generate_kwargs": {"logits_to_keep": 1},
                },
            },
        )

        assert diff == {
            "generation": {
                "repetition_penalty": {"baseline": 1.05, "candidate": 1.2},
                "extra_generate_kwargs": {
                    "baseline": {},
                    "candidate": {"logits_to_keep": 1},
                },
            }
        }

    def test_quality_baseline_allows_existing_errors(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline_gate.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "reference_metric": "chrf2",
                    "cases": [
                        {
                            "id": "case-1",
                            "errors": [{"code": "missing_code_fences"}],
                            "warnings": [],
                        }
                    ],
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

        result = _quality_gate(
            rows, quality_baseline_json=baseline_path, fail_on_warnings=False
        )

        assert result["new_error_count"] == 0
        assert not result["issues"]
        assert rows[0]["new_errors"] == []

    def test_quality_baseline_reports_new_errors(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline_gate.json"
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

        result = _quality_gate(
            rows, quality_baseline_json=baseline_path, fail_on_warnings=False
        )

        assert result["new_error_count"] == 1
        assert "new_quality_errors" in {issue.code for issue in result["issues"]}
        assert rows[0]["new_errors"] == [{"code": "target_language_mismatch"}]

    def test_quality_baseline_reports_reference_similarity_drop(
        self, tmp_path: Path
    ) -> None:
        baseline_path = tmp_path / "baseline_gate.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "reference_metric": "chrf2",
                    "cases": [
                        {
                            "id": "case-1",
                            "errors": [],
                            "warnings": [],
                            "reference_similarity": 92.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "id": "case-1",
                "errors": [],
                "warnings": [],
                "reference_similarity": 70.0,
            }
        ]

        result = _quality_gate(
            rows, quality_baseline_json=baseline_path, fail_on_warnings=False
        )

        # A per-case chrF drop is recorded and flagged for human review, but is
        # deliberately NOT promoted to an error (single-reference chrF is too noisy
        # on one sentence; systematic loss is judged per direction via
        # --max-mean-chrf-drop). So it produces no new error here.
        assert result["new_error_count"] == 0
        assert result["compared_case_count"] == 1
        assert result["mean_chrf_drop"] == 22.0
        assert result["case_chrf_drop_flag_count"] == 1
        assert rows[0]["reference_similarity_drop"] == 22.0
