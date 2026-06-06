# coding=utf-8
from __future__ import annotations

import sys

import pytest

from tools.benchmark_translation import (
    _parse_args,
    _parse_json_object,
    _pearson,
    _resolve_warmup_count,
    _run_case,
    _summarize,
)


class FakeResult:
    text = "translated text"
    prompt_tokens = 5
    generated_tokens = 3
    encode_wall_sec = 0.01
    generate_wall_sec = 0.25
    decode_wall_sec = 0.02
    total_wall_sec = 0.3


class FakeTranslator:
    def profile_translate(self, *args: object, **kwargs: object) -> FakeResult:
        return FakeResult()


class TestBenchmarkTranslation:
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
            ["benchmark_translation.py", "--dataset", "cases.jsonl", *extra_args],
        )

        args = _parse_args()

        assert args.sample is expected_sample
        assert not args.greedy

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("all", 42),
            ("3", 3),
            ("-1", 0),
        ],
    )
    def test_resolve_warmup_count_accepts_all(self, raw: str, expected: int) -> None:
        assert _resolve_warmup_count(raw, 42) == expected

    def test_parse_json_object_rejects_non_object(self) -> None:
        assert _parse_json_object('{"logits_to_keep": 1}') == {"logits_to_keep": 1}

        with pytest.raises(ValueError, match="JSON object"):
            _parse_json_object("[]")

    def test_run_case_omits_output_by_default(self) -> None:
        row = _run_case(
            FakeTranslator(),
            {"id": "case1", "target_language": "English", "text": "source text"},
            repeats=1,
            max_new_tokens=16,
        )

        assert row["output_chars"] == len(FakeResult.text)
        assert row["encode_wall_sec_median"] == 0.01
        assert row["decode_wall_sec_median"] == 0.02
        assert "output" not in row

    def test_summarize_reports_generate_token_correlation(self) -> None:
        rows = [
            {
                "prompt_tokens": 10,
                "generated_tokens_median": 2,
                "total_wall_sec_median": 0.2,
                "encode_wall_sec_median": 0.01,
                "generate_wall_sec_median": 0.15,
                "decode_wall_sec_median": 0.01,
                "decode_tokens_per_sec": 13.3,
            },
            {
                "prompt_tokens": 20,
                "generated_tokens_median": 4,
                "total_wall_sec_median": 0.4,
                "encode_wall_sec_median": 0.01,
                "generate_wall_sec_median": 0.35,
                "decode_wall_sec_median": 0.01,
                "decode_tokens_per_sec": 11.4,
            },
        ]

        summary = _summarize(rows)

        assert summary["generate_wall_sec_sum"] == 0.5
        assert summary["generated_tokens_sum"] == 6.0
        assert summary["decode_tokens_per_sec_total"] == 12.0
        assert summary["end_to_end_tokens_per_sec_total"] == 10.0
        assert summary["correlation"]["generated_tokens_vs_total_sec"] == 1.0

    def test_pearson_returns_none_for_constant_values(self) -> None:
        assert _pearson([1.0, 1.0], [2.0, 3.0]) is None
