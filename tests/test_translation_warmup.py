# coding=utf-8
from __future__ import annotations

from tools.translation_warmup import (
    resolve_translation_warmup_count,
    select_translation_warmup_cases,
)


def test_profile_warmup_selects_representative_cases_per_direction() -> None:
    cases = [
        {
            "id": "en-zh-short",
            "source_language": "English",
            "target_language": "Chinese",
            "text": "hi",
        },
        {
            "id": "zh-en-short",
            "source_language": "Chinese",
            "target_language": "English",
            "text": "你好",
        },
        {
            "id": "en-zh-mid",
            "source_language": "English",
            "target_language": "Chinese",
            "text": "hello " * 20,
        },
        {
            "id": "zh-en-mid",
            "source_language": "Chinese",
            "target_language": "English",
            "text": "你好" * 20,
        },
        {
            "id": "en-zh-long",
            "source_language": "English",
            "target_language": "Chinese",
            "text": "hello " * 80,
        },
        {
            "id": "zh-en-long",
            "source_language": "Chinese",
            "target_language": "English",
            "text": "你好" * 80,
        },
    ]

    selected = select_translation_warmup_cases(cases, "profile")

    assert [case["id"] for case in selected] == [
        "en-zh-short",
        "en-zh-mid",
        "en-zh-long",
        "zh-en-short",
        "zh-en-mid",
        "zh-en-long",
    ]


def test_numeric_warmup_preserves_first_n_behavior_with_repeats() -> None:
    cases = [{"id": "a"}, {"id": "b"}]

    selected = select_translation_warmup_cases(cases, "3")

    assert [case["id"] for case in selected] == ["a", "b", "a"]


def test_resolve_warmup_count_accepts_all() -> None:
    assert resolve_translation_warmup_count("all", 42) == 42
    assert resolve_translation_warmup_count("-1", 42) == 0
