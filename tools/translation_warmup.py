# coding=utf-8
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any


def select_translation_warmup_cases(
    cases: Sequence[Mapping[str, Any]], spec: str
) -> list[Mapping[str, Any]]:
    """Select warmup cases before timing translation steady state.

    Numeric warmups preserve the historical "first N cases" behavior. The
    default profile mode warms representative source lengths for every selected
    source/target direction so one-time compile and shape generalization costs
    do not land in measured rows.
    """

    if not cases:
        return []
    text = str(spec).strip().lower()
    if text in {"profile", "representative", "auto"}:
        return _representative_cases_by_direction(cases)
    count = resolve_translation_warmup_count(text, len(cases))
    return [cases[index % len(cases)] for index in range(count)]


def resolve_translation_warmup_count(value: str, case_count: int) -> int:
    text = str(value).strip().lower()
    if text == "all":
        return max(0, int(case_count))
    return max(0, int(text))


def _representative_cases_by_direction(
    cases: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_direction: OrderedDict[tuple[str, str], list[Mapping[str, Any]]] = OrderedDict()
    for case in cases:
        by_direction.setdefault(_direction_key(case), []).append(case)

    selected: list[Mapping[str, Any]] = []
    for direction_cases in by_direction.values():
        selected.extend(_short_middle_long(direction_cases))
    return selected


def _direction_key(case: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(case.get("source_language") or "").strip(),
        str(case.get("target_language") or "").strip(),
    )


def _short_middle_long(
    cases: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(cases) <= 3:
        return list(cases)
    by_length = sorted(
        enumerate(cases),
        key=lambda item: (_source_text_length(item[1]), item[0]),
    )
    indexes = sorted({0, len(by_length) // 2, len(by_length) - 1})
    return [by_length[index][1] for index in indexes]


def _source_text_length(case: Mapping[str, Any]) -> int:
    return len(str(case.get("text") or ""))
