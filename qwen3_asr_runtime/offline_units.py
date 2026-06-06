# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import unicodedata
from typing import Any, Iterable, Sequence


_WEAK_END = frozenset("，、,;；:：")
_STRONG_UNIT_END = frozenset("。！？!?")
_DEFAULT_MAX_SOURCE_UNIT_MS = 15_000
_DEFAULT_MAX_SOURCE_UNIT_WIDTH = 180
_ESTIMATED_DURATION_MIN_WIDTH = 60
_MIN_SOURCE_CUE_MS = 80


@dataclass(frozen=True)
class TimedToken:
    text: str
    start_ms: int
    end_ms: int
    starts_new_batch: bool = False
    timing_status: str = "aligned"


@dataclass(frozen=True)
class SourceUnit:
    text: str
    language: str
    timing_status: str
    tokens: tuple[TimedToken, ...]


@dataclass(frozen=True)
class SourceCue:
    text: str
    start_ms: int
    end_ms: int


class SourceUnitBuilder:
    def __init__(
        self,
        *,
        max_unit_ms: int = _DEFAULT_MAX_SOURCE_UNIT_MS,
        max_unit_width: int = _DEFAULT_MAX_SOURCE_UNIT_WIDTH,
    ) -> None:
        self._pending: list[TimedToken] = []
        self._pending_language = ""
        self._max_unit_ms = max(1, int(max_unit_ms))
        self._max_unit_width = max(1, int(max_unit_width))

    @property
    def pending_tokens(self) -> tuple[TimedToken, ...]:
        return tuple(self._pending)

    def add_tokens(
        self,
        tokens: Sequence[TimedToken],
        *,
        language: str = "",
        timing_status: str = "aligned",
    ) -> list[SourceUnit]:
        language = str(language or "")
        timing_status = str(timing_status or "aligned")
        pending_tokens = [
            replace(token, timing_status=timing_status)
            for token in tokens
            if str(token.text or "").strip()
        ]
        if not pending_tokens:
            return []

        units: list[SourceUnit] = []
        if self._pending and language != self._pending_language:
            units.extend(self.flush())
        if not self._pending:
            self._pending_language = language
        else:
            pending_tokens[0] = replace(pending_tokens[0], starts_new_batch=True)
        self._pending.extend(pending_tokens)
        units.extend(self._drain(is_final=False))
        return units

    def flush(self) -> list[SourceUnit]:
        return self._drain(is_final=True)

    def take_pending_unit(self) -> SourceUnit | None:
        unit = self._make_unit(self._pending)
        self._pending = []
        self._pending_language = ""
        return unit

    def _drain(self, *, is_final: bool) -> list[SourceUnit]:
        units: list[SourceUnit] = []
        while self._pending:
            close_index = self._next_close_index(is_final=is_final)
            if close_index is None:
                break
            tokens = self._pending[: close_index + 1]
            del self._pending[: close_index + 1]
            unit = self._make_unit(tokens)
            if unit is not None:
                units.append(unit)
        if not self._pending:
            self._pending_language = ""
        return units

    def _next_close_index(self, *, is_final: bool) -> int | None:
        if not self._pending:
            return None
        last_safe = -1
        for index, token in enumerate(self._pending):
            if _is_unit_end_token(self._pending, index):
                return index
            current = self._pending[: index + 1]
            current_width = _tokens_text_width(current)
            current_ms = int(current[-1].end_ms) - int(current[0].start_ms)
            timing_status = _tokens_timing_status(current)
            can_close_by_duration = timing_status == "aligned" or current_width >= _ESTIMATED_DURATION_MIN_WIDTH
            duration_at_limit = can_close_by_duration and current_ms >= self._max_unit_ms
            duration_over_limit = can_close_by_duration and current_ms > self._max_unit_ms
            if current_width >= self._max_unit_width or duration_at_limit:
                return _close_index_for_limit(
                    index,
                    last_safe=last_safe,
                    hard_over_limit=current_width > self._max_unit_width or duration_over_limit,
                )
            last_safe = index
        return len(self._pending) - 1 if is_final else None

    def _make_unit(self, tokens: Sequence[TimedToken]) -> SourceUnit | None:
        text = _join_token_texts(tokens)
        if not text:
            return None
        return SourceUnit(
            text=text,
            language=self._pending_language,
            timing_status=_tokens_timing_status(tokens),
            tokens=tuple(tokens),
        )


def timed_tokens_from_aligned_items(
    text: str,
    items: Iterable[Any],
    *,
    base_ms: int = 0,
    duration_ms: int | None = None,
) -> list[TimedToken]:
    item_list = [item for item in items if str(getattr(item, "text", "") or "").strip()]
    if not item_list:
        return []

    source_text = str(text or "")
    spans = _find_item_spans(source_text, [str(getattr(item, "text", "") or "") for item in item_list])
    span_list: list[tuple[int, int]] = []
    for span in spans:
        if span is None:
            return []
        span_list.append(span)
    tokens: list[TimedToken] = []
    max_duration_ms = None if duration_ms is None else max(0, int(duration_ms))
    if max_duration_ms == 0:
        return []
    display_start = 0
    for index, item in enumerate(item_list):
        start, end = span_list[index]
        next_start = span_list[index + 1][0] if index + 1 < len(span_list) else len(source_text)
        display_end = _aligned_token_display_end(source_text, end, next_start)
        display_text = source_text[display_start:display_end]
        if not display_text.strip():
            display_text = source_text[start:end]
        display_start = display_end
        start_time = float(getattr(item, "start_time", 0.0))
        end_time = float(getattr(item, "end_time", 0.0))
        if not math.isfinite(start_time) or not math.isfinite(end_time) or end_time < start_time:
            return []
        local_start_ms = max(0, int(round(start_time * 1000)))
        local_end_ms = max(local_start_ms, int(round(end_time * 1000)))
        if max_duration_ms is not None:
            if local_start_ms >= max_duration_ms:
                return []
            local_end_ms = min(local_end_ms, max_duration_ms)
            local_start_ms = min(local_start_ms, local_end_ms)
        if tokens and local_start_ms < int(tokens[-1].end_ms) - int(base_ms):
            return []
        start_ms = int(base_ms) + local_start_ms
        end_ms = int(base_ms) + local_end_ms
        tokens.append(TimedToken(text=display_text, start_ms=start_ms, end_ms=end_ms))
    return tokens


def estimated_timed_tokens_from_text(
    text: str,
    *,
    base_ms: int = 0,
    duration_ms: int | None = None,
) -> list[TimedToken]:
    source_text = str(text or "")
    if not source_text.strip():
        return []
    duration = max(0, int(duration_ms or 0))
    if duration <= 0:
        return []

    spans = _estimated_token_spans(source_text)
    if not spans:
        return []
    weights = [max(1, _display_width(source_text[start:end].strip())) for start, end in spans]
    total_weight = sum(weights)
    elapsed = 0
    tokens: list[TimedToken] = []
    for (start, end), weight in zip(spans, weights):
        token_start = int(base_ms) + int(round(duration * elapsed / total_weight))
        elapsed += weight
        token_end = int(base_ms) + int(round(duration * elapsed / total_weight))
        tokens.append(
            TimedToken(
                text=source_text[start:end],
                start_ms=token_start,
                end_ms=max(token_start, token_end),
            )
        )
    return tokens


def layout_source_cues(
    unit: SourceUnit,
    *,
    max_cue_ms: int = 6000,
    max_cue_width: int = 72,
    min_cue_width: int = 12,
) -> list[SourceCue]:
    tokens = list(unit.tokens)
    if not tokens:
        return []
    max_cue_ms = max(1, int(max_cue_ms))
    max_cue_width = max(1, int(max_cue_width))
    min_cue_width = max(0, min(int(min_cue_width), max_cue_width))
    if _tokens_fit_one_cue(tokens, max_cue_ms=max_cue_ms, max_cue_width=max_cue_width):
        groups = [tokens]
    else:
        groups = []
        start = 0
        while start < len(tokens):
            close_index = _next_cue_close_index(
                tokens,
                start,
                max_cue_ms=max_cue_ms,
                max_cue_width=max_cue_width,
                min_cue_width=min_cue_width,
            )
            groups.append(tokens[start : close_index + 1])
            start = close_index + 1
    if unit.timing_status != "aligned":
        return _make_estimated_cues(groups, max_cue_ms=max_cue_ms)
    return [cue for group in groups if (cue := _make_cue(group)) is not None]


def _make_estimated_cues(groups: Sequence[Sequence[TimedToken]], *, max_cue_ms: int) -> list[SourceCue]:
    cues: list[SourceCue] = []
    if not groups:
        return cues
    cursor_ms = int(groups[0][0].start_ms)
    for group in groups:
        text = _join_token_texts(group)
        if not text:
            continue
        raw_duration_ms = max(1, int(group[-1].end_ms) - int(group[0].start_ms))
        duration_ms = min(max(1, int(max_cue_ms)), raw_duration_ms)
        cues.append(SourceCue(text=text, start_ms=cursor_ms, end_ms=cursor_ms + duration_ms))
        cursor_ms += duration_ms
    return cues


def _aligned_token_display_end(text: str, item_end: int, next_item_start: int) -> int:
    display_end = max(0, int(next_item_start))
    item_end = max(0, int(item_end))
    while display_end > item_end and str(text[display_end - 1]).isspace():
        display_end -= 1
    return display_end


def _next_cue_close_index(
    tokens: Sequence[TimedToken],
    start: int,
    *,
    max_cue_ms: int,
    max_cue_width: int,
    min_cue_width: int,
) -> int:
    last_safe = -1
    for index in range(start, len(tokens)):
        current = tokens[start : index + 1]
        remaining = tokens[index + 1 :]
        current_width = _tokens_text_width(current)
        current_ms = int(current[-1].end_ms) - int(current[0].start_ms)
        has_aligned_timing = _tokens_timing_status(current) == "aligned"
        duration_at_limit = has_aligned_timing and current_ms >= max_cue_ms
        duration_over_limit = has_aligned_timing and current_ms > max_cue_ms
        at_limit = current_width >= max_cue_width or duration_at_limit
        if at_limit:
            close_index = _close_index_for_limit(
                index,
                last_safe=last_safe,
                hard_over_limit=current_width > max_cue_width or duration_over_limit,
            )
            return close_index
        if remaining:
            text = str(tokens[index].text or "").rstrip()
            if (
                _ends_with_any(text, _WEAK_END)
                and current_width >= min_cue_width
                and _tokens_text_width(remaining) >= min_cue_width
            ):
                return index
        last_safe = index
    return len(tokens) - 1


def _tokens_fit_one_cue(
    tokens: Sequence[TimedToken],
    *,
    max_cue_ms: int,
    max_cue_width: int,
) -> bool:
    if not tokens:
        return False
    duration_ms = int(tokens[-1].end_ms) - int(tokens[0].start_ms)
    duration_fits = _tokens_timing_status(tokens) != "aligned" or duration_ms <= max_cue_ms
    return duration_fits and _tokens_text_width(tokens) <= max_cue_width


def _close_index_for_limit(
    index: int,
    *,
    last_safe: int,
    hard_over_limit: bool,
) -> int:
    if hard_over_limit and last_safe >= 0:
        return last_safe
    return index


def _make_cue(tokens: Sequence[TimedToken]) -> SourceCue | None:
    text = _join_token_texts(tokens)
    if not text:
        return None
    start_ms = int(tokens[0].start_ms)
    end_ms = max(start_ms, int(tokens[-1].end_ms))
    if end_ms <= start_ms:
        if start_ms >= _MIN_SOURCE_CUE_MS:
            start_ms -= _MIN_SOURCE_CUE_MS
        else:
            end_ms = start_ms + _MIN_SOURCE_CUE_MS
    return SourceCue(
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _find_item_spans(text: str, item_texts: Sequence[str]) -> list[tuple[int, int] | None]:
    spans: list[tuple[int, int] | None] = []
    normalized_text, original_indices = _alignment_index(text)
    lower_text = normalized_text.lower()
    cursor = 0
    for raw_item in item_texts:
        item = _alignment_key(str(raw_item or ""))
        if not item:
            spans.append(None)
            continue
        start = lower_text.find(item.lower(), cursor)
        if start < 0:
            spans.append(None)
            continue
        end = start + len(item)
        spans.append((original_indices[start], original_indices[end - 1] + 1))
        cursor = end
    return spans


def _estimated_token_spans(text: str) -> list[tuple[int, int]]:
    if _has_word_spacing(text):
        return _word_token_spans(text)
    return [(index, index + 1) for index, char in enumerate(text) if not char.isspace()]


def _has_word_spacing(text: str) -> bool:
    return any(char.isspace() for char in text)


def _word_token_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        token_start = index
        if _is_word_char(text[index]):
            index += 1
            while index < len(text) and _is_word_char(text[index]):
                index += 1
        else:
            index += 1
        spans.append((token_start, index))
    return _spans_with_interstitial_text(text, spans)


def _spans_with_interstitial_text(text: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    cursor = 0
    for start, end in spans:
        if text[start:end].strip():
            expanded.append((cursor, end))
            cursor = end
    return expanded


def _join_token_texts(tokens: Sequence[TimedToken]) -> str:
    joined = ""
    previous_token: TimedToken | None = None
    for token in tokens:
        text = str(token.text or "")
        if not text.strip():
            continue
        if joined and _needs_text_separator(joined, text, previous_token, token):
            joined += " "
        joined += text
        previous_token = token
    return joined.strip()


def _tokens_text_width(tokens: Sequence[TimedToken]) -> int:
    return _display_width(_join_token_texts(tokens))


def _tokens_timing_status(tokens: Sequence[TimedToken]) -> str:
    statuses = {str(token.timing_status or "estimated") for token in tokens}
    return "aligned" if statuses == {"aligned"} else "estimated"


def _display_width(text: str) -> int:
    width = 0
    for char in str(text or ""):
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _ends_with_any(text: str, chars: frozenset[str]) -> bool:
    return bool(text) and text[-1] in chars


def _is_unit_end_token(tokens: Sequence[TimedToken], index: int) -> bool:
    text = str(tokens[index].text or "").rstrip()
    if not text:
        return False
    if text[-1] in _STRONG_UNIT_END:
        return True
    if text[-1] != ".":
        return False
    return not _period_is_numeric_separator(tokens, index, text)


def _period_is_numeric_separator(tokens: Sequence[TimedToken], index: int, text: str) -> bool:
    left = _last_non_space(text[:-1])
    if left is None and index > 0:
        left = _last_non_space(str(tokens[index - 1].text or ""))
    if left is None or not left.isdigit():
        return False

    if index + 1 >= len(tokens):
        return False
    next_text = str(tokens[index + 1].text or "")
    if not next_text or next_text[0].isspace():
        return False
    right = _first_non_space(next_text)
    return right is not None and right.isdigit()


def _needs_text_separator(
    joined: str,
    text: str,
    previous_token: TimedToken | None,
    token: TimedToken,
) -> bool:
    if joined[-1].isspace() or text[0].isspace():
        return False
    left = _last_non_space(joined)
    right = _first_non_space(text)
    if left is None or right is None or not _can_separate_before_ascii_token(left, right):
        return False
    if previous_token is None:
        return False
    return bool(token.starts_new_batch) or int(token.start_ms) > int(previous_token.end_ms)


def _can_separate_before_ascii_token(left: str, right: str) -> bool:
    if not _is_ascii_alnum(right):
        return False
    return _is_ascii_alnum(left) or left in ",.!?;:"


def _alignment_index(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    original_indices: list[int] = []
    for index, char in enumerate(text):
        if _is_alignment_char(char):
            normalized_chars.append(char)
            original_indices.append(index)
    return "".join(normalized_chars), original_indices


def _alignment_key(text: str) -> str:
    return "".join(char for char in text if _is_alignment_char(char))


def _is_alignment_char(char: str) -> bool:
    if char == "'":
        return True
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("N")


def _is_word_char(char: str) -> bool:
    return char == "'" or unicodedata.category(char)[0] in {"L", "N"}


def _last_non_space(text: str) -> str | None:
    for char in reversed(text):
        if not char.isspace():
            return char
    return None


def _first_non_space(text: str) -> str | None:
    for char in text:
        if not char.isspace():
            return char
    return None


def _is_ascii_alnum(value: str) -> bool:
    return len(value) == 1 and value.isascii() and value.isalnum()


__all__ = [
    "SourceCue",
    "SourceUnit",
    "SourceUnitBuilder",
    "TimedToken",
    "estimated_timed_tokens_from_text",
    "layout_source_cues",
    "timed_tokens_from_aligned_items",
]
