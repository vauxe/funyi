# coding=utf-8
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SubtitleLine:
    text: str
    start_ms: int | None
    end_ms: int | None
    language: str = ""
    id: str | None = None
    index: int | None = None
    source_revision: int | None = None
    timing_status: str | None = None
    translation: str | None = None
    translation_status: str | None = None
    translation_message: str | None = None


@dataclass(frozen=True)
class SubtitleWindow:
    previous: SubtitleLine | None
    current: SubtitleLine | None


@dataclass(frozen=True)
class _StableTranslationUnit:
    source_segment_ids: tuple[str, ...]
    source_segment_indices: tuple[int, ...]
    text: str | None = None
    translation_status: str | None = None
    translation_message: str | None = None


@dataclass(frozen=True)
class _SourceLineIndex:
    offset_by_id: dict[str, int]
    offset_by_index: dict[int, int]


class SubtitleDocument:
    """Client-side replay model for realtime subtitle events.

    Source transcript events own timing and scroll state. Stable translation
    events annotate adjacent source segments and are projected into display/SRT
    lines at render time.
    """

    def __init__(self, *, translation_enabled: bool = True) -> None:
        self.translation_enabled = bool(translation_enabled)
        self.revision = 0
        self._stable_lines: list[SubtitleLine] = []
        self._stable_line_offset_by_id: dict[str, int] = {}
        self._stable_line_offset_by_index: dict[int, int] = {}
        self._stable_projection: list[SubtitleLine] | None = None
        self._stable_translation_units: list[_StableTranslationUnit] = []
        self._stable_translation_unit_index: dict[str, int] = {}
        self._pending_stable_unit_ids: list[str] = []
        self.current: SubtitleLine | None = None

    @property
    def stable_lines(self) -> list[SubtitleLine]:
        return list(self._projected_stable_lines())

    def set_translation_enabled(self, enabled: bool) -> None:
        self.translation_enabled = bool(enabled)

    def apply_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "transcript_update":
            self._apply_transcript_update(event)
        elif event_type == "transcript_timing_update":
            self._apply_transcript_timing_update(event)
        elif event_type == "transcript_final":
            self._apply_transcript_final(event)
        elif event_type == "translation_stable":
            self._apply_stable_translation(event)
        elif event_type == "translation_preview":
            self._apply_preview_translation(event)
        elif event_type == "translation_status":
            self._apply_translation_status(event)

    def window(self, *, include_translation: bool | None = None) -> SubtitleWindow:
        enabled = (
            self.translation_enabled
            if include_translation is None
            else bool(include_translation)
        )
        durable_lines = self._projected_stable_lines(include_translation=enabled)
        previous = durable_lines[-1] if durable_lines else None
        return SubtitleWindow(
            previous=_render_line(previous, include_translation=enabled),
            current=_render_line(
                self._current_display_line(include_translation=enabled),
                include_translation=enabled,
            ),
        )

    def to_srt(self, *, include_translation: bool | None = None) -> str:
        enabled = (
            self.translation_enabled
            if include_translation is None
            else bool(include_translation)
        )
        blocks: list[str] = []
        number = 1
        for line in self._projected_stable_lines(include_translation=enabled):
            if line.start_ms is None or line.end_ms is None:
                continue
            text_lines = [line.text]
            if enabled and line.translation:
                text_lines.append(line.translation)
            blocks.append(
                "\n".join(
                    [
                        str(number),
                        f"{_format_srt_time(line.start_ms)} --> {_format_srt_time(line.end_ms)}",
                        *text_lines,
                    ]
                )
            )
            number += 1
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _apply_transcript_update(self, event: dict[str, Any]) -> None:
        stable_base = int(event.get("stable_base") or 0)
        stable_appends = event.get("stable_appends") or []
        stable_count = int(event.get("stable_count") or 0)
        if stable_base != len(self._stable_lines):
            raise ValueError(
                f"stable cursor mismatch: stable_base={stable_base}, local_count={len(self._stable_lines)}"
            )
        if stable_base + len(stable_appends) != stable_count:
            raise ValueError(
                f"stable count mismatch: stable_base={stable_base}, "
                f"appends={len(stable_appends)}, stable_count={stable_count}"
            )

        revision = int(event.get("revision") or self.revision)
        partial = event.get("partial")
        for segment in stable_appends:
            if isinstance(segment, dict):
                line = _line_from_segment(segment, source_revision=revision)
                offset = len(self._stable_lines)
                self._stable_lines.append(line)
                self._index_stable_line(offset, line)
                if isinstance(partial, dict) and line.id:
                    self._pending_stable_unit_ids.append(line.id)
        if stable_appends:
            self._invalidate_stable_projection()

        self.current = (
            _line_from_segment(partial, source_revision=revision)
            if isinstance(partial, dict)
            else None
        )
        if self.current is None:
            self._pending_stable_unit_ids = []
        self.revision = revision

    def _apply_transcript_timing_update(self, event: dict[str, Any]) -> None:
        index = self._stable_index(event)
        if index is None:
            return
        self._stable_lines[index] = replace(
            self._stable_lines[index],
            start_ms=_optional_int(event.get("start_ms")),
            end_ms=_optional_int(event.get("end_ms")),
            timing_status=str(event.get("timing_status") or "") or None,
        )
        self._invalidate_stable_projection()

    def _apply_transcript_final(self, event: dict[str, Any]) -> None:
        if "segments" not in event:
            self.current = None
            self._pending_stable_unit_ids = []
            self.revision = int(event.get("revision") or self.revision)
            return

        lines: list[SubtitleLine] = []
        segment_units: list[_StableTranslationUnit] = []
        explicitly_translated_ids: set[str] = set()
        explicitly_translated_indices: set[int] = set()
        for segment in event.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            line = _line_from_segment(
                segment, source_revision=int(event.get("revision") or self.revision)
            )
            if _segment_has_translation_state(segment):
                if line.id:
                    explicitly_translated_ids.add(line.id)
                if line.index is not None:
                    explicitly_translated_indices.add(line.index)
                unit = _translation_unit_from_segment(segment)
                if unit is not None:
                    segment_units.append(unit)
            lines.append(line)
        self._stable_lines = lines
        self._rebuild_stable_line_index()
        document = event.get("document")
        if isinstance(document, dict):
            self._replace_stable_translation_units(
                _translation_units_from_document(document)
            )
        elif explicitly_translated_ids or explicitly_translated_indices:
            self._replace_stable_translation_units(
                [
                    unit
                    for unit in self._stable_translation_units
                    if not _translation_unit_touches_coverage(
                        unit, explicitly_translated_ids, explicitly_translated_indices
                    )
                ]
                + segment_units
            )
        self._pending_stable_unit_ids = []
        self.current = None
        self.revision = int(event.get("revision") or self.revision)

    def _apply_stable_translation(self, event: dict[str, Any]) -> None:
        text = str(event.get("text") or "").strip()
        if not text:
            return
        if self._clear_translation_coverage_pending(event):
            self._clear_current_translation_preview()
        self._upsert_stable_translation_unit(
            _translation_unit_from_event(event, text=text)
        )

    def _apply_preview_translation(self, event: dict[str, Any]) -> None:
        if self.current is None:
            return
        source_revision = int(event.get("source_revision") or 0)
        if self.current.source_revision != source_revision:
            return
        text = str(event.get("text") or "").strip()
        if text:
            self.current = replace(self.current, translation=text)

    def _apply_translation_status(self, event: dict[str, Any]) -> None:
        if self._clear_translation_coverage_pending(event):
            self._clear_current_translation_preview()
        self._upsert_stable_translation_unit(
            _translation_unit_from_event(
                event,
                translation_status=str(event.get("code") or ""),
                translation_message=str(event.get("message") or ""),
            )
        )

    def _stable_index(self, event: dict[str, Any]) -> int | None:
        segment_id = str(event.get("source_segment_id") or "")
        if segment_id:
            index = self._stable_line_offset_by_id.get(segment_id)
            if index is not None:
                return index
        try:
            segment_index = int(event.get("source_segment_index") or 0)
        except (TypeError, ValueError):
            return None
        if segment_index <= 0:
            return None
        return self._stable_line_offset_by_index.get(segment_index)

    def _current_display_line(
        self, *, include_translation: bool
    ) -> SubtitleLine | None:
        if self.current is None:
            return None
        if not include_translation or not self.current.translation:
            return self.current
        pending_lines = self._pending_stable_lines()
        if not pending_lines:
            return self.current
        unit_line = _combined_source_line(
            [*pending_lines, self.current],
            translation=self.current.translation,
            translation_status=self.current.translation_status,
            translation_message=self.current.translation_message,
        )
        return replace(
            unit_line,
            id=self.current.id,
            index=self.current.index,
            source_revision=self.current.source_revision,
        )

    def _pending_stable_lines(self) -> list[SubtitleLine]:
        lines: list[SubtitleLine] = []
        for segment_id in self._pending_stable_unit_ids:
            offset = self._stable_line_offset_by_id.get(segment_id)
            if offset is not None and 0 <= offset < len(self._stable_lines):
                lines.append(self._stable_lines[offset])
        return lines

    def _clear_translation_coverage_pending(self, event: dict[str, Any]) -> bool:
        previous_count = len(self._pending_stable_unit_ids)
        if previous_count == 0:
            return False
        pending_ids = self._translation_coverage_ids(event)
        if not pending_ids:
            return False
        self._pending_stable_unit_ids = [
            segment_id
            for segment_id in self._pending_stable_unit_ids
            if segment_id not in pending_ids
        ]
        return len(self._pending_stable_unit_ids) < previous_count

    def _translation_coverage_ids(self, event: dict[str, Any]) -> set[str]:
        ids = set(_string_tuple(event.get("source_segment_ids")))
        anchor_id = str(event.get("source_segment_id") or "").strip()
        if anchor_id:
            ids.add(anchor_id)
        indices = set(_int_tuple(event.get("source_segment_indices")))
        anchor_index = _optional_int(event.get("source_segment_index"))
        if anchor_index is not None and anchor_index > 0:
            indices.add(anchor_index)
        for index in indices:
            offset = self._stable_line_offset_by_index.get(index)
            if offset is None or not 0 <= offset < len(self._stable_lines):
                continue
            segment_id = self._stable_lines[offset].id
            if segment_id:
                ids.add(segment_id)
        return ids

    def _clear_current_translation_preview(self) -> None:
        if self.current is None or self.current.translation is None:
            return
        self.current = replace(
            self.current,
            translation=None,
            translation_status=None,
            translation_message=None,
        )

    def _upsert_stable_translation_unit(self, unit: _StableTranslationUnit) -> None:
        key = _translation_coverage_key(unit)
        if key is None:
            return
        index = self._stable_translation_unit_index.get(key)
        if index is not None:
            previous = self._stable_translation_units[index]
            merged = replace(
                unit, text=unit.text if unit.text is not None else previous.text
            )
            self._stable_translation_units[index] = merged
            if not self._try_patch_stable_projection(merged):
                self._invalidate_stable_projection()
            return
        self._stable_translation_unit_index[key] = len(self._stable_translation_units)
        self._stable_translation_units.append(unit)
        if not self._try_patch_stable_projection(unit):
            self._invalidate_stable_projection()

    def _replace_stable_translation_units(
        self, units: list[_StableTranslationUnit]
    ) -> None:
        self._stable_translation_units = units
        self._stable_translation_unit_index = _translation_unit_index(units)
        self._invalidate_stable_projection()

    def _projected_stable_lines(
        self, *, include_translation: bool | None = None
    ) -> list[SubtitleLine]:
        enabled = (
            self.translation_enabled
            if include_translation is None
            else bool(include_translation)
        )
        if not enabled:
            return list(self._stable_lines)
        if self._stable_projection is None:
            self._stable_projection = _project_stable_lines(
                self._stable_lines,
                self._stable_translation_units,
                include_translation=True,
            )
        return self._stable_projection

    def _rebuild_stable_line_index(self) -> None:
        self._stable_line_offset_by_id = {}
        self._stable_line_offset_by_index = {}
        for offset, line in enumerate(self._stable_lines):
            self._index_stable_line(offset, line)
        self._invalidate_stable_projection()

    def _index_stable_line(self, offset: int, line: SubtitleLine) -> None:
        if line.id:
            self._stable_line_offset_by_id[line.id] = offset
        if line.index is not None:
            self._stable_line_offset_by_index[line.index] = offset

    def _invalidate_stable_projection(self) -> None:
        self._stable_projection = None

    def _try_patch_stable_projection(self, unit: _StableTranslationUnit) -> bool:
        if self._stable_projection is None or len(self._stable_projection) != len(
            self._stable_lines
        ):
            return False
        offset = self._stable_offset_for_single_translation_unit(unit)
        if offset is None or offset >= len(self._stable_lines):
            return False
        source = self._stable_lines[offset]
        self._stable_projection[offset] = replace(
            source,
            translation=unit.text,
            translation_status=unit.translation_status,
            translation_message=unit.translation_message,
        )
        return True

    def _stable_offset_for_single_translation_unit(
        self, unit: _StableTranslationUnit
    ) -> int | None:
        if _translation_coverage_len(unit) != 1:
            return None
        if unit.source_segment_indices:
            return self._stable_line_offset_by_index.get(unit.source_segment_indices[0])
        if unit.source_segment_ids:
            return self._stable_line_offset_by_id.get(unit.source_segment_ids[0])
        return None


def _line_from_segment(
    segment: dict[str, Any], *, source_revision: int
) -> SubtitleLine:
    text = str(segment.get("text") or "").strip()
    return SubtitleLine(
        id=str(segment.get("id") or "") or None,
        index=_optional_int(segment.get("index")),
        start_ms=_optional_int(segment.get("start_ms")),
        end_ms=_optional_int(segment.get("end_ms")),
        text=text,
        language=str(segment.get("language") or ""),
        source_revision=int(source_revision),
        timing_status=str(segment.get("timing_status") or "") or None,
    )


def _render_line(
    line: SubtitleLine | None, *, include_translation: bool
) -> SubtitleLine | None:
    if line is None or include_translation:
        return line
    return replace(
        line,
        translation=None,
        translation_status=None,
        translation_message=None,
    )


def _segment_has_translation_state(segment: dict[str, Any]) -> bool:
    return any(
        key in segment
        for key in ("translation", "translation_status", "translation_message")
    )


def _translation_units_from_document(
    document: dict[str, Any],
) -> list[_StableTranslationUnit]:
    units: list[_StableTranslationUnit] = []
    for unit in document.get("translationUnits") or []:
        if not isinstance(unit, dict):
            continue
        text = str(unit.get("text") or "").strip()
        translation_status = str(unit.get("translationStatus") or "").strip() or None
        translation_message = str(unit.get("translationMessage") or "").strip() or None
        source_segment_ids = _string_tuple(unit.get("sourceSegmentIds"))
        source_segment_indices = _int_tuple(unit.get("sourceSegmentIndices"))
        if not source_segment_ids and not source_segment_indices:
            continue
        if not text and translation_status is None and translation_message is None:
            continue
        units.append(
            _StableTranslationUnit(
                source_segment_ids=source_segment_ids,
                source_segment_indices=source_segment_indices,
                text=text or None,
                translation_status=translation_status,
                translation_message=translation_message,
            )
        )
    for segment in document.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        unit = _translation_unit_from_document_segment(segment)
        if unit is not None:
            units.append(unit)
    return units


def _translation_unit_from_document_segment(
    segment: dict[str, Any],
) -> _StableTranslationUnit | None:
    translation = str(segment.get("translation") or "").strip() or None
    translation_status = str(segment.get("translationStatus") or "").strip() or None
    translation_message = str(segment.get("translationMessage") or "").strip() or None
    if (
        translation is None
        and translation_status is None
        and translation_message is None
    ):
        return None
    segment_id = str(segment.get("id") or "").strip()
    segment_index = _optional_int(segment.get("index"))
    if not segment_id and segment_index is None:
        return None
    return _StableTranslationUnit(
        source_segment_ids=(segment_id,) if segment_id else (),
        source_segment_indices=(segment_index,) if segment_index is not None else (),
        text=translation,
        translation_status=translation_status,
        translation_message=translation_message,
    )


def _translation_unit_from_segment(
    segment: dict[str, Any],
) -> _StableTranslationUnit | None:
    segment_id = str(segment.get("id") or "").strip()
    segment_index = _optional_int(segment.get("index"))
    if not segment_id and segment_index is None:
        return None
    return _StableTranslationUnit(
        source_segment_ids=(segment_id,) if segment_id else (),
        source_segment_indices=(segment_index,) if segment_index is not None else (),
        text=str(segment.get("translation") or "").strip() or None,
        translation_status=str(segment.get("translation_status") or "").strip() or None,
        translation_message=str(segment.get("translation_message") or "").strip()
        or None,
    )


def _translation_unit_from_event(
    event: dict[str, Any],
    *,
    text: str | None = None,
    translation_status: str | None = None,
    translation_message: str | None = None,
) -> _StableTranslationUnit:
    source_segment_ids = _string_tuple(event.get("source_segment_ids"))
    source_segment_indices = _int_tuple(event.get("source_segment_indices"))
    anchor_id = str(event.get("source_segment_id") or "").strip()
    anchor_index = _optional_int(event.get("source_segment_index"))
    return _StableTranslationUnit(
        source_segment_ids=source_segment_ids or ((anchor_id,) if anchor_id else ()),
        source_segment_indices=source_segment_indices
        or ((anchor_index,) if anchor_index is not None and anchor_index > 0 else ()),
        text=text,
        translation_status=translation_status,
        translation_message=translation_message,
    )


def _project_stable_lines(
    source_lines: list[SubtitleLine],
    translation_units: list[_StableTranslationUnit],
    *,
    include_translation: bool,
) -> list[SubtitleLine]:
    if not include_translation:
        return list(source_lines)
    source_index = _build_source_line_index(source_lines)
    resolved_by_start: dict[int, tuple[_StableTranslationUnit, list[int]]] = {}
    for unit in translation_units:
        offsets = _resolve_translation_unit_offsets(source_index, unit)
        if offsets and offsets[0] not in resolved_by_start:
            resolved_by_start[offsets[0]] = (unit, offsets)

    lines: list[SubtitleLine] = []
    offset = 0
    while offset < len(source_lines):
        match = resolved_by_start.get(offset)
        if match is None:
            lines.append(source_lines[offset])
            offset += 1
            continue
        unit, offsets = match
        covered = [source_lines[index] for index in offsets]
        lines.append(
            _combined_source_line(
                covered,
                translation=unit.text,
                translation_status=unit.translation_status,
                translation_message=unit.translation_message,
            )
        )
        offset = max(offset + 1, max(offsets) + 1)
    return lines


def _resolve_translation_unit_offsets(
    source_index: _SourceLineIndex, unit: _StableTranslationUnit
) -> list[int] | None:
    offsets: list[int] = []
    coverage_len = _translation_coverage_len(unit)
    if coverage_len == 0:
        return None
    for coverage_index in range(coverage_len):
        offset = -1
        if coverage_index < len(unit.source_segment_ids):
            segment_id = unit.source_segment_ids[coverage_index]
            offset = source_index.offset_by_id.get(segment_id, -1)
        if offset < 0 and coverage_index < len(unit.source_segment_indices):
            segment_index = unit.source_segment_indices[coverage_index]
            offset = source_index.offset_by_index.get(segment_index, -1)
        if offset < 0:
            return None
        if offsets and offset != offsets[-1] + 1:
            return None
        offsets.append(offset)
    return offsets


def _build_source_line_index(source_lines: list[SubtitleLine]) -> _SourceLineIndex:
    offset_by_id: dict[str, int] = {}
    offset_by_index: dict[int, int] = {}
    for offset, line in enumerate(source_lines):
        if line.id:
            offset_by_id[line.id] = offset
        if line.index is not None:
            offset_by_index[line.index] = offset
    return _SourceLineIndex(
        offset_by_id=offset_by_id,
        offset_by_index=offset_by_index,
    )


def _combined_source_line(
    source_lines: list[SubtitleLine],
    *,
    translation: str | None,
    translation_status: str | None,
    translation_message: str | None,
) -> SubtitleLine:
    first = source_lines[0]
    last = source_lines[-1]
    return SubtitleLine(
        id=last.id,
        index=last.index,
        start_ms=first.start_ms,
        end_ms=last.end_ms,
        text=_join_source_texts(line.text for line in source_lines),
        language=first.language,
        source_revision=max(
            (line.source_revision for line in source_lines if line.source_revision),
            default=None,
        ),
        timing_status=_combined_timing_status(source_lines),
        translation=translation,
        translation_status=translation_status,
        translation_message=translation_message,
    )


def _join_source_texts(texts: Iterable[str]) -> str:
    joined = ""
    for raw_text in texts:
        text = str(raw_text or "").strip()
        if not text:
            continue
        if joined and _needs_ascii_separator(joined[-1], text[0]):
            joined += " "
        joined += text
    return joined.strip()


def _needs_ascii_separator(left: str, right: str) -> bool:
    return bool(
        left
        and right
        and _is_ascii_alnum(right)
        and (_is_ascii_alnum(left) or left in ",.!?;:")
    )


def _is_ascii_alnum(value: str) -> bool:
    return len(value) == 1 and value.isascii() and value.isalnum()


def _combined_timing_status(lines: list[SubtitleLine]) -> str | None:
    statuses = [line.timing_status for line in lines if line.timing_status]
    if "failed" in statuses:
        return "failed"
    if len(statuses) == len(lines) and all(status == "aligned" for status in statuses):
        return "aligned"
    return None


def _translation_unit_touches_coverage(
    unit: _StableTranslationUnit,
    segment_ids: set[str],
    segment_indices: set[int],
) -> bool:
    return any(
        segment_id in segment_ids for segment_id in unit.source_segment_ids
    ) or any(index in segment_indices for index in unit.source_segment_indices)


def _translation_unit_index(
    units: list[_StableTranslationUnit],
) -> dict[str, int]:
    index: dict[str, int] = {}
    for offset, unit in enumerate(units):
        key = _translation_coverage_key(unit)
        if key is not None and key not in index:
            index[key] = offset
    return index


def _translation_coverage_key(unit: _StableTranslationUnit) -> str | None:
    if unit.source_segment_indices:
        return "index:" + ",".join(str(index) for index in unit.source_segment_indices)
    if unit.source_segment_ids:
        return "id:" + "\x1f".join(unit.source_segment_ids)
    return None


def _translation_coverage_len(unit: _StableTranslationUnit) -> int:
    return max(len(unit.source_segment_ids), len(unit.source_segment_indices))


def _format_srt_time(ms: int) -> str:
    total_ms = max(0, int(ms))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    # Defensive: a malformed timing value from the server must not abort event replay.
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item or "").strip() for item in value if str(item or "").strip())


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    values: list[int] = []
    for item in value:
        parsed = _optional_int(item)
        if parsed is not None and parsed > 0:
            values.append(parsed)
    return tuple(values)


__all__ = ["SubtitleDocument", "SubtitleLine", "SubtitleWindow"]
