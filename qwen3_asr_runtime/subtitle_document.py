# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SubtitleLine:
    text: str
    start_ms: int
    end_ms: int
    language: str = ""
    id: str | None = None
    index: int | None = None
    source_revision: int | None = None
    translation: str | None = None
    translation_status: str | None = None
    translation_message: str | None = None


@dataclass(frozen=True)
class SubtitleWindow:
    previous: SubtitleLine | None
    current: SubtitleLine | None


class SubtitleDocument:
    """Client-side replay model for realtime subtitle events.

    Source transcript events own timing and scroll state. Translation events only
    annotate matching source lines.
    """

    def __init__(self, *, translation_enabled: bool = True) -> None:
        self.translation_enabled = bool(translation_enabled)
        self.revision = 0
        self.stable_lines: list[SubtitleLine] = []
        self.current: SubtitleLine | None = None

    def set_translation_enabled(self, enabled: bool) -> None:
        self.translation_enabled = bool(enabled)

    def apply_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "transcript_update":
            self._apply_transcript_update(event)
        elif event_type == "transcript_final":
            self._apply_transcript_final(event)
        elif event_type == "translation_stable":
            self._apply_stable_translation(event)
        elif event_type == "translation_preview":
            self._apply_preview_translation(event)
        elif event_type == "translation_status":
            self._apply_translation_status(event)

    def window(self, *, include_translation: bool | None = None) -> SubtitleWindow:
        enabled = self.translation_enabled if include_translation is None else bool(include_translation)
        previous = self.stable_lines[-1] if self.stable_lines else None
        return SubtitleWindow(
            previous=_render_line(previous, include_translation=enabled),
            current=_render_line(self.current, include_translation=enabled),
        )

    def to_srt(self, *, include_translation: bool | None = None) -> str:
        enabled = self.translation_enabled if include_translation is None else bool(include_translation)
        blocks: list[str] = []
        for number, line in enumerate(self.stable_lines, start=1):
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
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _apply_transcript_update(self, event: dict[str, Any]) -> None:
        stable_base = int(event.get("stable_base") or 0)
        stable_appends = event.get("stable_appends") or []
        stable_count = int(event.get("stable_count") or 0)
        if stable_base != len(self.stable_lines):
            raise ValueError(
                f"stable cursor mismatch: stable_base={stable_base}, local_count={len(self.stable_lines)}"
            )
        if stable_base + len(stable_appends) != stable_count:
            raise ValueError(
                f"stable count mismatch: stable_base={stable_base}, "
                f"appends={len(stable_appends)}, stable_count={stable_count}"
            )

        revision = int(event.get("revision") or self.revision)
        for segment in stable_appends:
            if isinstance(segment, dict):
                self.stable_lines.append(_line_from_segment(segment, source_revision=revision))

        partial = event.get("partial")
        self.current = _line_from_segment(partial, source_revision=revision) if isinstance(partial, dict) else None
        self.revision = revision

    def _apply_transcript_final(self, event: dict[str, Any]) -> None:
        existing = {line.id: line for line in self.stable_lines if line.id}
        lines: list[SubtitleLine] = []
        for segment in event.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            line = _line_from_segment(segment, source_revision=int(event.get("revision") or self.revision))
            previous = existing.get(line.id or "")
            if previous is not None:
                line = replace(
                    line,
                    translation=previous.translation,
                    translation_status=previous.translation_status,
                    translation_message=previous.translation_message,
                )
            lines.append(line)
        self.stable_lines = lines
        self.current = None
        self.revision = int(event.get("revision") or self.revision)

    def _apply_stable_translation(self, event: dict[str, Any]) -> None:
        index = self._stable_index(event)
        if index is None:
            return
        text = str(event.get("text") or "").strip()
        if not text:
            return
        self.stable_lines[index] = replace(
            self.stable_lines[index],
            translation=text,
            translation_status=None,
            translation_message=None,
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
        index = self._stable_index(event)
        if index is None:
            return
        self.stable_lines[index] = replace(
            self.stable_lines[index],
            translation_status=str(event.get("code") or ""),
            translation_message=str(event.get("message") or ""),
        )

    def _stable_index(self, event: dict[str, Any]) -> int | None:
        segment_id = str(event.get("source_segment_id") or "")
        if segment_id:
            for index, line in enumerate(self.stable_lines):
                if line.id == segment_id:
                    return index
        try:
            segment_index = int(event.get("source_segment_index") or 0)
        except (TypeError, ValueError):
            return None
        if segment_index <= 0:
            return None
        for index, line in enumerate(self.stable_lines):
            if line.index == segment_index:
                return index
        return None


def _line_from_segment(segment: dict[str, Any], *, source_revision: int) -> SubtitleLine:
    text = str(segment.get("text") or "").strip()
    return SubtitleLine(
        id=str(segment.get("id") or "") or None,
        index=_optional_int(segment.get("index")),
        start_ms=int(segment.get("start_ms") or 0),
        end_ms=int(segment.get("end_ms") or 0),
        text=text,
        language=str(segment.get("language") or ""),
        source_revision=int(source_revision),
    )


def _render_line(line: SubtitleLine | None, *, include_translation: bool) -> SubtitleLine | None:
    if line is None or include_translation:
        return line
    return replace(line, translation=None, translation_status=None, translation_message=None)


def _format_srt_time(ms: int) -> str:
    total_ms = max(0, int(ms))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


__all__ = ["SubtitleDocument", "SubtitleLine", "SubtitleWindow"]
