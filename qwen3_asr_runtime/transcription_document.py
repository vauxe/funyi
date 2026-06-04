# coding=utf-8
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    index: int
    start_ms: int | None
    end_ms: int | None
    text: str
    language: str = ""
    timing_status: str | None = None
    translation: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "index": int(self.index),
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "text": self.text,
            "language": self.language,
        }
        if self.timing_status is not None:
            payload["timingStatus"] = self.timing_status
        if self.translation is not None:
            payload["translation"] = self.translation
        return payload


@dataclass(frozen=True)
class TranscriptDocument:
    duration_ms: int
    language: str
    segments: list[TranscriptSegment]
    schema_version: int = 1

    @property
    def text(self) -> str:
        return _join_segment_texts(segment.text for segment in self.segments)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": int(self.schema_version),
            "durationMs": int(self.duration_ms),
            "language": self.language,
            "text": self.text,
            "segments": [segment.to_payload() for segment in self.segments],
        }


def _join_segment_texts(texts: Iterable[str]) -> str:
    joined = ""
    for raw_text in texts:
        text = str(raw_text or "").strip()
        if not text:
            continue
        if not joined:
            joined = text
            continue
        if _needs_text_separator(joined[-1], text[0]):
            joined += " "
        joined += text
    return joined.strip()


def _needs_text_separator(left: str, right: str) -> bool:
    if not left or not right or not _is_ascii_alnum(right):
        return False
    return _is_ascii_alnum(left) or left in ",.!?;:"


def _is_ascii_alnum(value: str) -> bool:
    return len(value) == 1 and value.isascii() and value.isalnum()


__all__ = ["TranscriptDocument", "TranscriptSegment"]
