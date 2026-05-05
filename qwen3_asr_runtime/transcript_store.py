# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    id: str
    index: int
    start_ms: int
    end_ms: int
    text: str
    language: str


@dataclass
class TranscriptDocument:
    id: str
    segments: list[TranscriptSegment] = field(default_factory=list)


class TranscriptStore:
    """Authoritative in-memory transcript history."""

    def __init__(self, transcript_id: str = "default") -> None:
        self.document = TranscriptDocument(id=str(transcript_id))
        self._next_segment_index = 1

    @property
    def segments(self) -> list[TranscriptSegment]:
        return list(self.document.segments)

    def append_segment(
        self,
        *,
        text: str,
        start_ms: int,
        end_ms: int,
        language: str = "",
    ) -> TranscriptSegment:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("segment text must not be empty")

        previous_end = self.document.segments[-1].end_ms if self.document.segments else 0
        start = max(int(previous_end), int(start_ms))
        end = max(start, int(end_ms))
        index = self._next_segment_index
        self._next_segment_index += 1
        segment = TranscriptSegment(
            id=f"seg_{index:06d}",
            index=index,
            start_ms=start,
            end_ms=end,
            text=normalized_text,
            language=str(language or ""),
        )
        self.document.segments.append(segment)
        return segment


__all__ = [
    "TranscriptSegment",
    "TranscriptDocument",
    "TranscriptStore",
]
