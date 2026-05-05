# coding=utf-8
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StableSegment:
    id: str
    index: int
    start_ms: int
    end_ms: int
    text: str
    language: str


@dataclass
class PartialSegment:
    start_ms: int
    end_ms: int
    text: str
    language: str


@dataclass
class TranscriptState:
    id: str
    revision: int = 0
    stable_count: int = 0
    stable_segments: list[StableSegment] = field(default_factory=list)
    partial: PartialSegment | None = None


class TranscriptStore:
    """Authoritative in-memory transcript state.

    The stable prefix is append-only. The partial tail is a replace-only
    snapshot of the newest ASR text that may still be rewritten.
    """

    def __init__(self, transcript_id: str = "default") -> None:
        self.state = TranscriptState(id=str(transcript_id))
        self._next_segment_index = 1

    @property
    def revision(self) -> int:
        return int(self.state.revision)

    @property
    def stable_count(self) -> int:
        return int(self.state.stable_count)

    @property
    def stable_segments(self) -> list[StableSegment]:
        return list(self.state.stable_segments)

    @property
    def partial(self) -> PartialSegment | None:
        return self.state.partial

    def append_stable_segment(
        self,
        *,
        text: str,
        start_ms: int,
        end_ms: int,
        language: str = "",
    ) -> StableSegment:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("segment text must not be empty")

        previous_end = self.state.stable_segments[-1].end_ms if self.state.stable_segments else 0
        start = max(int(previous_end), int(start_ms))
        end = max(start, int(end_ms))
        index = self._next_segment_index
        self._next_segment_index += 1
        segment = StableSegment(
            id=f"seg_{index:06d}",
            index=index,
            start_ms=start,
            end_ms=end,
            text=normalized_text,
            language=str(language or ""),
        )
        self.state.stable_segments.append(segment)
        self.state.stable_count = len(self.state.stable_segments)
        return segment

    def replace_partial(self, segment: PartialSegment | None) -> bool:
        normalized = segment if segment is not None and str(segment.text or "").strip() else None
        if normalized == self.state.partial:
            return False
        self.state.partial = normalized
        return True

    def clear_partial(self) -> bool:
        return self.replace_partial(None)

    def update_event(
        self,
        *,
        stable_base: int,
        stable_appends: list[StableSegment],
    ) -> dict[str, object]:
        expected_count = int(stable_base) + len(stable_appends)
        if expected_count != self.stable_count:
            raise ValueError(
                f"stable cursor mismatch: stable_base={stable_base}, "
                f"appends={len(stable_appends)}, stable_count={self.stable_count}"
            )
        self.state.revision += 1
        return {
            "type": "transcript_update",
            "revision": self.state.revision,
            "stable_base": int(stable_base),
            "stable_count": self.stable_count,
            "stable_appends": [asdict(segment) for segment in stable_appends],
            "partial": asdict(self.state.partial) if self.state.partial is not None else None,
        }

    def final_event(self) -> dict[str, object]:
        self.state.revision += 1
        return {
            "type": "transcript_final",
            "revision": self.state.revision,
            "stable_count": self.stable_count,
            "segments": [asdict(segment) for segment in self.state.stable_segments],
        }


__all__ = [
    "StableSegment",
    "PartialSegment",
    "TranscriptState",
    "TranscriptStore",
]
