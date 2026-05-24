# coding=utf-8
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StableSegment:
    id: str
    index: int
    start_ms: int | None
    end_ms: int | None
    text: str
    language: str
    timing_status: str | None = None


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
        start_ms: int | None,
        end_ms: int | None,
        language: str = "",
        timing_status: str | None = None,
    ) -> StableSegment:
        segment_text = str(text or "")
        if not segment_text.strip():
            raise ValueError("segment text must not be empty")

        if (start_ms is None) != (end_ms is None):
            raise ValueError("segment timing must include both start_ms and end_ms, or neither")
        if start_ms is None:
            start = None
            end = None
        else:
            previous_end = self._previous_known_end()
            start = max(int(previous_end), int(start_ms))
            end = max(start, int(end_ms))
        index = self._next_segment_index
        self._next_segment_index += 1
        segment = StableSegment(
            id=f"seg_{index:06d}",
            index=index,
            start_ms=start,
            end_ms=end,
            text=segment_text,
            language=str(language or ""),
            timing_status=str(timing_status or "") or None,
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
            "stable_appends": [self._stable_segment_payload(segment) for segment in stable_appends],
            "partial": asdict(self.state.partial) if self.state.partial is not None else None,
        }

    def update_segment_timing(
        self,
        *,
        source_segment_id: str,
        start_ms: int | None,
        end_ms: int | None,
        timing_status: str,
    ) -> dict[str, object]:
        segment = self._find_stable_segment(source_segment_id)
        if segment is None:
            raise ValueError(f"unknown stable segment id: {source_segment_id}")

        status = str(timing_status or "").strip() or "failed"
        if start_ms is None or end_ms is None:
            segment.start_ms = None
            segment.end_ms = None
            segment.timing_status = status
            return self._timing_update_payload(segment)

        previous_end = self._previous_known_end(before_index=segment.index)
        start = max(int(previous_end), int(start_ms))
        end = max(start, int(end_ms))
        segment.start_ms = start
        segment.end_ms = end
        segment.timing_status = status
        return self._timing_update_payload(segment)

    def final_event(self) -> dict[str, object]:
        self.state.revision += 1
        return {
            "type": "transcript_final",
            "revision": self.state.revision,
            "stable_count": self.stable_count,
            "segments": [self._stable_segment_payload(segment) for segment in self.state.stable_segments],
        }

    def _find_stable_segment(self, source_segment_id: str) -> StableSegment | None:
        segment_id = str(source_segment_id or "")
        for segment in self.state.stable_segments:
            if segment.id == segment_id:
                return segment
        return None

    def _previous_known_end(self, *, before_index: int | None = None) -> int:
        for segment in reversed(self.state.stable_segments):
            if before_index is not None and segment.index >= before_index:
                continue
            if segment.end_ms is not None:
                return int(segment.end_ms)
        return 0

    def _stable_segment_payload(self, segment: StableSegment) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": segment.id,
            "index": int(segment.index),
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "text": segment.text,
            "language": segment.language,
        }
        if segment.timing_status is not None:
            payload["timing_status"] = segment.timing_status
        return payload

    def _timing_update_payload(self, segment: StableSegment) -> dict[str, object]:
        return {
            "type": "transcript_timing_update",
            "source_segment_id": segment.id,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "timing_status": segment.timing_status or "failed",
        }


__all__ = [
    "StableSegment",
    "PartialSegment",
    "TranscriptState",
    "TranscriptStore",
]
