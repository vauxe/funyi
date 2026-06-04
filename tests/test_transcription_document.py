# coding=utf-8
from __future__ import annotations

from qwen3_asr_runtime.transcription_document import TranscriptDocument, TranscriptSegment


def test_transcript_document_payload_uses_stable_snapshot_schema() -> None:
    document = TranscriptDocument(
        duration_ms=1800,
        language="Chinese",
        segments=[
            TranscriptSegment(
                id="seg_000001",
                index=1,
                start_ms=0,
                end_ms=1000,
                text="你好",
                language="Chinese",
                timing_status="aligned",
                translation="hello",
            ),
            TranscriptSegment(
                id="seg_000002",
                index=2,
                start_ms=1000,
                end_ms=1800,
                text="世界",
                language="Chinese",
            ),
        ],
    )

    assert document.to_payload() == {
        "schemaVersion": 1,
        "durationMs": 1800,
        "language": "Chinese",
        "text": "你好世界",
        "segments": [
            {
                "id": "seg_000001",
                "index": 1,
                "startMs": 0,
                "endMs": 1000,
                "text": "你好",
                "language": "Chinese",
                "timingStatus": "aligned",
                "translation": "hello",
            },
            {
                "id": "seg_000002",
                "index": 2,
                "startMs": 1000,
                "endMs": 1800,
                "text": "世界",
                "language": "Chinese",
            },
        ],
    }


def test_transcript_document_text_preserves_readable_ascii_boundaries() -> None:
    document = TranscriptDocument(
        duration_ms=3000,
        language="English",
        segments=[
            TranscriptSegment(id="seg_000001", index=1, start_ms=0, end_ms=1000, text="hello", language="English"),
            TranscriptSegment(id="seg_000002", index=2, start_ms=1000, end_ms=2000, text="world.", language="English"),
            TranscriptSegment(id="seg_000003", index=3, start_ms=2000, end_ms=3000, text="Next", language="English"),
        ],
    )

    assert document.text == "hello world. Next"
