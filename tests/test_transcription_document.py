# coding=utf-8
from __future__ import annotations

from qwen3_asr_runtime.transcription_document import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptTranslationUnit,
)


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
                translation_status="ok",
                translation_message="translated",
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
                "translationStatus": "ok",
                "translationMessage": "translated",
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
            TranscriptSegment(
                id="seg_000001",
                index=1,
                start_ms=0,
                end_ms=1000,
                text="hello",
                language="English",
            ),
            TranscriptSegment(
                id="seg_000002",
                index=2,
                start_ms=1000,
                end_ms=2000,
                text="world.",
                language="English",
            ),
            TranscriptSegment(
                id="seg_000003",
                index=3,
                start_ms=2000,
                end_ms=3000,
                text="Next",
                language="English",
            ),
        ],
    )

    assert document.text == "hello world. Next"


def test_transcript_document_payload_can_express_grouped_translation_units() -> None:
    document = TranscriptDocument(
        duration_ms=1800,
        language="Chinese",
        segments=[
            TranscriptSegment(
                id="seg_000001",
                index=1,
                start_ms=0,
                end_ms=1000,
                text="今天讨论字幕显示问题，",
            ),
            TranscriptSegment(
                id="seg_000002",
                index=2,
                start_ms=1000,
                end_ms=1800,
                text="并且保持翻译输入完整。",
                translation="We discuss subtitle display while preserving translation context.",
            ),
        ],
        translation_units=[
            TranscriptTranslationUnit(
                text="We discuss subtitle display while preserving translation context.",
                target_language="English",
                source_segment_ids=("seg_000001", "seg_000002"),
                source_segment_indices=(1, 2),
            )
        ],
    )

    assert document.to_payload()["translationUnits"] == [
        {
            "text": "We discuss subtitle display while preserving translation context.",
            "targetLanguage": "English",
            "sourceSegmentIds": ["seg_000001", "seg_000002"],
            "sourceSegmentIndices": [1, 2],
        }
    ]


def test_transcript_document_translation_unit_payload_includes_status() -> None:
    unit = TranscriptTranslationUnit(
        text="",
        target_language="English",
        source_segment_ids=("seg_000001", "seg_000002"),
        source_segment_indices=(1, 2),
        translation_status="timeout",
        translation_message="translation failed",
    )

    assert unit.to_payload() == {
        "text": "",
        "targetLanguage": "English",
        "sourceSegmentIds": ["seg_000001", "seg_000002"],
        "sourceSegmentIndices": [1, 2],
        "translationStatus": "timeout",
        "translationMessage": "translation failed",
    }
