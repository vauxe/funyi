# coding=utf-8
from __future__ import annotations

import pytest

from qwen3_asr_runtime.subtitle_document import SubtitleDocument


def stable_segment(
    index: int,
    text: str,
    *,
    start_ms: int | None,
    end_ms: int | None,
    timing_status: str | None = None,
) -> dict[str, object]:
    segment = {
        "id": f"seg_{index:06d}",
        "index": index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "language": "Chinese",
    }
    if timing_status is not None:
        segment["timing_status"] = timing_status
    return segment


def partial_segment(text: str, *, start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "language": "Chinese",
    }


class TestSubtitleDocument:
    def test_window_scrolls_when_current_partial_becomes_stable(self) -> None:
        document = SubtitleDocument()

        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 0,
                "stable_appends": [],
                "partial": partial_segment("正在处理", start_ms=0, end_ms=1000),
            }
        )
        window = document.window()
        assert window.previous is None
        assert window.current.text == '正在处理'  # type: ignore[union-attr]

        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 2,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "正在处理", start_ms=0, end_ms=1000)],
                "partial": partial_segment("下一句", start_ms=1000, end_ms=1600),
            }
        )

        window = document.window()
        assert window.previous.text == '正在处理'  # type: ignore[union-attr]
        assert window.current.text == '下一句'  # type: ignore[union-attr]

    def test_malformed_timing_value_does_not_break_replay(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1200)],
                "partial": None,
            }
        )

        # A malformed timing value from the server must not raise out of replay.
        document.apply_event(
            {
                "type": "transcript_timing_update",
                "source_segment_id": "seg_000001",
                "start_ms": "oops",
                "end_ms": None,
                "timing_status": "failed",
            }
        )

        assert document.stable_lines[0].start_ms is None
        assert document.stable_lines[0].timing_status == "failed"

    def test_srt_uses_stable_history_only(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1200)],
                "partial": partial_segment("不应进入详情", start_ms=1200, end_ms=1800),
            }
        )

        assert document.to_srt() == '1\n00:00:00,000 --> 00:00:01,200\n第一句\n'

    def test_window_returns_complete_latest_lines(self) -> None:
        document = SubtitleDocument()
        stable_text = "一二三四五六七八九十甲乙丙丁戊己庚辛。后续文本"
        partial_text = "当前文本也可能很长。最后显示"
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, stable_text, start_ms=0, end_ms=2300)],
                "partial": partial_segment(partial_text, start_ms=2300, end_ms=3300),
            }
        )

        assert len(document.stable_lines) == 1
        assert document.stable_lines[0].text == stable_text
        window = document.window()
        assert window.previous.text == stable_text  # type: ignore[union-attr]
        assert window.current.text == partial_text  # type: ignore[union-attr]

    def test_timing_update_patches_pending_stable_line(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(
                        1,
                        "第一句",
                        start_ms=None,
                        end_ms=None,
                        timing_status="pending",
                    )
                ],
                "partial": None,
            }
        )

        window = document.window()
        assert window.previous.start_ms is None  # type: ignore[union-attr]
        assert window.previous.timing_status == 'pending'  # type: ignore[union-attr]
        assert document.to_srt() == ''

        document.apply_event(
            {
                "type": "transcript_timing_update",
                "source_segment_id": "seg_000001",
                "start_ms": 120,
                "end_ms": 860,
                "timing_status": "aligned",
            }
        )

        assert document.stable_lines[0].start_ms == 120
        assert document.stable_lines[0].end_ms == 860
        assert document.stable_lines[0].timing_status == 'aligned'
        assert document.to_srt() == '1\n00:00:00,120 --> 00:00:00,860\n第一句\n'

    def test_translation_events_are_annotations_not_scroll_inputs(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "稳定行", start_ms=0, end_ms=1000)],
                "partial": partial_segment("当前行", start_ms=1000, end_ms=1800),
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "stable line",
            }
        )
        document.apply_event(
            {
                "type": "translation_preview",
                "source_revision": 0,
                "target_language": "English",
                "text": "stale preview",
            }
        )
        document.apply_event(
            {
                "type": "translation_preview",
                "source_revision": 1,
                "target_language": "English",
                "text": "current line",
            }
        )

        window = document.window()
        assert window.previous.text == '稳定行'  # type: ignore[union-attr]
        assert window.previous.translation == 'stable line'  # type: ignore[union-attr]
        assert window.current.text == '当前行'  # type: ignore[union-attr]
        assert window.current.translation == 'current line'  # type: ignore[union-attr]
        assert document.to_srt() == '1\n00:00:00,000 --> 00:00:01,000\n稳定行\nstable line\n'

    def test_translation_status_clears_existing_translation(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "稳定行", start_ms=0, end_ms=1000)],
                "partial": None,
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "stable line",
            }
        )
        document.apply_event(
            {
                "type": "translation_status",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "code": "failed",
                "message": "translation failed",
            }
        )

        window = document.window()
        assert window.previous.translation is None  # type: ignore[union-attr]
        assert window.previous.translation_status == "failed"  # type: ignore[union-attr]
        assert window.previous.translation_message == "translation failed"  # type: ignore[union-attr]

    def test_grouped_translation_anchors_without_folding_source_segments(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 2,
                "stable_appends": [
                    stable_segment(1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000),
                    stable_segment(2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800),
                ],
                "partial": None,
            }
        )

        document.apply_event(
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000002",
                "source_segment_index": 2,
                "source_segment_ids": ["seg_000001", "seg_000002"],
                "source_segment_indices": [1, 2],
                "target_language": "English",
                "text": "We discuss subtitle display while preserving translation context.",
            }
        )

        assert document.stable_lines[1].translation == "We discuss subtitle display while preserving translation context."

    def test_translation_can_be_hidden_without_changing_source_state(self) -> None:
        document = SubtitleDocument(translation_enabled=False)
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "稳定行", start_ms=0, end_ms=1000)],
                "partial": partial_segment("当前行", start_ms=1000, end_ms=1800),
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "stable line",
            }
        )

        window = document.window()
        assert window.previous.text == '稳定行'  # type: ignore[union-attr]
        assert window.previous.translation is None  # type: ignore[union-attr]
        assert document.to_srt() == '1\n00:00:00,000 --> 00:00:01,000\n稳定行\n'

    def test_rejects_stale_stable_base(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
                "partial": None,
            }
        )

        with pytest.raises(ValueError):
            document.apply_event(
                {
                    "type": "transcript_update",
                    "revision": 2,
                    "stable_base": 0,
                    "stable_count": 2,
                    "stable_appends": [stable_segment(2, "第二句", start_ms=1000, end_ms=2000)],
                    "partial": None,
                }
            )

    def test_final_snapshot_clears_current_and_keeps_stable_translations(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
                "partial": partial_segment("临时", start_ms=1000, end_ms=1400),
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "first line",
            }
        )

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 2,
                "stable_count": 1,
                "segments": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
            }
        )

        window = document.window()
        assert window.previous.text == '第一句'  # type: ignore[union-attr]
        assert window.previous.translation == 'first line'  # type: ignore[union-attr]
        assert window.current is None

    def test_final_snapshot_preserves_stable_translation_by_index_when_ids_change(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
                "partial": None,
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "first line",
            }
        )
        final_segment = stable_segment(1, "第一句", start_ms=0, end_ms=1000)
        final_segment["id"] = "rebuilt_seg_1"

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 2,
                "stable_count": 1,
                "segments": [final_segment],
            }
        )

        window = document.window()
        assert window.previous.translation == "first line"  # type: ignore[union-attr]
        assert window.current is None

    def test_final_snapshot_prefers_segment_translation_over_replayed_state(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
                "partial": None,
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "old translation",
            }
        )
        final_segment = stable_segment(1, "第一句", start_ms=0, end_ms=1000)
        final_segment["translation"] = "final translation"

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 2,
                "stable_count": 1,
                "segments": [final_segment],
            }
        )

        window = document.window()
        assert window.previous.translation == "final translation"  # type: ignore[union-attr]

    def test_final_snapshot_anchors_document_translation_units(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 1,
                "stable_count": 2,
                "segments": [
                    stable_segment(1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000),
                    stable_segment(2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800),
                ],
                "document": {
                    "translationUnits": [
                        {
                            "text": "We discuss subtitle display while preserving translation context.",
                            "targetLanguage": "English",
                            "sourceSegmentIds": ["seg_000001", "seg_000002"],
                            "sourceSegmentIndices": [1, 2],
                        }
                    ]
                },
            }
        )

        assert document.stable_lines[1].translation == "We discuss subtitle display while preserving translation context."

    def test_final_snapshot_anchors_translation_unit_with_paired_index_fallback(self) -> None:
        document = SubtitleDocument()
        second_segment = stable_segment(2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800)
        second_segment["id"] = "rebuilt_seg_2"

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 1,
                "stable_count": 2,
                "segments": [
                    stable_segment(1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000),
                    second_segment,
                ],
                "document": {
                    "translationUnits": [
                        {
                            "text": "We discuss subtitle display while preserving translation context.",
                            "targetLanguage": "English",
                            "sourceSegmentIds": ["seg_000001", "old_seg_000002"],
                            "sourceSegmentIndices": [1, 2],
                        }
                    ]
                },
            }
        )

        assert document.stable_lines[0].translation is None
        assert document.stable_lines[1].translation == "We discuss subtitle display while preserving translation context."

    def test_final_snapshot_prefers_segment_translation_status_over_replayed_state(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
                "partial": None,
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_revision": 1,
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "target_language": "English",
                "text": "old translation",
            }
        )
        final_segment = stable_segment(1, "第一句", start_ms=0, end_ms=1000)
        final_segment["translation_status"] = "timeout"
        final_segment["translation_message"] = "translation failed"

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 2,
                "stable_count": 1,
                "segments": [final_segment],
            }
        )

        window = document.window()
        assert window.previous.translation is None  # type: ignore[union-attr]
        assert window.previous.translation_status == "timeout"  # type: ignore[union-attr]
        assert window.previous.translation_message == "translation failed"  # type: ignore[union-attr]

    def test_final_marker_without_snapshot_clears_current_and_keeps_replayed_stable_history(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "第一句", start_ms=0, end_ms=1000)],
                "partial": partial_segment("临时", start_ms=1000, end_ms=1400),
            }
        )

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 2,
                "final_revision": 2,
                "stable_count": 1,
            }
        )

        window = document.window()
        assert window.previous.text == "第一句"  # type: ignore[union-attr]
        assert window.current is None
