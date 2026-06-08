# coding=utf-8
from __future__ import annotations

from typing import Any

import pytest

from qwen3_asr_runtime import subtitle_document as subtitle_document_module
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
        assert window.current.text == "正在处理"  # type: ignore[union-attr]

        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 2,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "正在处理", start_ms=0, end_ms=1000)
                ],
                "partial": partial_segment("下一句", start_ms=1000, end_ms=1600),
            }
        )

        window = document.window()
        assert window.previous.text == "正在处理"  # type: ignore[union-attr]
        assert window.current.text == "下一句"  # type: ignore[union-attr]

    def test_malformed_timing_value_does_not_break_replay(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1200)
                ],
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
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1200)
                ],
                "partial": partial_segment("不应进入详情", start_ms=1200, end_ms=1800),
            }
        )

        assert document.to_srt() == "1\n00:00:00,000 --> 00:00:01,200\n第一句\n"

    def test_preview_composes_pending_stable_prefix_without_hiding_history(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "今天我们来讲", start_ms=0, end_ms=1000)
                ],
                "partial": partial_segment("一下这个", start_ms=1000, end_ms=1800),
            }
        )

        document.apply_event(
            {
                "type": "translation_preview",
                "source_revision": 1,
                "text": "Today we discuss this",
            }
        )

        window = document.window()
        assert window.previous.text == "今天我们来讲"  # type: ignore[union-attr]
        assert window.previous.translation is None  # type: ignore[union-attr]
        assert window.current.text == "今天我们来讲一下这个"  # type: ignore[union-attr]
        assert window.current.translation == "Today we discuss this"  # type: ignore[union-attr]
        assert [line.text for line in document.stable_lines] == ["今天我们来讲"]
        assert document.to_srt() == "1\n00:00:00,000 --> 00:00:01,000\n今天我们来讲\n"

    def test_stable_translation_clears_pending_preview_prefix(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "今天我们来讲", start_ms=0, end_ms=1000)
                ],
                "partial": partial_segment("一下这个", start_ms=1000, end_ms=1800),
            }
        )
        document.apply_event(
            {
                "type": "translation_preview",
                "source_revision": 1,
                "text": "Today we discuss this",
            }
        )

        document.apply_event(
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "source_segment_ids": ["seg_000001"],
                "source_segment_indices": [1],
                "text": "Today we discuss",
            }
        )

        window = document.window()
        assert window.previous.text == "今天我们来讲"  # type: ignore[union-attr]
        assert window.previous.translation == "Today we discuss"  # type: ignore[union-attr]
        assert window.current.text == "一下这个"  # type: ignore[union-attr]
        assert window.current.translation is None  # type: ignore[union-attr]

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
                "stable_appends": [
                    stable_segment(1, stable_text, start_ms=0, end_ms=2300)
                ],
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
        assert window.previous.timing_status == "pending"  # type: ignore[union-attr]
        assert document.to_srt() == ""

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
        assert document.stable_lines[0].timing_status == "aligned"
        assert document.to_srt() == "1\n00:00:00,120 --> 00:00:00,860\n第一句\n"

    def test_translation_events_are_annotations_not_scroll_inputs(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "稳定行", start_ms=0, end_ms=1000)
                ],
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
        assert window.previous.text == "稳定行"  # type: ignore[union-attr]
        assert window.previous.translation == "stable line"  # type: ignore[union-attr]
        assert window.current.text == "当前行"  # type: ignore[union-attr]
        assert window.current.translation == "current line"  # type: ignore[union-attr]
        assert (
            document.to_srt()
            == "1\n00:00:00,000 --> 00:00:01,000\n稳定行\nstable line\n"
        )

    def test_translation_status_preserves_existing_translation(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "稳定行", start_ms=0, end_ms=1000)
                ],
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
        assert window.previous.translation == "stable line"  # type: ignore[union-attr]
        assert window.previous.translation_status == "failed"  # type: ignore[union-attr]
        assert window.previous.translation_message == "translation failed"  # type: ignore[union-attr]

    def test_grouped_translation_folds_covered_source_segments(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 2,
                "stable_appends": [
                    stable_segment(
                        1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000
                    ),
                    stable_segment(
                        2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800
                    ),
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

        assert len(document.stable_lines) == 1
        assert (
            document.stable_lines[0].text
            == "今天讨论字幕显示问题，并且保持翻译输入完整。"
        )
        assert (
            document.stable_lines[0].translation
            == "We discuss subtitle display while preserving translation context."
        )
        assert document.stable_lines[0].start_ms == 0
        assert document.stable_lines[0].end_ms == 3800
        assert document.to_srt() == (
            "1\n"
            "00:00:00,000 --> 00:00:03,800\n"
            "今天讨论字幕显示问题，并且保持翻译输入完整。\n"
            "We discuss subtitle display while preserving translation context.\n"
        )

    def test_grouped_translation_ignores_non_contiguous_coverage(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 3,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000),
                    stable_segment(2, "中间句", start_ms=1000, end_ms=2000),
                    stable_segment(3, "第三句", start_ms=2000, end_ms=3000),
                ],
                "partial": None,
            }
        )

        document.apply_event(
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000003",
                "source_segment_index": 3,
                "source_segment_ids": ["seg_000001", "seg_000003"],
                "source_segment_indices": [1, 3],
                "target_language": "English",
                "text": "first and third",
            }
        )

        assert [line.text for line in document.stable_lines] == [
            "第一句",
            "中间句",
            "第三句",
        ]
        assert [line.translation for line in document.stable_lines] == [
            None,
            None,
            None,
        ]
        assert document.to_srt() == (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "第一句\n\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "中间句\n\n"
            "3\n"
            "00:00:02,000 --> 00:00:03,000\n"
            "第三句\n"
        )

    def test_grouped_translation_stays_untimed_when_covered_tail_is_pending(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 2,
                "stable_appends": [
                    stable_segment(
                        1,
                        "第一句",
                        start_ms=0,
                        end_ms=1000,
                        timing_status="aligned",
                    ),
                    stable_segment(
                        2,
                        "第二句",
                        start_ms=None,
                        end_ms=None,
                        timing_status="pending",
                    ),
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
                "text": "first second",
            }
        )

        assert len(document.stable_lines) == 1
        assert document.stable_lines[0].text == "第一句第二句"
        assert document.stable_lines[0].start_ms == 0
        assert document.stable_lines[0].end_ms is None
        assert document.stable_lines[0].timing_status is None
        assert document.to_srt() == ""

    def test_grouped_translation_separates_ascii_sentence_boundaries(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 2,
                "stable_appends": [
                    stable_segment(1, "Hello.", start_ms=0, end_ms=1000),
                    stable_segment(2, "World", start_ms=1000, end_ms=2000),
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
                "target_language": "French",
                "text": "Bonjour le monde",
            }
        )

        assert document.stable_lines[0].text == "Hello. World"
        assert document.to_srt() == (
            "1\n00:00:00,000 --> 00:00:02,000\nHello. World\nBonjour le monde\n"
        )

    def test_grouped_translation_ignores_partially_unresolved_coverage(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 2,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000),
                    stable_segment(2, "第二句", start_ms=1000, end_ms=2000),
                ],
                "partial": None,
            }
        )

        document.apply_event(
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000002",
                "source_segment_index": 2,
                "source_segment_ids": ["seg_000001", "seg_missing", "seg_000002"],
                "source_segment_indices": [1, 99, 2],
                "target_language": "English",
                "text": "bad coverage",
            }
        )

        assert [line.text for line in document.stable_lines] == ["第一句", "第二句"]
        assert [line.translation for line in document.stable_lines] == [None, None]

    def test_grouped_translation_status_preserves_existing_translation(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 2,
                "stable_appends": [
                    stable_segment(
                        1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000
                    ),
                    stable_segment(
                        2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800
                    ),
                ],
                "partial": None,
            }
        )
        coverage = {
            "source_segment_id": "seg_000002",
            "source_segment_index": 2,
            "source_segment_ids": ["seg_000001", "seg_000002"],
            "source_segment_indices": [1, 2],
            "target_language": "English",
        }
        document.apply_event(
            {
                "type": "translation_stable",
                **coverage,
                "text": "We discuss subtitle display while preserving translation context.",
            }
        )

        document.apply_event(
            {
                "type": "translation_status",
                **coverage,
                "code": "failed",
                "message": "translation failed",
            }
        )

        assert len(document.stable_lines) == 1
        assert (
            document.stable_lines[0].translation
            == "We discuss subtitle display while preserving translation context."
        )
        assert document.stable_lines[0].translation_status == "failed"
        assert document.stable_lines[0].translation_message == "translation failed"

    def test_translation_can_be_hidden_without_changing_source_state(self) -> None:
        document = SubtitleDocument(translation_enabled=False)
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "稳定行", start_ms=0, end_ms=1000)
                ],
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
        assert window.previous.text == "稳定行"  # type: ignore[union-attr]
        assert window.previous.translation is None  # type: ignore[union-attr]
        assert document.to_srt() == "1\n00:00:00,000 --> 00:00:01,000\n稳定行\n"

    def test_rejects_stale_stable_base(self) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000)
                ],
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
                    "stable_appends": [
                        stable_segment(2, "第二句", start_ms=1000, end_ms=2000)
                    ],
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
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000)
                ],
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
        assert window.previous.text == "第一句"  # type: ignore[union-attr]
        assert window.previous.translation == "first line"  # type: ignore[union-attr]
        assert window.current is None

    def test_final_snapshot_preserves_stable_translation_by_index_when_ids_change(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000)
                ],
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

    def test_final_snapshot_prefers_segment_translation_over_replayed_state(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000)
                ],
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
                    stable_segment(
                        1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000
                    ),
                    stable_segment(
                        2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800
                    ),
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

        assert len(document.stable_lines) == 1
        assert (
            document.stable_lines[0].text
            == "今天讨论字幕显示问题，并且保持翻译输入完整。"
        )
        assert (
            document.stable_lines[0].translation
            == "We discuss subtitle display while preserving translation context."
        )

    def test_final_snapshot_anchors_translation_unit_with_paired_index_fallback(
        self,
    ) -> None:
        document = SubtitleDocument()
        second_segment = stable_segment(
            2, "并且保持翻译输入完整。", start_ms=2000, end_ms=3800
        )
        second_segment["id"] = "rebuilt_seg_2"

        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 1,
                "stable_count": 2,
                "segments": [
                    stable_segment(
                        1, "今天讨论字幕显示问题，", start_ms=0, end_ms=2000
                    ),
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

        assert len(document.stable_lines) == 1
        assert (
            document.stable_lines[0].text
            == "今天讨论字幕显示问题，并且保持翻译输入完整。"
        )
        assert (
            document.stable_lines[0].translation
            == "We discuss subtitle display while preserving translation context."
        )

    def test_final_snapshot_preserves_grouped_translation_status_from_document_unit(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 1,
                "stable_count": 2,
                "segments": [
                    stable_segment(1, "one ", start_ms=0, end_ms=1000),
                    stable_segment(2, "two", start_ms=1000, end_ms=2000),
                ],
                "document": {
                    "translationUnits": [
                        {
                            "text": "",
                            "targetLanguage": "English",
                            "sourceSegmentIds": ["seg_000001", "seg_000002"],
                            "sourceSegmentIndices": [1, 2],
                            "translationStatus": "timeout",
                            "translationMessage": "translation failed",
                        }
                    ]
                },
            }
        )

        assert len(document.stable_lines) == 1
        assert document.stable_lines[0].text == "one two"
        assert document.stable_lines[0].translation is None
        assert document.stable_lines[0].translation_status == "timeout"
        assert document.stable_lines[0].translation_message == "translation failed"

    def test_long_single_segment_translation_projection_is_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        segment_count = 5000
        document = SubtitleDocument()
        projection_call_count = 0
        original_project = subtitle_document_module._project_stable_lines

        def counted_project(*args: Any, **kwargs: Any) -> list[Any]:
            nonlocal projection_call_count
            projection_call_count += 1
            return original_project(*args, **kwargs)

        monkeypatch.setattr(
            subtitle_document_module, "_project_stable_lines", counted_project
        )
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": segment_count,
                "stable_appends": [
                    stable_segment(
                        index,
                        f"s{index}",
                        start_ms=index - 1,
                        end_ms=index,
                    )
                    for index in range(1, segment_count + 1)
                ],
                "partial": None,
            }
        )
        for index in range(1, segment_count + 1):
            document.apply_event(
                {
                    "type": "translation_stable",
                    "source_segment_id": f"seg_{index:06d}",
                    "source_segment_index": index,
                    "source_segment_ids": [f"seg_{index:06d}"],
                    "source_segment_indices": [index],
                    "text": f"t{index}",
                }
            )

        lines = document.stable_lines
        again = document.stable_lines

        assert len(lines) == segment_count
        assert lines[-1].translation == f"t{segment_count}"
        assert again[-1].translation == f"t{segment_count}"
        assert projection_call_count == 1

    def test_long_single_segment_translation_ingest_builds_one_projection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        segment_count = 5000
        document = SubtitleDocument()
        projection_call_count = 0
        original_project = subtitle_document_module._project_stable_lines

        def counted_project(*args: Any, **kwargs: Any) -> list[Any]:
            nonlocal projection_call_count
            projection_call_count += 1
            return original_project(*args, **kwargs)

        monkeypatch.setattr(
            subtitle_document_module, "_project_stable_lines", counted_project
        )
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": segment_count,
                "stable_appends": [
                    stable_segment(
                        index,
                        f"s{index}",
                        start_ms=index - 1,
                        end_ms=index,
                    )
                    for index in range(1, segment_count + 1)
                ],
                "partial": None,
            }
        )

        for index in range(1, segment_count + 1):
            document.apply_event(
                {
                    "type": "translation_stable",
                    "source_segment_id": f"seg_{index:06d}",
                    "source_segment_index": index,
                    "source_segment_ids": [f"seg_{index:06d}"],
                    "source_segment_indices": [index],
                    "text": f"t{index}",
                }
            )
        lines = document.stable_lines

        assert len(lines) == segment_count
        assert lines[-1].translation == f"t{segment_count}"
        assert projection_call_count == 1

    def test_stable_translation_status_updates_rebuilt_coverage_by_index_identity(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [stable_segment(1, "one", start_ms=0, end_ms=1000)],
                "partial": None,
            }
        )
        document.apply_event(
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "source_segment_ids": ["seg_000001"],
                "source_segment_indices": [1],
                "text": "old translation",
            }
        )
        rebuilt_segment = stable_segment(1, "one", start_ms=0, end_ms=1000)
        rebuilt_segment["id"] = "rebuilt_seg_1"
        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 2,
                "stable_count": 1,
                "segments": [rebuilt_segment],
            }
        )

        document.apply_event(
            {
                "type": "translation_status",
                "scope": "stable",
                "code": "timeout",
                "message": "translation failed",
                "source_segment_id": "rebuilt_seg_1",
                "source_segment_index": 1,
                "source_segment_ids": ["rebuilt_seg_1"],
                "source_segment_indices": [1],
            }
        )

        assert len(document.stable_lines) == 1
        assert document.stable_lines[0].translation == "old translation"
        assert document.stable_lines[0].translation_status == "timeout"
        assert document.stable_lines[0].translation_message == "translation failed"

    def test_stable_translation_status_preserves_segment_owned_translation(
        self,
    ) -> None:
        document = SubtitleDocument()
        segment = stable_segment(1, "one", start_ms=0, end_ms=1000)
        segment["translation"] = "final translation"
        document.apply_event(
            {
                "type": "transcript_final",
                "revision": 1,
                "stable_count": 1,
                "segments": [segment],
            }
        )

        document.apply_event(
            {
                "type": "translation_status",
                "scope": "stable",
                "code": "timeout",
                "message": "translation failed",
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "source_segment_ids": ["seg_000001"],
                "source_segment_indices": [1],
            }
        )

        assert len(document.stable_lines) == 1
        assert document.stable_lines[0].translation == "final translation"
        assert document.stable_lines[0].translation_status == "timeout"
        assert document.stable_lines[0].translation_message == "translation failed"

    def test_long_single_segment_translation_render_after_each_event_patches_projection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        segment_count = 3000
        document = SubtitleDocument()
        projection_call_count = 0
        original_project = subtitle_document_module._project_stable_lines

        def counted_project(*args: Any, **kwargs: Any) -> list[Any]:
            nonlocal projection_call_count
            projection_call_count += 1
            return original_project(*args, **kwargs)

        monkeypatch.setattr(
            subtitle_document_module, "_project_stable_lines", counted_project
        )
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": segment_count,
                "stable_appends": [
                    stable_segment(
                        index,
                        f"s{index}",
                        start_ms=index - 1,
                        end_ms=index,
                    )
                    for index in range(1, segment_count + 1)
                ],
                "partial": None,
            }
        )
        assert len(document.stable_lines) == segment_count
        assert projection_call_count == 1

        for index in range(1, segment_count + 1):
            document.apply_event(
                {
                    "type": "translation_stable",
                    "source_segment_id": f"seg_{index:06d}",
                    "source_segment_index": index,
                    "source_segment_ids": [f"seg_{index:06d}"],
                    "source_segment_indices": [index],
                    "text": f"t{index}",
                }
            )
            assert document.stable_lines[index - 1].translation == f"t{index}"

        assert projection_call_count == 1

    def test_final_snapshot_prefers_segment_translation_status_over_replayed_state(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000)
                ],
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

    def test_final_marker_without_snapshot_clears_current_and_keeps_replayed_stable_history(
        self,
    ) -> None:
        document = SubtitleDocument()
        document.apply_event(
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_base": 0,
                "stable_count": 1,
                "stable_appends": [
                    stable_segment(1, "第一句", start_ms=0, end_ms=1000)
                ],
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
