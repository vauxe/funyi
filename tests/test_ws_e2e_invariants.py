# coding=utf-8
from __future__ import annotations

from pathlib import Path

from tools.ws_e2e_leak_check import (
    _compute_timestamp_quality,
    _detect_repetition_loop,
    _final_event_contract_issues,
    _record_event_contract,
    _repetition_validation_issues,
)


class TestWebSocketE2EInvariant:
    def test_preview_must_not_lag_behind_latest_transcript_revision(self) -> None:
        state: dict[str, object] = {}

        assert (
            _record_event_contract(
                state,
                {
                    "type": "transcript_update",
                    "revision": 2,
                    "stable_appends": [],
                    "partial": {"text": "new"},
                },
            )
            == []
        )
        issues = _record_event_contract(
            state,
            {"type": "translation_preview", "source_revision": 1, "text": "old"},
        )

        assert issues == [
            "translation_preview source_revision 1 is older than latest transcript_update revision 2"
        ]

    def test_stable_translation_history_must_cover_final_source_segments(self) -> None:
        state: dict[str, object] = {}
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_appends": [{"id": "seg_000001", "index": 1, "text": "one"}],
                "partial": None,
            },
        )
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 2,
                "stable_appends": [{"id": "seg_000002", "index": 2, "text": "two"}],
                "partial": None,
            },
        )
        _record_event_contract(
            state,
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000001",
                "source_segment_index": 1,
                "text": "English:one",
            },
        )

        issues = _final_event_contract_issues(
            state,
            {
                "type": "transcript_final",
                "segments": [
                    {"id": "seg_000001", "index": 1, "text": "one"},
                    {"id": "seg_000002", "index": 2, "text": "two"},
                ],
            },
            expect_translation=True,
        )

        assert issues == ["missing translation_stable for source segments: seg_000002"]

    def test_grouped_stable_translation_can_cover_multiple_source_segments(
        self,
    ) -> None:
        state: dict[str, object] = {}
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_appends": [{"id": "seg_000001", "index": 1, "text": "one"}],
                "partial": None,
            },
        )
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 2,
                "stable_appends": [{"id": "seg_000002", "index": 2, "text": "two"}],
                "partial": None,
            },
        )
        assert (
            _record_event_contract(
                state,
                {
                    "type": "translation_stable",
                    "source_segment_id": "seg_000002",
                    "source_segment_index": 2,
                    "source_segment_ids": ["seg_000001", "seg_000002"],
                    "source_segment_indices": [1, 2],
                    "text": "English:one two",
                },
            )
            == []
        )

        issues = _final_event_contract_issues(
            state,
            {
                "type": "transcript_final",
                "segments": [
                    {"id": "seg_000001", "index": 1, "text": "one"},
                    {"id": "seg_000002", "index": 2, "text": "two"},
                ],
            },
            expect_translation=True,
        )

        assert issues == []

    def test_grouped_stable_translation_rejects_mismatched_coverage_indices(
        self,
    ) -> None:
        state: dict[str, object] = {}
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_appends": [{"id": "seg_000001", "index": 1, "text": "one"}],
                "partial": None,
            },
        )
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 2,
                "stable_appends": [{"id": "seg_000002", "index": 2, "text": "two"}],
                "partial": None,
            },
        )

        issues = _record_event_contract(
            state,
            {
                "type": "translation_stable",
                "source_segment_id": "seg_000002",
                "source_segment_index": 2,
                "source_segment_ids": ["seg_000001", "seg_000002"],
                "source_segment_indices": [1, 99],
                "text": "English:one two",
            },
        )

        assert issues == [
            "translation_stable source_segment_indices[1] 99 does not match source segment seg_000002 index 2"
        ]

    def test_final_marker_without_segments_uses_replayed_stable_history(self) -> None:
        state: dict[str, object] = {}
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": 1,
                "stable_appends": [{"id": "seg_000001", "index": 1, "text": "one"}],
                "partial": None,
            },
        )

        issues = _final_event_contract_issues(
            state,
            {
                "type": "transcript_final",
                "revision": 2,
                "final_revision": 2,
                "stable_count": 1,
            },
            expect_translation=False,
        )

        assert issues == []

    def test_timing_update_must_reference_known_source_segment(self) -> None:
        state: dict[str, object] = {}

        issues = _record_event_contract(
            state,
            {
                "type": "transcript_timing_update",
                "source_segment_id": "seg_000001",
                "start_ms": 0,
                "end_ms": 1000,
                "timing_status": "aligned",
            },
        )

        assert issues == [
            "transcript_timing_update references unknown source segment: seg_000001"
        ]

    def test_timestamp_quality_uses_srt_text_match_for_boundary_errors(
        self, tmp_path: Path
    ) -> None:
        srt_path = tmp_path / "ref.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n第一句\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\n第二句\n",
            encoding="utf-8",
        )

        quality = _compute_timestamp_quality(
            reference_srt=str(srt_path),
            final_event={
                "type": "transcript_final",
                "segments": [
                    {
                        "id": "seg_000001",
                        "start_ms": 100,
                        "end_ms": 900,
                        "timing_status": "aligned",
                        "text": "第一句",
                    },
                    {
                        "id": "seg_000002",
                        "start_ms": 1000,
                        "end_ms": 2000,
                        "timing_status": "aligned",
                        "text": "第二句",
                    },
                ],
            },
            start_sec=0.0,
            duration_sec=2.0,
            strip_ruby=False,
        )

        assert quality is not None
        assert quality["matched_segments"] == 2  # type: ignore[index]
        assert quality["boundary_abs_error_ms"]["p50"] == 0.0  # type: ignore[index]

    def test_repetition_loop_detector_flags_long_adjacent_repeated_text(self) -> None:
        loop = _detect_repetition_loop("前缀" + "重复内容甲乙丙丁" * 12 + "后缀")

        assert loop is not None
        assert loop["repeat_count"] >= 5

    def test_repetition_loop_detector_ignores_short_natural_repeats(self) -> None:
        assert _detect_repetition_loop("谢谢谢谢，好的好的，我们继续。") is None

    def test_repetition_loop_detector_ignores_text_below_normalized_floor(self) -> None:
        # 79 normalized chars sits just under the 80-char floor and must never be flagged.
        text = "甲乙丙丁戊己庚辛" * 9 + "甲乙丙丁戊己庚"
        assert len("".join(ch for ch in text if not ch.isspace())) == 79
        assert _detect_repetition_loop(text) is None

    def test_repetition_loop_detector_flags_repeats_separated_by_punctuation(
        self,
    ) -> None:
        # Punctuation and spaces are normalized away, so an interleaved loop is still caught.
        loop = _detect_repetition_loop("，".join(["重复内容甲乙丙丁"] * 12))

        assert loop is not None
        assert loop["unit_chars"] >= 8

    def test_repetition_validation_issues_reports_loop_spanning_segments(self) -> None:
        # The loop only emerges once segment texts are concatenated.
        segments = [
            {"text": "重复内容甲乙丙丁" * 6},
            {"text": "重复内容甲乙丙丁" * 6},
        ]
        issues = _repetition_validation_issues(segments)

        assert len(issues) == 1
        assert "repetition loop" in issues[0]

    def test_repetition_validation_issues_passes_clean_transcript(self) -> None:
        segments = [{"text": "今天天气不错。"}, {"text": "我们去公园散步看花。"}]
        assert _repetition_validation_issues(segments) == []
