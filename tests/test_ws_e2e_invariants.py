# coding=utf-8
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools.ws_e2e_leak_check import (
    _compute_timestamp_quality,
    _detect_repetition_loop,
    _expects_translation,
    _final_document_contract_coverage,
    _final_document_translation_coverage_issues,
    _final_event_contract_issues,
    _record_event_contract,
    _repetition_validation_issues,
)


def source_segment(index: int, text: str, **extra: object) -> dict[str, object]:
    return {"id": f"seg_{index:06d}", "index": index, "text": text, **extra}


def replay_source_segments(
    state: dict[str, object], *segments: dict[str, object]
) -> None:
    for revision, segment in enumerate(segments, start=1):
        _record_event_contract(
            state,
            {
                "type": "transcript_update",
                "revision": revision,
                "stable_appends": [segment],
                "partial": None,
            },
        )


def record_translation(
    state: dict[str, object],
    source_indices: list[int],
    text: str,
    *,
    source_ids: list[str] | None = None,
    paired_indices: list[int] | None = None,
) -> list[str]:
    ids = source_ids or [f"seg_{index:06d}" for index in source_indices]
    indices = paired_indices or source_indices
    return _record_event_contract(
        state,
        {
            "type": "translation_stable",
            "source_segment_id": ids[-1],
            "source_segment_index": indices[-1],
            "source_segment_ids": ids,
            "source_segment_indices": indices,
            "text": text,
        },
    )


def record_translation_status(
    state: dict[str, object],
    source_indices: list[int],
    *,
    code: str = "timeout",
    source_ids: list[str] | None = None,
    paired_indices: list[int] | None = None,
) -> list[str]:
    ids = source_ids or [f"seg_{index:06d}" for index in source_indices]
    indices = paired_indices or source_indices
    return _record_event_contract(
        state,
        {
            "type": "translation_status",
            "scope": "stable",
            "code": code,
            "source_segment_id": ids[-1],
            "source_segment_index": indices[-1],
            "source_segment_ids": ids,
            "source_segment_indices": indices,
            "message": "translation failed",
        },
    )


def final_event(
    segments: list[dict[str, object]],
    *,
    translation_units: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "type": "transcript_final",
        "segments": segments,
        "document": {
            "schemaVersion": 1,
            "durationMs": 1000 * len(segments),
            "language": "English",
            "text": " ".join(str(segment.get("text") or "") for segment in segments),
            "segments": segments,
            "translationUnits": translation_units or [],
        },
    }


def translation_unit(
    source_indices: list[int],
    text: str,
    *,
    source_ids: list[str] | None = None,
    paired_indices: list[int] | None = None,
) -> dict[str, object]:
    return {
        "sourceSegmentIds": source_ids
        or [f"seg_{index:06d}" for index in source_indices],
        "sourceSegmentIndices": paired_indices or source_indices,
        "targetLanguage": "English",
        "text": text,
    }


class TestWebSocketE2EInvariant:
    def test_target_language_implies_translation_validation(self) -> None:
        assert _expects_translation(
            SimpleNamespace(expect_translation=False, target_language="English")
        )
        assert not _expects_translation(
            SimpleNamespace(expect_translation=False, target_language="")
        )

    def test_final_document_contract_coverage_reports_unchecked_realtime_final(
        self,
    ) -> None:
        assert _final_document_contract_coverage(
            {"type": "transcript_final", "segments": []}
        ) == {
            "checked": False,
            "reason": "transcript_final has no document",
        }
        assert _final_document_contract_coverage(
            {"type": "transcript_final", "segments": [], "document": {}}
        ) == {
            "checked": True,
            "reason": None,
        }

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
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        record_translation(state, [1], "English:one")

        issues = _final_event_contract_issues(
            state,
            {
                "type": "transcript_final",
                "segments": segments,
            },
            expect_translation=True,
        )

        assert issues == [
            "missing stable translation outcome for source segments: seg_000002"
        ]

    def test_grouped_stable_translation_can_cover_multiple_source_segments(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        assert record_translation(state, [1, 2], "English:one two") == []

        issues = _final_event_contract_issues(
            state,
            {
                "type": "transcript_final",
                "segments": segments,
            },
            expect_translation=True,
        )

        assert issues == []

    def test_empty_stable_translation_does_not_count_as_coverage(self) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one")]
        replay_source_segments(state, *segments)

        issues = record_translation(state, [1], "   ")

        assert issues == ["translation_stable text is empty"]
        assert _final_event_contract_issues(
            state,
            final_event(segments),
            expect_translation=True,
        ) == [
            "missing stable translation outcome for source segments: seg_000001",
            "transcript_final document missing translation coverage for source segments: seg_000001",
        ]

    def test_final_document_accepts_single_segment_translation_on_document_segment(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one", translation="English:one")]
        replay_source_segments(state, source_segment(1, "one"))
        record_translation(state, [1], "English:one")

        issues = _final_event_contract_issues(
            state,
            final_event(segments),
            expect_translation=True,
        )

        assert issues == []

    def test_final_document_segment_translation_must_match_stable_translation_text(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one", translation="different")]
        replay_source_segments(state, source_segment(1, "one"))
        record_translation(state, [1], "English:one")

        issues = _final_event_contract_issues(
            state,
            final_event(segments),
            expect_translation=True,
        )

        assert issues == [
            "transcript_final document.segments translation text mismatch for source coverage: seg_000001"
        ]

    def test_final_document_translation_units_must_cover_grouped_source(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        record_translation(state, [1, 2], "English:one two")

        issues = _final_event_contract_issues(
            state,
            final_event(segments),
            expect_translation=True,
        )

        assert issues == [
            "transcript_final document missing translation coverage for source segments: seg_000001, seg_000002"
        ]

    def test_final_document_segment_translations_do_not_cover_grouped_source(
        self,
    ) -> None:
        state: dict[str, object] = {}
        source_segments = [source_segment(1, "one"), source_segment(2, "two")]
        translated_segments = [
            source_segment(1, "one", translation="English:one"),
            source_segment(2, "two", translation="English:two"),
        ]
        replay_source_segments(state, *source_segments)
        record_translation(state, [1, 2], "English:one two")

        issues = _final_event_contract_issues(
            state,
            final_event(translated_segments),
            expect_translation=True,
        )

        assert issues == [
            "transcript_final document missing translationUnits coverage for grouped source segments: seg_000001, seg_000002"
        ]

    def test_final_document_translation_units_must_match_stable_translation_text(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one")]
        replay_source_segments(state, *segments)
        record_translation(state, [1], "English:one")

        issues = _final_event_contract_issues(
            state,
            final_event(
                segments, translation_units=[translation_unit([1], "different")]
            ),
            expect_translation=True,
        )

        assert issues == [
            "transcript_final document.translationUnits text mismatch for source coverage: seg_000001"
        ]

    def test_final_document_translation_units_validate_paired_indices(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        record_translation(state, [1, 2], "English:one two")

        issues = _final_event_contract_issues(
            state,
            final_event(
                segments,
                translation_units=[
                    translation_unit([1, 2], "English:one two", paired_indices=[1, 99])
                ],
            ),
            expect_translation=True,
        )

        assert issues == [
            "transcript_final document.translationUnits sourceSegmentIndices[1] 99 does not match source segment seg_000002 index 2"
        ]

    def test_final_document_translation_units_accept_stale_id_with_index_fallback(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        record_translation(state, [1, 2], "English:one two")

        issues = _final_event_contract_issues(
            state,
            final_event(
                segments,
                translation_units=[
                    translation_unit(
                        [1, 2],
                        "English:one two",
                        source_ids=["seg_000001", "old_seg_000002"],
                    )
                ],
            ),
            expect_translation=True,
        )

        assert issues == []

    def test_final_document_translation_units_accept_grouped_status_coverage(
        self,
    ) -> None:
        segments = [source_segment(1, "one"), source_segment(2, "two")]

        issues = _final_document_translation_coverage_issues(
            {},
            final_event(
                segments,
                translation_units=[
                    {
                        "sourceSegmentIds": ["seg_000001", "seg_000002"],
                        "sourceSegmentIndices": [1, 2],
                        "targetLanguage": "English",
                        "text": "",
                        "translationStatus": "timeout",
                        "translationMessage": "translation failed",
                    }
                ],
            ),
            ["seg_000001", "seg_000002"],
        )

        assert issues == []

    def test_final_contract_accepts_grouped_stable_status_as_terminal_coverage(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        assert record_translation_status(state, [1, 2]) == []

        issues = _final_event_contract_issues(
            state,
            final_event(
                segments,
                translation_units=[
                    {
                        "sourceSegmentIds": ["seg_000001", "seg_000002"],
                        "sourceSegmentIndices": [1, 2],
                        "targetLanguage": "English",
                        "text": "",
                        "translationStatus": "timeout",
                        "translationMessage": "translation failed",
                    }
                ],
            ),
            expect_translation=True,
        )

        assert issues == []

    def test_final_document_translation_units_must_match_stable_status(
        self,
    ) -> None:
        state: dict[str, object] = {}
        segments = [source_segment(1, "one"), source_segment(2, "two")]
        replay_source_segments(state, *segments)
        assert record_translation_status(state, [1, 2], code="timeout") == []

        issues = _final_event_contract_issues(
            state,
            final_event(
                segments,
                translation_units=[
                    {
                        "sourceSegmentIds": ["seg_000001", "seg_000002"],
                        "sourceSegmentIndices": [1, 2],
                        "targetLanguage": "English",
                        "text": "",
                        "translationStatus": "failed",
                        "translationMessage": "translation failed",
                    }
                ],
            ),
            expect_translation=True,
        )

        assert issues == [
            "transcript_final document.translationUnits status mismatch for source coverage: seg_000001, seg_000002"
        ]

    def test_grouped_stable_translation_rejects_mismatched_coverage_indices(
        self,
    ) -> None:
        state: dict[str, object] = {}
        replay_source_segments(
            state, source_segment(1, "one"), source_segment(2, "two")
        )

        issues = record_translation(
            state, [1, 2], "English:one two", paired_indices=[1, 99]
        )

        assert issues == [
            "translation_stable source_segment_indices[1] 99 does not match source segment seg_000002 index 2"
        ]

    def test_grouped_stable_translation_rejects_non_contiguous_coverage(
        self,
    ) -> None:
        state: dict[str, object] = {}
        replay_source_segments(
            state,
            source_segment(1, "one"),
            source_segment(2, "middle"),
            source_segment(3, "three"),
        )

        issues = record_translation(
            state,
            [1, 3],
            "English:one three",
        )

        assert issues == [
            "translation_stable source coverage is not contiguous in stable history: seg_000001, seg_000003"
        ]

    def test_final_marker_without_segments_uses_replayed_stable_history(self) -> None:
        state: dict[str, object] = {}
        replay_source_segments(state, source_segment(1, "one"))

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
