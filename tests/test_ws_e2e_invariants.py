# coding=utf-8
from __future__ import annotations

import unittest

from tools.ws_e2e_leak_check import _compute_timestamp_quality, _final_event_contract_issues, _record_event_contract


class WebSocketE2EInvariantTest(unittest.TestCase):
    def test_preview_must_not_lag_behind_latest_transcript_revision(self) -> None:
        state: dict[str, object] = {}

        self.assertEqual(
            _record_event_contract(
                state,
                {"type": "transcript_update", "revision": 2, "stable_appends": [], "partial": {"text": "new"}},
            ),
            [],
        )
        issues = _record_event_contract(
            state,
            {"type": "translation_preview", "source_revision": 1, "text": "old"},
        )

        self.assertEqual(
            issues,
            ["translation_preview source_revision 1 is older than latest transcript_update revision 2"],
        )

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

        self.assertEqual(issues, ["missing translation_stable for source segments: seg_000002"])

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

        self.assertEqual(issues, ["transcript_timing_update references unknown source segment: seg_000001"])

    def test_timestamp_quality_uses_srt_text_match_for_boundary_errors(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "ref.srt"
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

        self.assertIsNotNone(quality)
        self.assertEqual(quality["matched_segments"], 2)  # type: ignore[index]
        self.assertEqual(quality["boundary_abs_error_ms"]["p50"], 0.0)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
