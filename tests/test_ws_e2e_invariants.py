# coding=utf-8
from __future__ import annotations

import unittest

from tools.ws_e2e_leak_check import _final_event_contract_issues, _record_event_contract


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


if __name__ == "__main__":
    unittest.main()
