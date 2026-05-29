import test from "node:test";
import assert from "node:assert/strict";

import { copyToClipboard, formatTranscript, type TranscriptLine } from "./transcript-export.js";

const LINES: TranscriptLine[] = [
  { startMs: 0, text: "hello", translation: "bonjour" },
  { startMs: 1000, text: "world", translation: "monde" },
];

test("formats lines with timestamps and translations", () => {
  assert.equal(
    formatTranscript(LINES, { translationEnabled: true }),
    "[00:00.000] hello\n-> bonjour\n[00:01.000] world\n-> monde",
  );
});

test("omits translations when translation is disabled", () => {
  assert.equal(formatTranscript(LINES, { translationEnabled: false }), "[00:00.000] hello\n[00:01.000] world");
});

test("drops empty lines and omits the timestamp when timing is missing", () => {
  const lines: TranscriptLine[] = [
    { startMs: null, text: "no timing", translation: null },
    { startMs: 5, text: "   ", translation: null },
  ];

  assert.equal(formatTranscript(lines, { translationEnabled: false }), "no timing");
});

test("emits a translation-only block when the source is empty", () => {
  const lines: TranscriptLine[] = [{ startMs: 0, text: "   ", translation: "hola" }];

  assert.equal(formatTranscript(lines, { translationEnabled: true }), "-> hola");
});

test("rejects when the clipboard API is unavailable", async () => {
  await assert.rejects(copyToClipboard("anything"), /Clipboard is unavailable/);
});
