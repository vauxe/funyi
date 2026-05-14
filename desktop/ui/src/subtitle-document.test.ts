import test from "node:test";
import assert from "node:assert/strict";

import { SubtitleDocument } from "./subtitle-document.js";

interface SegmentOptions {
  startMs?: number;
  endMs?: number;
  timingStatus?: string;
}

function stableSegment(index: number, text: string, { startMs, endMs, timingStatus }: SegmentOptions = {}) {
  const segment: Record<string, unknown> = {
    id: `seg_${String(index).padStart(6, "0")}`,
    index,
    start_ms: startMs,
    end_ms: endMs,
    text,
    language: "Chinese",
  };
  if (timingStatus !== undefined) {
    segment.timing_status = timingStatus;
  }
  return segment;
}

function partialSegment(text: string, { startMs, endMs }: SegmentOptions = {}) {
  return {
    start_ms: startMs,
    end_ms: endMs,
    text,
    language: "Chinese",
  };
}

test("window scrolls when current partial becomes stable", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("draft", { startMs: 0, endMs: 1000 }),
  });

  assert.equal(document.window().current?.text, "draft");

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "draft", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("next", { startMs: 1000, endMs: 1600 }),
  });

  const window = document.window();
  assert.equal(window.previous?.text, "draft");
  assert.equal(window.current?.text, "next");
});

test("translation annotates matching lines and stale preview is ignored", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "stable", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("current", { startMs: 1000, endMs: 1800 }),
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    text: "stable line",
  });
  document.applyEvent({ type: "translation_preview", source_revision: 0, text: "stale" });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "current line" });

  const window = document.window();
  assert.equal(window.previous?.translation, "stable line");
  assert.equal(window.current?.translation, "current line");
  assert.equal(
    document.toSrt(),
    "1\n00:00:00,000 --> 00:00:01,000\nstable\nstable line\n",
  );
});

test("stale stable base is rejected", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "one", { startMs: 0, endMs: 1000 })],
    partial: null,
  });

  assert.throws(() => {
    document.applyEvent({
      type: "transcript_update",
      revision: 2,
      stable_base: 0,
      stable_count: 2,
      stable_appends: [stableSegment(2, "two", { startMs: 1000, endMs: 2000 })],
      partial: null,
    });
  }, /stable cursor mismatch/);
});

test("final snapshot clears current and preserves stable translation", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "one", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("tail", { startMs: 1000, endMs: 1500 }),
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    text: "first line",
  });
  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    stable_count: 1,
    segments: [stableSegment(1, "one", { startMs: 0, endMs: 1000 })],
  });

  assert.equal(document.window().previous?.translation, "first line");
  assert.equal(document.window().current, null);
});
