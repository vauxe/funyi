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

test("keeps latest stable line visible until a new partial arrives", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("draft", { startMs: 0, endMs: 1000 }),
  });

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "draft", { startMs: 0, endMs: 1000 })],
    partial: null,
  });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "translated draft" });

  let window = document.window();
  assert.equal(window.previous?.text, "draft");
  assert.equal(window.current?.text, "draft");
  assert.equal(window.current?.translation, "translated draft");

  document.applyEvent({
    type: "transcript_update",
    revision: 3,
    stable_base: 1,
    stable_count: 1,
    stable_appends: [],
    partial: null,
  });
  assert.equal(document.window().current?.text, "draft");

  document.applyEvent({
    type: "transcript_update",
    revision: 4,
    stable_base: 1,
    stable_count: 1,
    stable_appends: [],
    partial: partialSegment("next", { startMs: 1000, endMs: 1800 }),
  });

  window = document.window();
  assert.equal(window.previous?.text, "draft");
  assert.equal(window.current?.text, "next");
  assert.equal(window.current?.translation, null);
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
  assert.equal(document.toSrt(), "1\n00:00:00,000 --> 00:00:01,000\nstable\nstable line\n");
});

test("window returns complete latest lines", () => {
  const document = new SubtitleDocument();
  const stableText = "一二三四五六七八九十甲乙丙丁戊己庚辛。后续文本";
  const partialText = "当前文本也可能很长。最后显示";
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, stableText, { startMs: 0, endMs: 2300 })],
    partial: partialSegment(partialText, { startMs: 2300, endMs: 3300 }),
  });

  assert.equal(document.stableLines.length, 1);
  assert.equal(document.stableLines[0]?.text, stableText);
  const window = document.window();
  assert.equal(window.previous?.text, stableText);
  assert.equal(window.current?.text, partialText);
});

test("current translation preview survives repeated partial revisions", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current", { startMs: 1000, endMs: 1800 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "current line" });

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current", { startMs: 1000, endMs: 1800 }),
  });
  assert.equal(document.window().current?.translation, "current line");

  document.applyEvent({
    type: "transcript_update",
    revision: 3,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current text", { startMs: 1000, endMs: 2200 }),
  });
  assert.equal(document.window().current?.translation, "current line");

  document.applyEvent({
    type: "transcript_update",
    revision: 4,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current text extended", { startMs: 1000, endMs: 3000 }),
  });
  assert.equal(document.window().current?.translation, "current line");

  document.applyEvent({
    type: "transcript_update",
    revision: 5,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "current text", { startMs: 1000, endMs: 2200 })],
    partial: partialSegment("next line", { startMs: 2200, endMs: 3000 }),
  });
  assert.equal(document.window().previous?.translation, "current line");
  assert.equal(document.window().current?.translation, null);

  document.applyEvent({ type: "translation_preview", source_revision: 4, text: "late old line" });
  assert.equal(document.window().current?.translation, null);

  document.applyEvent({ type: "translation_preview", source_revision: 5, text: "next translation" });
  assert.equal(document.window().current?.translation, "next translation");
});

test("current translation preview survives same-timed partial rewrites", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("we start with this wording", { startMs: 1000, endMs: 1800 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "translated wording" });

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("we began with another wording", { startMs: 1000, endMs: 2200 }),
  });

  assert.equal(document.window().current?.translation, "translated wording");
});

test("current translation preview is not carried to a different partial", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("first topic", { startMs: 0, endMs: 1000 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "first translation" });

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("different sentence", { startMs: 1000, endMs: 2000 }),
  });

  assert.equal(document.window().current?.text, "different sentence");
  assert.equal(document.window().current?.translation, null);
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "late first translation" });
  assert.equal(document.window().current?.translation, null);
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

test("invalid transcript segment shapes are rejected", () => {
  const document = new SubtitleDocument();

  assert.throws(() => {
    document.applyEvent({
      type: "transcript_update",
      revision: 1,
      stable_base: 0,
      stable_count: 1,
      stable_appends: {},
      partial: null,
    });
  }, /stable_appends must be an array/);

  assert.throws(() => {
    document.applyEvent({
      type: "transcript_update",
      revision: 1,
      stable_base: 0,
      stable_count: 1,
      stable_appends: [[]],
      partial: null,
    });
  }, /stable_appends item must be an object/);

  assert.throws(() => {
    document.applyEvent({
      type: "transcript_update",
      revision: 1,
      stable_base: 0,
      stable_count: 0,
      stable_appends: [],
      partial: [],
    });
  }, /partial must be an object/);

  assert.throws(() => {
    document.applyEvent({
      type: "transcript_final",
      revision: 1,
      stable_count: 1,
      segments: [[]],
    });
  }, /segments item must be an object/);
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
