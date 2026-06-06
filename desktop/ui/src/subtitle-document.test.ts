import test from "node:test";
import assert from "node:assert/strict";

import { SubtitleDocument } from "./subtitle-document.js";
import { parseTranscriptDocumentSnapshot } from "./transcription-document.js";

interface SegmentOptions {
  startMs?: number;
  endMs?: number;
  timingStatus?: string;
  translation?: string;
}

function stableSegment(
  index: number,
  text: string,
  { startMs, endMs, timingStatus, translation }: SegmentOptions = {},
) {
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
  if (translation !== undefined) {
    segment.translation = translation;
  }
  return segment;
}

function documentWithReplayedStableTranslation(): SubtitleDocument {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "one", { startMs: 0, endMs: 1000 })],
    partial: null,
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    text: "old translation",
  });
  return document;
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
  assert.equal(document.stableLines.at(-1)?.text, "draft");
  assert.equal(window.current?.text, "next");
});

test("offline transcript snapshot replaces document history directly", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  const snapshot = parseTranscriptDocumentSnapshot({
    schemaVersion: 1,
    durationMs: 1800,
    language: "Chinese",
    text: "你好世界",
    segments: [
      {
        id: "seg_000001",
        index: 1,
        startMs: 0,
        endMs: 1000,
        text: "你好",
        language: "Chinese",
        timingStatus: "aligned",
        translation: "hello",
      },
      {
        id: "seg_000002",
        index: 2,
        startMs: 1000,
        endMs: 1800,
        text: "世界",
        language: "Chinese",
        timingStatus: "estimated",
      },
    ],
  });

  document.replaceSnapshot(snapshot);

  assert.equal(document.stableLines.length, 2);
  assert.equal(document.window().current?.text, "世界");
  assert.equal(document.stableLines[0]?.translation, "hello");
  assert.equal(document.stableLines[1]?.timingStatus, "estimated");
});

test("offline transcript snapshot anchors grouped translation without folding source cues", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  const snapshot = parseTranscriptDocumentSnapshot({
    schemaVersion: 1,
    durationMs: 3800,
    language: "Chinese",
    text: "今天讨论字幕显示问题，并且保持翻译输入完整。",
    segments: [
      {
        id: "seg_000001",
        index: 1,
        startMs: 0,
        endMs: 2000,
        text: "今天讨论字幕显示问题，",
        language: "Chinese",
        timingStatus: "aligned",
      },
      {
        id: "seg_000002",
        index: 2,
        startMs: 2000,
        endMs: 3800,
        text: "并且保持翻译输入完整。",
        language: "Chinese",
        timingStatus: "aligned",
      },
    ],
    translationUnits: [
      {
        text: "We discuss subtitle display while preserving translation context.",
        targetLanguage: "English",
        sourceSegmentIds: ["seg_000001", "seg_000002"],
        sourceSegmentIndices: [1, 2],
      },
    ],
  });

  document.replaceSnapshot(snapshot);

  assert.equal(document.stableLines.length, 2);
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天讨论字幕显示问题，", "并且保持翻译输入完整。"],
  );
  assert.equal(
    document.stableLines[1]?.translation,
    "We discuss subtitle display while preserving translation context.",
  );
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
  assert.equal(document.stableLines.at(-1)?.text, "draft");
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
  assert.equal(document.stableLines.at(-1)?.text, "draft");
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
  assert.equal(document.stableLines.at(-1)?.translation, "stable line");
  assert.equal(window.current?.translation, "current line");
});

test("grouped stable translation annotates the anchor without folding source segments", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "一下这个问题", { startMs: 1000, endMs: 1800 }),
    ],
    partial: null,
  });

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "Today we discuss this issue",
  });

  assert.equal(document.stableLines.length, 2);
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲", "一下这个问题"],
  );
  assert.equal(document.stableLines[1]?.translation, "Today we discuss this issue");

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 2,
    stable_count: 3,
    stable_appends: [stableSegment(3, "下一句", { startMs: 1800, endMs: 2400 })],
    partial: null,
  });

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲", "一下这个问题", "下一句"],
  );
});

test("preview display composes pending stable prefix with current partial after translation arrives", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("一下这个", { startMs: 1000, endMs: 1800 }),
  });

  assert.equal(document.window().current?.text, "一下这个");
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲"],
  );

  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "Today we discuss this" });

  assert.equal(document.window().current?.text, "今天我们来讲一下这个");
  assert.equal(document.window().current?.translation, "Today we discuss this");
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    [],
  );
});

test("grouped stable translation restores all pending source cues", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "一下这个问题", { startMs: 1000, endMs: 1800 }),
    ],
    partial: partialSegment("后面继续", { startMs: 1800, endMs: 2400 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "Today we discuss this issue" });

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    [],
  );

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "Today we discuss this issue",
  });

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲", "一下这个问题"],
  );
  assert.equal(document.stableLines[1]?.translation, "Today we discuss this issue");
  assert.equal(document.window().current?.text, "后面继续");
  assert.equal(document.window().current?.translation, null);
});

test("grouped stable translation restores pending source cues by index fallback", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "一下这个问题", { startMs: 1000, endMs: 1800 }),
    ],
    partial: partialSegment("后面继续", { startMs: 1800, endMs: 2400 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "Today we discuss this issue" });

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    [],
  );

  document.applyEvent({
    type: "translation_stable",
    source_segment_index: 2,
    source_segment_indices: [1, 2],
    text: "Today we discuss this issue",
  });

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲", "一下这个问题"],
  );
  assert.equal(document.stableLines[1]?.translation, "Today we discuss this issue");
  assert.equal(document.window().current?.text, "后面继续");
  assert.equal(document.window().current?.translation, null);
});

test("source-only display ignores translation preview composition", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("一下这个", { startMs: 1000, endMs: 1800 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "Today we discuss this" });

  document.setTranslationEnabled(false);

  assert.equal(document.window().current?.text, "一下这个");
  assert.equal(document.window().current?.translation, null);
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲"],
  );
});

test("source-only history keeps grouped stable translation source cues", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "一下这个问题", { startMs: 1000, endMs: 1800 }),
    ],
    partial: null,
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "Today we discuss this issue",
  });

  document.setTranslationEnabled(false);

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲", "一下这个问题"],
  );
});

test("stable translation status clears an existing translation", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "hello", { startMs: 0, endMs: 1000 })],
    partial: null,
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    text: "bonjour",
  });

  document.applyEvent({
    type: "translation_status",
    scope: "stable",
    code: "failed",
    message: "translation failed",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
  });

  assert.equal(document.stableLines[0]?.translation, null);
  assert.equal(document.stableLines[0]?.translationMessage, "translation failed");
});

test("stable translation status clears preview carried to a stable line", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current text", { startMs: 1000, endMs: 2200 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "preview translation" });

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "current text", { startMs: 1000, endMs: 2200 })],
    partial: partialSegment("next line", { startMs: 2200, endMs: 3000 }),
  });

  assert.equal(document.stableLines[0]?.translation, "preview translation");

  document.applyEvent({
    type: "translation_status",
    scope: "stable",
    code: "failed",
    message: "translation failed",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
  });

  assert.equal(document.stableLines[0]?.translation, null);
  assert.equal(document.stableLines[0]?.translationMessage, "translation failed");
  assert.equal(document.window().current?.translation, null);
});

test("grouped stable translation status annotates the anchor without folding covered source segments", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "一下这个问题", { startMs: 1000, endMs: 1800 }),
    ],
    partial: null,
  });

  document.applyEvent({
    type: "translation_status",
    scope: "stable",
    code: "failed",
    message: "translation failed",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
  });

  assert.equal(document.stableLines.length, 2);
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["今天我们来讲", "一下这个问题"],
  );
  assert.equal(document.stableLines[1]?.translation, null);
  assert.equal(document.stableLines[1]?.translationMessage, "translation failed");
});

test("stable translation status clears pending stable line visibility", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "今天我们来讲", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("一下这个", { startMs: 1000, endMs: 1800 }),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "Today we discuss this" });

  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    [],
  );

  document.applyEvent({
    type: "translation_status",
    scope: "stable",
    code: "failed",
    message: "translation failed",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
  });

  assert.equal(document.stableLines.length, 1);
  assert.equal(document.stableLines[0]?.translationMessage, "translation failed");
  assert.equal(document.window().current?.text, "一下这个");
  assert.equal(document.window().current?.translation, null);
});

test("translation falls back to source_segment_index when the id does not match", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "hello", { startMs: 0, endMs: 1000 })],
    partial: null,
  });

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_does_not_match",
    source_segment_index: 1,
    text: "bonjour",
  });

  assert.equal(document.stableLines.at(-1)?.translation, "bonjour");
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
  assert.equal(document.stableLines.at(-1)?.text, stableText);
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
  assert.equal(document.stableLines.at(-1)?.translation, "current line");
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

  assert.equal(document.stableLines.at(-1)?.translation, "first line");
  assert.equal(document.window().current, null);
});

test("final snapshot preserves stable translation by index when ids change", () => {
  const document = documentWithReplayedStableTranslation();
  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    stable_count: 1,
    segments: [{ ...stableSegment(1, "one", { startMs: 0, endMs: 1000 }), id: "rebuilt_seg_1" }],
  });

  assert.equal(document.stableLines.at(-1)?.translation, "old translation");
  assert.equal(document.window().current, null);
});

test("final snapshot prefers segment translation over replayed state", () => {
  const document = documentWithReplayedStableTranslation();
  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    stable_count: 1,
    segments: [stableSegment(1, "one", { startMs: 0, endMs: 1000, translation: "final translation" })],
  });

  assert.equal(document.stableLines.at(-1)?.translation, "final translation");
  assert.equal(document.window().current, null);
});

test("final snapshot anchors document translation units", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_final",
    revision: 1,
    stable_count: 2,
    segments: [
      stableSegment(1, "今天讨论字幕显示问题，", { startMs: 0, endMs: 2000 }),
      stableSegment(2, "并且保持翻译输入完整。", { startMs: 2000, endMs: 3800 }),
    ],
    document: {
      schemaVersion: 1,
      durationMs: 3800,
      language: "Chinese",
      text: "今天讨论字幕显示问题，并且保持翻译输入完整。",
      segments: [
        {
          id: "seg_000001",
          index: 1,
          startMs: 0,
          endMs: 2000,
          text: "今天讨论字幕显示问题，",
          language: "Chinese",
          timingStatus: "aligned",
        },
        {
          id: "seg_000002",
          index: 2,
          startMs: 2000,
          endMs: 3800,
          text: "并且保持翻译输入完整。",
          language: "Chinese",
          timingStatus: "aligned",
        },
      ],
      translationUnits: [
        {
          text: "We discuss subtitle display while preserving translation context.",
          targetLanguage: "English",
          sourceSegmentIds: ["seg_000001", "seg_000002"],
          sourceSegmentIndices: [1, 2],
        },
      ],
    },
  });

  assert.equal(
    document.stableLines[1]?.translation,
    "We discuss subtitle display while preserving translation context.",
  );
});

test("final snapshot anchors translation unit with paired index fallback", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_final",
    revision: 1,
    stable_count: 2,
    segments: [
      stableSegment(1, "今天讨论字幕显示问题，", { startMs: 0, endMs: 2000 }),
      { ...stableSegment(2, "并且保持翻译输入完整。", { startMs: 2000, endMs: 3800 }), id: "rebuilt_seg_2" },
    ],
    document: {
      schemaVersion: 1,
      durationMs: 3800,
      language: "Chinese",
      text: "今天讨论字幕显示问题，并且保持翻译输入完整。",
      segments: [],
      translationUnits: [
        {
          text: "We discuss subtitle display while preserving translation context.",
          targetLanguage: "English",
          sourceSegmentIds: ["seg_000001", "old_seg_000002"],
          sourceSegmentIndices: [1, 2],
        },
      ],
    },
  });

  assert.equal(document.stableLines[0]?.translation, null);
  assert.equal(
    document.stableLines[1]?.translation,
    "We discuss subtitle display while preserving translation context.",
  );
});

test("final snapshot prefers segment translation message over replayed state", () => {
  const document = documentWithReplayedStableTranslation();
  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    stable_count: 1,
    segments: [
      {
        ...stableSegment(1, "one", { startMs: 0, endMs: 1000 }),
        translation_status: "timeout",
        translation_message: "translation failed",
      },
    ],
  });

  assert.equal(document.stableLines.at(-1)?.translation, null);
  assert.equal(document.stableLines.at(-1)?.translationStatus, "timeout");
  assert.equal(document.stableLines.at(-1)?.translationMessage, "translation failed");
});

test("final snapshot translation status clears replayed translation even without a message", () => {
  const document = documentWithReplayedStableTranslation();
  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    stable_count: 1,
    segments: [
      {
        ...stableSegment(1, "one", { startMs: 0, endMs: 1000 }),
        translation_status: "timeout",
      },
    ],
  });

  assert.equal(document.stableLines.at(-1)?.translation, null);
  assert.equal(document.stableLines.at(-1)?.translationStatus, "timeout");
  assert.equal(document.stableLines.at(-1)?.translationMessage, null);
});

test("final snapshot with omitted segments keeps stable history (unbounded session)", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "one", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "two", { startMs: 1000, endMs: 2000 }),
    ],
    partial: partialSegment("tail", { startMs: 2000, endMs: 2500 }),
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    text: "first line",
  });

  // No `segments` key: the protocol says clients keep the history they replayed.
  document.applyEvent({ type: "transcript_final", revision: 2, stable_count: 2 });

  assert.equal(document.stableLines.length, 2);
  assert.deepEqual(
    document.stableLines.map((line) => line.text),
    ["one", "two"],
  );
  assert.equal(document.stableLines[0]?.translation, "first line");
  // Current caption is cleared at session end, matching the present-segments path.
  assert.equal(document.window().current, null);
});

test("replaceSnapshot uses snapshot translation message", () => {
  const document = new SubtitleDocument();
  document.replaceSnapshot({
    durationMs: 1000,
    language: "Chinese",
    schemaVersion: 1,
    segments: [
      {
        id: "seg_000001",
        index: 1,
        startMs: 0,
        endMs: 1000,
        text: "one",
        language: "Chinese",
        timingStatus: null,
        translation: null,
        translationStatus: "timeout",
        translationMessage: "translation failed",
      },
    ],
    text: "one",
    translationUnits: [],
  });

  assert.equal(document.stableLines[0]?.translationStatus, "timeout");
  assert.equal(document.stableLines[0]?.translationMessage, "translation failed");
  assert.equal(document.window().current?.translationStatus, "timeout");
  assert.equal(document.window().current?.translationMessage, "translation failed");
});

test("replaceSnapshot prefers snapshot segment translation over replayed state", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "one", { startMs: 0, endMs: 1000 })],
    partial: null,
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    text: "old translation",
  });
  document.applyEvent({
    type: "translation_status",
    scope: "stable",
    code: "timeout",
    message: "old status",
    source_segment_id: "seg_000001",
    source_segment_index: 1,
  });

  document.replaceSnapshot({
    durationMs: 1000,
    language: "English",
    schemaVersion: 1,
    segments: [
      {
        id: "seg_000001",
        index: 1,
        startMs: 0,
        endMs: 1000,
        text: "one",
        language: "English",
        timingStatus: null,
        translation: "final translation",
        translationStatus: null,
        translationMessage: null,
      },
    ],
    text: "one",
    translationUnits: [],
  });

  assert.equal(document.stableLines[0]?.translation, "final translation");
  assert.equal(document.stableLines[0]?.translationMessage, null);
  assert.equal(document.window().current?.translation, "final translation");
});

test("final snapshot with an explicit empty segments array still clears history", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "one", { startMs: 0, endMs: 1000 })],
    partial: null,
  });

  // Key present but empty: a genuine "no stable segments" snapshot rebuild.
  document.applyEvent({ type: "transcript_final", revision: 2, stable_count: 0, segments: [] });

  assert.equal(document.stableLines.length, 0);
  assert.equal(document.window().current, null);
});

test("final snapshot with omitted segments clears the latest-stable-as-current view", () => {
  const document = new SubtitleDocument();
  // A `partial: null` update after appends makes window() surface the latest stable
  // line as the current caption (showLatestStableAsCurrent = true). The omitted-segments
  // final must reset that flag so the current view clears at session end — without the
  // reset, window().current would keep showing "two" while history is preserved.
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "one", { startMs: 0, endMs: 1000 }),
      stableSegment(2, "two", { startMs: 1000, endMs: 2000 }),
    ],
    partial: null,
  });
  assert.equal(document.window().current?.text, "two");

  document.applyEvent({ type: "transcript_final", revision: 2, stable_count: 2 });

  assert.equal(document.stableLines.length, 2);
  assert.equal(document.window().current, null);
});
