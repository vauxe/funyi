import test from "node:test";
import assert from "node:assert/strict";

import { CaptionView } from "./caption-view.js";
import { SubtitleDocument } from "./subtitle-document.js";
import { clearDomGlobals } from "./test-browser-globals.fixture.js";
import {
  asDomElement,
  FakeDocument,
  FakeElement,
  installFakeDocument,
  installFakeElementConstructors,
} from "./test-dom.fixture.js";

test.beforeEach(() => {
  installFakeDocument(new FakeDocument());
  installFakeElementConstructors();
});

test.afterEach(() => {
  clearDomGlobals();
});

test("renders current caption window and stable history", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [
      stableSegment(1, "hello", {
        startMs: 1_000,
        endMs: 2_500,
        timingStatus: "aligned",
      }),
    ],
    partial: partialSegment("working"),
  });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "bonjour" });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "en cours" });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false, translationLanguage: "French" });

  assert.equal(elements.currentSource.textContent, "working");
  assert.equal(elements.currentTranslation.textContent, "en cours");

  const [historyItem] = elements.historyList.children;
  assert.ok(historyItem);
  assert.equal(historyItem.children[0]?.textContent, "00:01.000 - 00:02.500 aligned");
  assert.equal(historyItem.children[1]?.textContent, "hello");
  assert.equal(historyItem.children[2]?.textContent, "bonjour");
  assert.equal(historyItem.children[1]?.attributes.get("contenteditable"), "plaintext-only");
  assert.equal(historyItem.children[2]?.attributes.get("contenteditable"), "plaintext-only");
  assert.equal(historyItem.children[1]?.attributes.get("lang"), "zh");
  assert.equal(historyItem.children[2]?.attributes.get("lang"), "fr");
  assert.ok(historyItem.className.split(/\s+/).includes("is-latest"));
});

test("renders complete long caption text and leaves visual clipping to layout", () => {
  const document = new SubtitleDocument();
  const stableText = "一二三四五六七八九十甲乙丙丁戊己庚辛。后续文本";
  const currentText = "当前文本也可能很长。最后显示";
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, stableText, { startMs: 0, endMs: 2300 })],
    partial: partialSegment(currentText),
  });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });

  assert.equal(elements.currentSource.textContent, currentText);
  const [historyItem] = elements.historyList.children;
  assert.equal(historyItem?.children[1]?.textContent, stableText);
});

test("anchors compact caption text to the latest visible tail", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "previous text", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("current text"),
  });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "previous translation" });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "current translation" });
  const elements = createElements();
  elements.currentSource.scrollHeight = 160;
  elements.currentTranslation.scrollHeight = 96;
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });

  assert.equal(elements.currentSource.scrollTop, 160);
  assert.equal(elements.currentTranslation.scrollTop, 96);
});

test("updates history when translation visibility changes and scrolls visible history", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "source", { startMs: 0, endMs: 900 })],
    partial: null,
  });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "target" });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: true });
  document.setTranslationEnabled(false);
  view.render(document, { historyVisible: true });

  const [historyItem] = elements.historyList.children;
  assert.ok(historyItem);
  assert.equal(historyItem.children[2]?.textContent, "");
  assert.deepEqual(historyItem.scrollCalls, [{ behavior: "smooth", block: "end" }]);
});

test("preserves user-edited history text when timing updates rerender the same line", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "source", { startMs: 0, endMs: 900 })],
    partial: null,
  });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });
  const [historyItem] = elements.historyList.children;
  const source = historyItem?.children[1];
  assert.ok(source);
  source.textContent = "edited source";
  source.dispatch("input", {});

  document.applyEvent({
    type: "transcript_timing_update",
    source_segment_id: "seg_000001",
    start_ms: 100,
    end_ms: 1200,
    timing_status: "aligned",
  });
  view.render(document, { historyVisible: false });

  assert.equal(historyItem.children[0]?.textContent, "00:00.100 - 00:01.200 aligned");
  assert.equal(source.textContent, "edited source");
});

test("announces only stabilized lines through the live region with language tags", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "hello", { startMs: 0, endMs: 1000 })],
    partial: partialSegment("typing"),
  });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "nihao" });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false, translationLanguage: "Japanese" });
  view.render(document, { historyVisible: false, translationLanguage: "Japanese" });

  assert.equal(elements.announcer.children.length, 1);
  const entry = elements.announcer.children[0];
  assert.equal(entry?.attributes.get("dir"), "auto");
  const [sourceSpan, translationSpan] = entry?.children ?? [];
  assert.equal(sourceSpan?.textContent, "hello");
  assert.equal(sourceSpan?.attributes.get("lang"), "zh");
  assert.equal(translationSpan?.textContent, " — nihao");
  assert.equal(translationSpan?.attributes.get("lang"), "ja");
  assert.equal(elements.currentSource.attributes.get("lang"), "zh");
  assert.equal(elements.currentTranslation.attributes.get("lang"), "ja");
  assert.equal(elements.currentSource.textContent, "typing");
});

test("announces a stable translation that arrives after the source line", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "hello", { startMs: 0, endMs: 1000 })],
    partial: null,
  });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false, translationLanguage: "English" });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "hello translation" });
  view.render(document, { historyVisible: false, translationLanguage: "English" });
  view.render(document, { historyVisible: false, translationLanguage: "English" });

  assert.equal(elements.announcer.children.length, 2);
  assert.equal(elements.announcer.children[0]?.children[0]?.textContent, "hello");
  assert.equal(elements.announcer.children[1]?.children[0]?.textContent, "hello translation");
  assert.equal(elements.announcer.children[1]?.children[0]?.attributes.get("lang"), "en");
});

test("caps the announce log at 40 lines, dropping the oldest", () => {
  const document = new SubtitleDocument({ translationEnabled: false });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  for (let index = 1; index <= 45; index += 1) {
    document.applyEvent({
      type: "transcript_update",
      revision: index,
      stable_base: index - 1,
      stable_count: index,
      stable_appends: [stableSegment(index, `line ${index}`, { startMs: index * 1000, endMs: index * 1000 + 500 })],
      partial: null,
    });
    view.render(document, { historyVisible: false });
  }

  assert.equal(elements.announcer.children.length, 40);
  assert.equal(elements.announcer.children[0]?.children[0]?.textContent, "line 6");
  assert.equal(elements.announcer.children.at(-1)?.children[0]?.textContent, "line 45");
});

test("does not re-announce stable lines when the list is rebuilt on final", () => {
  const document = new SubtitleDocument({ translationEnabled: false });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "one", { startMs: 0, endMs: 500 }),
      stableSegment(2, "two", { startMs: 500, endMs: 1000 }),
    ],
    partial: null,
  });
  view.render(document, { historyVisible: false });
  assert.equal(elements.announcer.children.length, 2);

  // transcript_final replaces the stable list with fresh objects plus one tail line.
  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    segments: [
      { ...stableSegment(1, "one", { startMs: 0, endMs: 500 }), id: "rebuilt_seg_1" },
      { ...stableSegment(2, "two", { startMs: 500, endMs: 1000 }), id: "rebuilt_seg_2" },
      stableSegment(3, "three", { startMs: 1000, endMs: 1500 }),
    ],
  });
  view.render(document, { historyVisible: false });

  assert.equal(elements.announcer.children.length, 3);
  assert.equal(elements.announcer.children[2]?.children[0]?.textContent, "three");
});

test("keeps inline history edits when final rebuild reassigns ids but keeps indexes", () => {
  const document = new SubtitleDocument({ translationEnabled: false });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [stableSegment(1, "one", { startMs: 0, endMs: 500 })],
    partial: null,
  });
  view.render(document, { historyVisible: true });

  const source = elements.historyList.children[0]?.children[1];
  assert.ok(source);
  source.textContent = "ONE edited";
  source.dispatch("input", {});

  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    segments: [{ ...stableSegment(1, "one", { startMs: 0, endMs: 500 }), id: "rebuilt_seg_1" }],
  });
  view.render(document, { historyVisible: true });

  assert.equal(elements.historyList.children[0]?.children[1]?.textContent, "ONE edited");
  assert.deepEqual(view.collectTranscriptLines(), [{ startMs: 0, text: "ONE edited", translation: null }]);
});

test("collectTranscriptLines reflects inline history edits", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "one", { startMs: 0, endMs: 500 }),
      stableSegment(2, "two", { startMs: 500, endMs: 1000 }),
    ],
    partial: null,
  });
  document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "uno" });
  document.applyEvent({
    type: "translation_status",
    scope: "stable",
    code: "timeout",
    message: "translation failed",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
  });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });

  const firstSource = elements.historyList.children[0]?.children[1];
  assert.ok(firstSource);
  firstSource.textContent = "ONE edited";
  firstSource.dispatch("input", {});

  assert.deepEqual(view.collectTranscriptLines(), [
    { startMs: 0, text: "ONE edited", translation: "uno" },
    { startMs: 500, text: "two", translation: null },
  ]);
});

test("collectTranscriptLines returns the visible grouped translation", () => {
  const document = new SubtitleDocument({ translationEnabled: true });
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 2,
    stable_appends: [
      stableSegment(1, "今天讨论字幕显示问题，", { startMs: 0, endMs: 2000 }),
      stableSegment(2, "并且保持翻译输入完整。", { startMs: 2000, endMs: 3800 }),
    ],
    partial: null,
  });
  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "We discuss subtitle display while preserving translation context.",
  });
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: true });

  assert.deepEqual(view.collectTranscriptLines(), [
    { startMs: 0, text: "今天讨论字幕显示问题，", translation: null },
    {
      startMs: 2000,
      text: "并且保持翻译输入完整。",
      translation: "We discuss subtitle display while preserving translation context.",
    },
  ]);
});

function createElements(): Record<"currentSource" | "currentTranslation" | "historyList" | "announcer", FakeElement> {
  return {
    currentSource: new FakeElement(),
    currentTranslation: new FakeElement(),
    historyList: new FakeElement(),
    announcer: new FakeElement(),
  };
}

function captionViewElements(
  elements: ReturnType<typeof createElements>,
): ConstructorParameters<typeof CaptionView>[0] {
  return {
    currentSource: asDomElement(elements.currentSource),
    currentTranslation: asDomElement(elements.currentTranslation),
    historyList: asDomElement(elements.historyList),
    announcer: asDomElement(elements.announcer),
  };
}

function stableSegment(
  index: number,
  text: string,
  { endMs, startMs, timingStatus }: { endMs?: number; startMs?: number; timingStatus?: string } = {},
): Record<string, unknown> {
  return {
    id: `seg_${String(index).padStart(6, "0")}`,
    index,
    start_ms: startMs,
    end_ms: endMs,
    text,
    language: "Chinese",
    timing_status: timingStatus,
  };
}

function partialSegment(text: string): Record<string, unknown> {
  return {
    text,
    language: "Chinese",
  };
}
