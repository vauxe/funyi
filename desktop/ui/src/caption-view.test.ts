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

  view.render(document, { historyVisible: false });

  assert.equal(elements.previousSource.textContent, "hello");
  assert.equal(elements.previousTranslation.textContent, "bonjour");
  assert.equal(elements.currentSource.textContent, "working");
  assert.equal(elements.currentTranslation.textContent, "en cours");

  const [historyItem] = elements.historyList.children;
  assert.ok(historyItem);
  assert.equal(historyItem.children[0]?.textContent, "00:01.000 - 00:02.500 aligned");
  assert.equal(historyItem.children[1]?.textContent, "hello");
  assert.equal(historyItem.children[2]?.textContent, "bonjour");
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

  assert.equal(elements.previousSource.textContent, stableText);
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
  elements.previousSource.scrollHeight = 120;
  elements.previousTranslation.scrollHeight = 80;
  elements.currentSource.scrollHeight = 160;
  elements.currentTranslation.scrollHeight = 96;
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });

  assert.equal(elements.previousSource.scrollTop, 120);
  assert.equal(elements.previousTranslation.scrollTop, 80);
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

function createElements(): Record<
  "previousSource" | "previousTranslation" | "currentSource" | "currentTranslation" | "historyList",
  FakeElement
> {
  return {
    previousSource: new FakeElement(),
    previousTranslation: new FakeElement(),
    currentSource: new FakeElement(),
    currentTranslation: new FakeElement(),
    historyList: new FakeElement(),
  };
}

function captionViewElements(
  elements: ReturnType<typeof createElements>,
): ConstructorParameters<typeof CaptionView>[0] {
  return {
    previousSource: asDomElement(elements.previousSource),
    previousTranslation: asDomElement(elements.previousTranslation),
    currentSource: asDomElement(elements.currentSource),
    currentTranslation: asDomElement(elements.currentTranslation),
    historyList: asDomElement(elements.historyList),
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
