import assert from "node:assert/strict";
import test from "node:test";

import { CaptionView } from "./caption-view.js";
import { SubtitleDocument } from "./subtitle-document.js";
import { clearDomGlobals } from "./test-browser-globals.fixture.js";
import {
  asDomElement,
  FakeDocument,
  FakeElement,
  installFakeDocument,
  installFakeElementConstructors,
  installedFakeDocument,
} from "./test-dom.fixture.js";

test.beforeEach(() => {
  installFakeDocument(new FakeDocument());
  installFakeElementConstructors();
});

test.afterEach(() => {
  clearDomGlobals();
  Reflect.deleteProperty(globalThis, "ResizeObserver");
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
  assert.equal(historyItem.children[1]?.attributes.get("contenteditable"), undefined);
  assert.equal(historyItem.children[2]?.attributes.get("contenteditable"), undefined);
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

test("does not force caption scroll when rendered caption text is unchanged", () => {
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current text"),
  });
  const elements = createElements();
  elements.currentSource.scrollHeight = 160;
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });
  elements.currentSource.scrollHeight = 240;
  view.render(document, { historyVisible: false });

  assert.equal(elements.currentSource.scrollTop, 160);
});

test("reanchors current caption tails when the caption layout changes", () => {
  const resizeObservers = installFakeResizeObserver();
  const document = new SubtitleDocument();
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 0,
    stable_appends: [],
    partial: partialSegment("current text"),
  });
  document.applyEvent({ type: "translation_preview", source_revision: 1, text: "current translation" });
  const elements = createElements();
  elements.currentSource.scrollHeight = 160;
  elements.currentTranslation.scrollHeight = 96;
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });
  elements.currentSource.scrollHeight = 240;
  elements.currentTranslation.scrollHeight = 180;
  resizeObservers[0]?.trigger();

  assert.equal(elements.currentSource.scrollTop, 240);
  assert.equal(elements.currentTranslation.scrollTop, 180);
});

test("skips stable history reads when the stable render version is unchanged", () => {
  let stableReads = 0;
  const line = {
    id: "seg_000001",
    index: 1,
    startMs: 0,
    endMs: 1000,
    text: "stable",
    language: "Chinese",
    sourceRevision: 1,
    timingStatus: null,
    translation: null,
    translationStatus: null,
    translationMessage: null,
  };
  const document = {
    stableRenderVersion: 1,
    translationEnabled: true,
    get stableLines() {
      stableReads += 1;
      return [line];
    },
    window: () => ({ current: null }),
  } as unknown as SubtitleDocument;
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: false });
  view.render(document, { historyVisible: false });

  assert.equal(stableReads, 1);
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
  const [initialHistoryItem] = elements.historyList.children;
  assert.ok(initialHistoryItem);
  initialHistoryItem.scrollCalls = [];
  document.setTranslationEnabled(false);
  view.render(document, { historyVisible: true });

  const [historyItem] = elements.historyList.children;
  assert.ok(historyItem);
  assert.equal(historyItem.children[2]?.textContent, "");
  assert.deepEqual(historyItem.scrollCalls, [{ behavior: "smooth", block: "end" }]);
});

test("updates read-only history text when timing updates rerender the same line", () => {
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

  document.applyEvent({
    type: "transcript_timing_update",
    source_segment_id: "seg_000001",
    start_ms: 100,
    end_ms: 1200,
    timing_status: "aligned",
  });
  view.render(document, { historyVisible: false });

  assert.equal(historyItem.children[0]?.textContent, "00:00.100 - 00:01.200 aligned");
  assert.equal(source.textContent, "source");
});

test("tags visible caption languages without maintaining a hidden announce log", () => {
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

  assert.equal(elements.currentSource.attributes.get("lang"), "zh");
  assert.equal(elements.currentTranslation.attributes.get("lang"), "ja");
  assert.equal(elements.currentSource.textContent, "typing");
});

test("keeps rendered history stable when the list is rebuilt on final", () => {
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
  assert.equal(elements.historyList.children.length, 2);

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

  assert.equal(elements.historyList.children.length, 3);
  assert.equal(elements.historyList.children[2]?.children[1]?.textContent, "three");
});

test("rerenders history from document state when final rebuild reassigns ids but keeps indexes", () => {
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

  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    segments: [{ ...stableSegment(1, "one", { startMs: 0, endMs: 500 }), id: "rebuilt_seg_1" }],
  });
  view.render(document, { historyVisible: true });

  assert.equal(elements.historyList.children[0]?.children[1]?.textContent, "one");
  assert.deepEqual(document.exportLines(), [
    {
      id: "rebuilt_seg_1",
      index: 1,
      startMs: 0,
      endMs: 500,
      text: "one",
      language: "Chinese",
      sourceRevision: 2,
      timingStatus: null,
      translation: null,
      translationStatus: null,
      translationMessage: null,
    },
  ]);
});

test("virtualizes long history while keeping the latest rendered at the tail", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));

  view.render(document, { historyVisible: true });

  const historyItems = historyItemsIn(elements.historyList);
  assert.ok(historyItems.length < 40);
  assert.equal(historyItems.at(-1)?.children[1]?.textContent, "line 120");
  assert.ok(historyItems.at(-1)?.className.split(/\s+/).includes("is-latest"));
  assert.ok(Number.parseFloat(elements.historyList.children[0]?.styleValues.get("height") || "0") > 0);
});

test("updates virtualized history rows when the user scrolls away from the latest tail", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });

  elements.historyList.scrollTop = 0;
  elements.historyList.dispatch("scroll", {});

  const historyItems = historyItemsIn(elements.historyList);
  assert.equal(historyItems[0]?.children[1]?.textContent, "line 1");
  assert.notEqual(historyItems.at(-1)?.children[1]?.textContent, "line 120");
});

test("keeps the virtual history anchor when new lines append while the user reads older history", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 0;
  elements.historyList.dispatch("scroll", {});
  const firstRenderedBeforeAppend = historyItemsIn(elements.historyList)[0]?.children[1]?.textContent;

  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: 120,
    stable_count: 121,
    stable_appends: [stableSegment(121, "line 121", { startMs: 120_000, endMs: 120_500 })],
    partial: null,
  });
  view.render(document, { historyVisible: true });

  const historyItems = historyItemsIn(elements.historyList);
  assert.equal(elements.historyList.scrollTop, 0);
  assert.equal(historyItems[0]?.children[1]?.textContent, firstRenderedBeforeAppend);
  assert.notEqual(historyItems.at(-1)?.children[1]?.textContent, "line 121");
});

test("scrollHistoryToLatest re-renders the latest row after virtualized history was scrolled up", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 0;
  elements.historyList.dispatch("scroll", {});

  view.scrollHistoryToLatest("auto");

  const latest = historyItemsIn(elements.historyList).at(-1);
  assert.equal(latest?.children[1]?.textContent, "line 120");
  assert.deepEqual(latest?.scrollCalls, [{ behavior: "auto", block: "end" }]);
});

test("clears virtual spacers when history shrinks below the virtualization threshold", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  assert.ok(elements.historyList.children.some((child) => child.className === "history-spacer"));

  document.applyEvent({
    type: "transcript_final",
    revision: 2,
    segments: [stableSegment(1, "final only", { startMs: 0, endMs: 500 })],
  });
  view.render(document, { historyVisible: true });

  assert.equal(elements.historyList.children.length, 1);
  assert.equal(elements.historyList.children[0]?.className, "history-item is-latest");
  assert.equal(elements.historyList.children[0]?.children[1]?.textContent, "final only");
});

test("keeps full history scroll position when new lines append while the user reads older history", () => {
  const document = longHistoryDocument({ lineCount: 20 });
  const elements = createElements();
  elements.historyList.clientHeight = 120;
  elements.historyList.scrollHeight = 1000;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 0;
  elements.historyList.dispatch("wheel", {});

  appendStableLine(document, 21);
  view.render(document, { historyVisible: true });

  const latest = elements.historyList.children.at(-1);
  assert.equal(elements.historyList.scrollTop, 0);
  assert.equal(latest?.children[1]?.textContent, "line 21");
  assert.deepEqual(latest?.scrollCalls, []);
});

test("keeps latest rendered when pinned full history crosses into virtualization", () => {
  const document = longHistoryDocument({ lineCount: 80 });
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  elements.historyList.scrollHeight = 5120;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = elements.historyList.scrollHeight - elements.historyList.clientHeight;

  appendStableLine(document, 81);
  view.render(document, { historyVisible: true });

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 81");
});

test("keeps the full history anchor when scrolled-away history crosses into virtualization", () => {
  const prototype = FakeElement.prototype as FakeElement & {
    getBoundingClientRect?: (this: FakeElement) => DOMRect;
  };
  const originalGetBoundingClientRect = prototype.getBoundingClientRect;
  prototype.getBoundingClientRect = function getBoundingClientRect(this: FakeElement): DOMRect {
    if (this.className.split(/\s+/).includes("history-item")) {
      return { height: 20 } as DOMRect;
    }
    return { height: 0 } as DOMRect;
  };
  try {
    const document = longHistoryDocument({ lineCount: 80 });
    const elements = createElements();
    elements.historyList.clientHeight = 256;
    const view = new CaptionView(captionViewElements(elements));
    view.render(document, { historyVisible: true });
    elements.historyList.scrollHeight = 80 * 29;
    elements.historyList.scrollTop = 30 * 29;
    elements.historyList.dispatch("wheel", {});
    const anchoredText = elements.historyList.children[30]?.children[1]?.textContent;

    appendStableLine(document, 81);
    view.render(document, { historyVisible: true });

    assert.equal(anchoredText, "line 31");
    assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
    assert.ok(elements.historyList.scrollTop > 1500);
    assert.ok(historyItemsIn(elements.historyList).some((item) => item.children[1]?.textContent === anchoredText));
  } finally {
    if (originalGetBoundingClientRect) {
      prototype.getBoundingClientRect = originalGetBoundingClientRect;
    } else {
      Reflect.deleteProperty(prototype, "getBoundingClientRect");
    }
  }
});

test("keeps full history pinned when the latest row content changes", () => {
  const document = longHistoryDocument({ lineCount: 20, translationEnabled: true });
  const elements = createElements();
  elements.historyList.clientHeight = 120;
  elements.historyList.scrollHeight = 1000;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 880;
  const latest = elements.historyList.children.at(-1);
  assert.ok(latest);
  latest.scrollCalls = [];

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000020",
    text: "translated latest line",
  });
  view.render(document, { historyVisible: true, translationLanguage: "English" });

  assert.equal(latest.children[2]?.textContent, "translated latest line");
  assert.deepEqual(latest.scrollCalls, [{ behavior: "smooth", block: "end" }]);
});

test("keeps full history pinned while a previous latest scroll is still pending", () => {
  const document = longHistoryDocument({ lineCount: 20, translationEnabled: true });
  const elements = createElements();
  elements.historyList.clientHeight = 120;
  elements.historyList.scrollHeight = 1000;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 880;
  const latest = elements.historyList.children.at(-1);
  assert.ok(latest);
  latest.scrollCalls = [];

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000020",
    text: "translated latest line",
  });
  view.render(document, { historyVisible: true, translationLanguage: "English" });
  elements.historyList.scrollHeight = 1200;

  document.applyEvent({
    type: "transcript_timing_update",
    source_segment_id: "seg_000020",
    start_ms: 19_000,
    end_ms: 20_000,
    timing_status: "aligned",
  });
  view.render(document, { historyVisible: true, translationLanguage: "English" });

  assert.deepEqual(latest.scrollCalls, [
    { behavior: "smooth", block: "end" },
    { behavior: "smooth", block: "end" },
  ]);
});

test("fills full history rows when grouped translation exits virtualization", () => {
  const document = longHistoryDocument({ lineCount: 81, translationEnabled: true });
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000081",
    source_segment_index: 81,
    source_segment_ids: ["seg_000080", "seg_000081"],
    source_segment_indices: [80, 81],
    text: "lines 80 and 81 translated",
  });
  view.render(document, { historyVisible: true });

  assert.equal(elements.historyList.className, "");
  assert.equal(elements.historyList.children.length, 80);
  assert.equal(elements.historyList.children[0]?.children[1]?.textContent, "line 1");
  assert.equal(elements.historyList.children[78]?.children[1]?.textContent, "line 79");
  assert.equal(elements.historyList.children[79]?.children[1]?.textContent, "line 80 line 81");
  assert.equal(elements.historyList.children[79]?.children[2]?.textContent, "lines 80 and 81 translated");
});

test("keeps the virtual history anchor when grouped translation exits virtualization above the viewport", () => {
  const document = longHistoryDocument({ lineCount: 81, translationEnabled: true });
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 1000;
  elements.historyList.dispatch("scroll", {});
  const scrollTopBeforeGrouping = elements.historyList.scrollTop;

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "lines 1 and 2 translated",
  });
  view.render(document, { historyVisible: true });

  assert.equal(elements.historyList.className, "");
  assert.ok(elements.historyList.scrollTop < scrollTopBeforeGrouping);
  assert.ok(elements.historyList.scrollTop > 0);
});

test("does not restore tail pin after scrolled-away virtual history exits virtualization", () => {
  const document = longHistoryDocument({ lineCount: 81, translationEnabled: true });
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  elements.historyList.scrollTop = 1000;
  elements.historyList.dispatch("scroll", {});

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "lines 1 and 2 translated",
  });
  view.render(document, { historyVisible: true });
  elements.historyList.scrollHeight = 80 * 73;
  const scrollTopAfterGrouping = elements.historyList.scrollTop;
  const latest = elements.historyList.children.at(-1);
  assert.ok(latest);
  latest.scrollCalls = [];

  document.applyEvent({
    type: "transcript_timing_update",
    source_segment_id: "seg_000080",
    start_ms: 79_000,
    end_ms: 79_500,
    timing_status: "aligned",
  });
  view.render(document, { historyVisible: true });

  assert.equal(elements.historyList.className, "");
  assert.equal(elements.historyList.scrollTop, scrollTopAfterGrouping);
  assert.deepEqual(latest.scrollCalls, []);
});

test("keeps the virtual scroll anchor when history content above the viewport changes height", () => {
  const prototype = FakeElement.prototype as FakeElement & {
    getBoundingClientRect?: (this: FakeElement) => DOMRect;
  };
  const originalGetBoundingClientRect = prototype.getBoundingClientRect;
  prototype.getBoundingClientRect = function getBoundingClientRect(this: FakeElement): DOMRect {
    if (this.className.split(/\s+/).includes("history-item")) {
      const sourceText = this.children[1]?.textContent || "";
      const translationText = this.children[2]?.textContent || "";
      const height = sourceText === "line 1" && !translationText ? 20 : 64;
      return { height } as DOMRect;
    }
    return { height: 0 } as DOMRect;
  };
  try {
    const document = longHistoryDocument({ translationEnabled: true });
    const elements = createElements();
    elements.historyList.clientHeight = 256;
    const view = new CaptionView(captionViewElements(elements));
    view.render(document, { historyVisible: true });
    elements.historyList.scrollTop = 0;
    elements.historyList.dispatch("scroll", {});
    elements.historyList.scrollTop = 1000;
    elements.historyList.dispatch("scroll", {});
    const scrollTopBeforeTranslation = elements.historyList.scrollTop;
    const firstRenderedBeforeTranslation = historyItemsIn(elements.historyList)[0]?.children[1]?.textContent;

    document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000001", text: "translated line 1" });
    view.render(document, { historyVisible: true });

    assert.equal(historyItemsIn(elements.historyList)[0]?.children[1]?.textContent, firstRenderedBeforeTranslation);
    assert.ok(elements.historyList.scrollTop > scrollTopBeforeTranslation);
  } finally {
    if (originalGetBoundingClientRect) {
      prototype.getBoundingClientRect = originalGetBoundingClientRect;
    } else {
      Reflect.deleteProperty(prototype, "getBoundingClientRect");
    }
  }
});

test("does not materialize virtual history on pointerdown", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  const visibleSource = historyItemsIn(elements.historyList)[0]?.children[1];
  assert.ok(visibleSource);

  elements.historyList.dispatch("pointerdown", { target: visibleSource });

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.ok(historyItemsIn(elements.historyList).length < 40);
  assert.equal(visibleSource.parentElement?.parentElement, elements.historyList);
});

test("defers virtual history updates while a pointer selection is active", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const visibleSource = historyItemsIn(elements.historyList)[0]?.children[1];
  assert.ok(visibleSource);

  elements.historyList.dispatch("pointerdown", { target: visibleSource });
  appendStableLine(document, 121);
  view.render(document, { historyVisible: true });

  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");
  assert.ok(historyItemsIn(elements.historyList).length < 40);

  elements.historyList.dispatch("pointerup", {});

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 121");
  assert.ok(historyItemsIn(elements.historyList).length < 40);
});

test("does not flush deferred history on selectionchange before pointer release", () => {
  const fakeDocument = installedFakeDocument();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const visibleSource = historyItemsIn(elements.historyList)[0]?.children[1];
  assert.ok(visibleSource);

  elements.historyList.dispatch("pointerdown", { target: visibleSource });
  appendStableLine(document, 121);
  view.render(document, { historyVisible: true });
  fakeDocument.selection = fakeSelection({ anchorNode: visibleSource, focusNode: visibleSource, isCollapsed: true });
  fakeDocument.dispatch("selectionchange", {});

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");

  elements.historyList.dispatch("pointerup", {});

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 121");
  assert.ok(historyItemsIn(elements.historyList).length < 40);
});

test("keeps deferred history updates while selected text is active", () => {
  const fakeDocument = installedFakeDocument();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const visibleSource = historyItemsIn(elements.historyList)[0]?.children[1];
  assert.ok(visibleSource);

  elements.historyList.dispatch("pointerdown", { target: visibleSource });
  fakeDocument.selection = fakeSelection({ anchorNode: visibleSource, focusNode: elements.historyList });
  elements.historyList.dispatch("pointerup", {});
  appendStableLine(document, 121);
  view.render(document, { historyVisible: true });

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");
  assert.ok(historyItemsIn(elements.historyList).length < 40);

  fakeDocument.selection = fakeSelection({ anchorNode: visibleSource, focusNode: visibleSource, isCollapsed: true });
  fakeDocument.dispatch("selectionchange", {});

  assert.ok(elements.historyList.className.split(/\s+/).includes("is-virtualized"));
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 121");
  assert.ok(historyItemsIn(elements.historyList).length < 40);
});

test("defers hidden history updates while selected text is active", () => {
  const fakeDocument = installedFakeDocument();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const visibleSource = historyItemsIn(elements.historyList)[0]?.children[1];
  assert.ok(visibleSource);

  elements.historyList.dispatch("pointerdown", { target: visibleSource });
  fakeDocument.selection = fakeSelection({ anchorNode: visibleSource, focusNode: elements.historyList });
  elements.historyList.dispatch("pointerup", {});
  appendStableLine(document, 121);
  view.render(document, { historyVisible: false });

  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");

  fakeDocument.selection = fakeSelection({ anchorNode: visibleSource, focusNode: visibleSource, isCollapsed: true });
  fakeDocument.dispatch("selectionchange", {});

  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 121");
});

test("does not rerender the virtual window on scroll while selection is active", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const firstRenderedBeforeScroll = historyItemsIn(elements.historyList)[0];
  assert.ok(firstRenderedBeforeScroll);
  const firstRenderedTextBeforeScroll = firstRenderedBeforeScroll.children[1]?.textContent;

  elements.historyList.dispatch("pointerdown", { target: firstRenderedBeforeScroll.children[1] });
  elements.historyList.scrollTop = 1000;
  elements.historyList.dispatch("scroll", {});

  assert.equal(historyItemsIn(elements.historyList)[0], firstRenderedBeforeScroll);
  assert.equal(historyItemsIn(elements.historyList)[0]?.children[1]?.textContent, firstRenderedTextBeforeScroll);
});

test("refreshes the deferred virtual window scroll after pointer release", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const firstRenderedTextBeforeScroll = historyItemsIn(elements.historyList)[0]?.children[1]?.textContent;
  const firstRendered = historyItemsIn(elements.historyList)[0];
  assert.ok(firstRendered);

  elements.historyList.dispatch("pointerdown", { target: firstRendered.children[1] });
  elements.historyList.scrollTop = 1000;
  elements.historyList.dispatch("scroll", {});
  elements.historyList.dispatch("pointerup", {});

  assert.notEqual(historyItemsIn(elements.historyList)[0]?.children[1]?.textContent, firstRenderedTextBeforeScroll);
});

test("does not rerender the virtual window on resize while selection is active", () => {
  const resizeObservers = installFakeResizeObserver();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const firstRenderedBeforeResize = historyItemsIn(elements.historyList)[0];
  assert.ok(firstRenderedBeforeResize);
  const renderedCountBeforeResize = historyItemsIn(elements.historyList).length;

  elements.historyList.dispatch("pointerdown", { target: firstRenderedBeforeResize.children[1] });
  elements.historyList.clientHeight = 640;
  resizeObservers.at(-1)?.trigger();

  assert.equal(historyItemsIn(elements.historyList)[0], firstRenderedBeforeResize);
  assert.equal(historyItemsIn(elements.historyList).length, renderedCountBeforeResize);
});

test("refreshes the deferred virtual window resize after pointer release", () => {
  const resizeObservers = installFakeResizeObserver();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const renderedCountBeforeResize = historyItemsIn(elements.historyList).length;
  const firstRendered = historyItemsIn(elements.historyList)[0];
  assert.ok(firstRendered);

  elements.historyList.dispatch("pointerdown", { target: firstRendered.children[1] });
  elements.historyList.clientHeight = 640;
  resizeObservers.at(-1)?.trigger();
  elements.historyList.dispatch("pointerup", {});

  assert.ok(historyItemsIn(elements.historyList).length > renderedCountBeforeResize);
});

test("deferred virtual resize preserves the same anchor as immediate resize", () => {
  const resizeObservers = installFakeResizeObserver();
  const prototype = FakeElement.prototype as FakeElement & {
    getBoundingClientRect?: (this: FakeElement) => DOMRect;
  };
  const originalGetBoundingClientRect = prototype.getBoundingClientRect;
  prototype.getBoundingClientRect = function getBoundingClientRect(this: FakeElement): DOMRect {
    if (this.className.split(/\s+/).includes("history-item")) {
      return { height: 20 } as DOMRect;
    }
    return { height: 0 } as DOMRect;
  };
  try {
    const immediateDocument = longHistoryDocument();
    const immediateElements = createElements();
    immediateElements.historyList.clientHeight = 256;
    const immediateView = new CaptionView(captionViewElements(immediateElements));
    immediateView.render(immediateDocument, { historyVisible: true });
    immediateElements.historyList.scrollTop = 1000;
    immediateElements.historyList.dispatch("scroll", {});
    immediateElements.historyList.clientHeight = 640;
    resizeObservers.at(-1)?.trigger();
    const immediateFirstTextAfterResize = historyItemsIn(immediateElements.historyList)[0]?.children[1]?.textContent;

    const deferredDocument = longHistoryDocument();
    const deferredElements = createElements();
    deferredElements.historyList.clientHeight = 256;
    const deferredView = new CaptionView(captionViewElements(deferredElements));
    deferredView.render(deferredDocument, { historyVisible: true });
    deferredElements.historyList.scrollTop = 1000;
    deferredElements.historyList.dispatch("scroll", {});
    const deferredFirstRenderedBeforeResize = historyItemsIn(deferredElements.historyList)[0];
    assert.ok(deferredFirstRenderedBeforeResize);

    deferredElements.historyList.dispatch("pointerdown", { target: deferredFirstRenderedBeforeResize.children[1] });
    deferredElements.historyList.clientHeight = 640;
    resizeObservers.at(-1)?.trigger();
    deferredElements.historyList.dispatch("pointerup", {});

    assert.equal(
      historyItemsIn(deferredElements.historyList)[0]?.children[1]?.textContent,
      immediateFirstTextAfterResize,
    );
  } finally {
    if (originalGetBoundingClientRect) {
      prototype.getBoundingClientRect = originalGetBoundingClientRect;
    } else {
      Reflect.deleteProperty(prototype, "getBoundingClientRect");
    }
  }
});

test("does not tail-follow while virtual history selection is active", () => {
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 256;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const latest = historyItemsIn(elements.historyList).at(-1);
  assert.ok(latest);
  latest.scrollCalls = [];

  elements.historyList.dispatch("pointerdown", { target: latest.children[1] });
  view.scrollHistoryToLatest("auto");

  assert.deepEqual(latest.scrollCalls, []);
  assert.equal(historyItemsIn(elements.historyList).at(-1), latest);
});

test("keeps the virtualized history pinned to latest when the latest row grows", () => {
  const prototype = FakeElement.prototype as FakeElement & {
    getBoundingClientRect?: (this: FakeElement) => DOMRect;
  };
  const originalGetBoundingClientRect = prototype.getBoundingClientRect;
  prototype.getBoundingClientRect = function getBoundingClientRect(this: FakeElement): DOMRect {
    if (this.className.split(/\s+/).includes("history-item")) {
      const sourceText = this.children[1]?.textContent || "";
      const translationText = this.children[2]?.textContent || "";
      if (sourceText === "line 120" && !translationText) {
        return { height: 20 } as DOMRect;
      }
      if (sourceText === "line 120" && translationText) {
        return { height: 160 } as DOMRect;
      }
      const height = 64;
      return { height } as DOMRect;
    }
    return { height: 0 } as DOMRect;
  };
  try {
    const document = longHistoryDocument({ translationEnabled: true });
    const elements = createElements();
    elements.historyList.clientHeight = 256;
    const view = new CaptionView(captionViewElements(elements));
    view.render(document, { historyVisible: true });
    const scrollTopBeforeTranslation = elements.historyList.scrollTop;

    document.applyEvent({ type: "translation_stable", source_segment_id: "seg_000120", text: "translated line 120" });
    view.render(document, { historyVisible: true, translationLanguage: "English" });

    const latest = historyItemsIn(elements.historyList).at(-1);
    assert.ok(elements.historyList.scrollTop >= scrollTopBeforeTranslation + 130);
    assert.equal(latest?.children[1]?.textContent, "line 120");
    assert.equal(latest?.children[2]?.textContent, "translated line 120");
  } finally {
    if (originalGetBoundingClientRect) {
      prototype.getBoundingClientRect = originalGetBoundingClientRect;
    } else {
      Reflect.deleteProperty(prototype, "getBoundingClientRect");
    }
  }
});

test("keeps the latest virtual history row rendered when a pinned viewport shrinks", () => {
  const resizeObservers = installFakeResizeObserver();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 1024;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");

  elements.historyList.clientHeight = 128;
  resizeObservers.at(-1)?.trigger();

  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");
});

test("recomputes the virtual history window when the history viewport resizes", () => {
  const resizeObservers = installFakeResizeObserver();
  const document = longHistoryDocument();
  const elements = createElements();
  elements.historyList.clientHeight = 128;
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });
  const initialRenderedCount = historyItemsIn(elements.historyList).length;

  elements.historyList.clientHeight = 640;
  resizeObservers.at(-1)?.trigger();

  assert.ok(historyItemsIn(elements.historyList).length > initialRenderedCount);
  assert.equal(historyItemsIn(elements.historyList).at(-1)?.children[1]?.textContent, "line 120");
});

test("exportLines comes from the document projection, not mutable DOM text", () => {
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
  firstSource.textContent = "accidental DOM mutation";

  assert.deepEqual(
    document.exportLines().map(({ startMs, text, translation }) => ({ startMs, text, translation })),
    [
      { startMs: 0, text: "one", translation: "uno" },
      { startMs: 500, text: "two", translation: null },
    ],
  );
});

test("exportLines returns the grouped translation", () => {
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

  assert.deepEqual(
    document.exportLines().map(({ startMs, text, translation }) => ({ startMs, text, translation })),
    [
      {
        startMs: 0,
        text: "今天讨论字幕显示问题，并且保持翻译输入完整。",
        translation: "We discuss subtitle display while preserving translation context.",
      },
    ],
  );
});

test("grouped translation folds rendered source rows from document state", () => {
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
  const elements = createElements();
  const view = new CaptionView(captionViewElements(elements));
  view.render(document, { historyVisible: true });

  document.applyEvent({
    type: "translation_stable",
    source_segment_id: "seg_000002",
    source_segment_index: 2,
    source_segment_ids: ["seg_000001", "seg_000002"],
    source_segment_indices: [1, 2],
    text: "one two translated",
  });
  view.render(document, { historyVisible: true });

  assert.deepEqual(
    document.exportLines().map(({ startMs, text, translation }) => ({ startMs, text, translation })),
    [{ startMs: 0, text: "one two", translation: "one two translated" }],
  );
});

function createElements(): Record<"currentSource" | "currentTranslation" | "historyList", FakeElement> {
  return {
    currentSource: new FakeElement(),
    currentTranslation: new FakeElement(),
    historyList: new FakeElement(),
  };
}

function captionViewElements(
  elements: ReturnType<typeof createElements>,
): ConstructorParameters<typeof CaptionView>[0] {
  return {
    currentSource: asDomElement(elements.currentSource),
    currentTranslation: asDomElement(elements.currentTranslation),
    historyList: asDomElement(elements.historyList),
  };
}

function fakeSelection({
  anchorNode,
  focusNode,
  isCollapsed = false,
}: {
  anchorNode: FakeElement;
  focusNode: FakeElement;
  isCollapsed?: boolean;
}): Selection {
  return {
    anchorNode: asDomElement(anchorNode),
    focusNode: asDomElement(focusNode),
    getRangeAt: () =>
      ({
        commonAncestorContainer: asDomElement(anchorNode),
        intersectsNode: (node: Node) => node === asDomElement(anchorNode) || node === asDomElement(focusNode),
      }) as unknown as Range,
    isCollapsed,
    rangeCount: 1,
    toString: () => (isCollapsed ? "" : "selected text"),
  } as unknown as Selection;
}

function historyItemsIn(historyList: FakeElement): FakeElement[] {
  return historyList.children.filter((child) => child.className.split(/\s+/).includes("history-item"));
}

function longHistoryDocument({
  lineCount = 120,
  translationEnabled = false,
}: {
  lineCount?: number;
  translationEnabled?: boolean;
} = {}): SubtitleDocument {
  const document = new SubtitleDocument({ translationEnabled });
  document.applyEvent({
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: lineCount,
    stable_appends: stableSegmentRange(lineCount),
    partial: null,
  });
  return document;
}

function stableSegmentRange(lineCount: number): Record<string, unknown>[] {
  return Array.from({ length: lineCount }, (_item, index) =>
    stableSegment(index + 1, `line ${index + 1}`, {
      startMs: index * 1000,
      endMs: index * 1000 + 500,
    }),
  );
}

function appendStableLine(document: SubtitleDocument, index: number): void {
  document.applyEvent({
    type: "transcript_update",
    revision: 2,
    stable_base: index - 1,
    stable_count: index,
    stable_appends: [
      stableSegment(index, `line ${index}`, { startMs: (index - 1) * 1000, endMs: (index - 1) * 1000 + 500 }),
    ],
    partial: null,
  });
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

function installFakeResizeObserver(): FakeResizeObserver[] {
  const observers: FakeResizeObserver[] = [];
  class TestResizeObserver extends FakeResizeObserver {
    constructor(callback: ResizeObserverCallback) {
      super(callback);
      observers.push(this);
    }
  }
  Object.defineProperty(globalThis, "ResizeObserver", {
    configurable: true,
    value: TestResizeObserver,
    writable: true,
  });
  return observers;
}

class FakeResizeObserver {
  private readonly observed = new Set<Element>();

  constructor(private readonly callback: ResizeObserverCallback) {}

  observe(target: Element): void {
    this.observed.add(target);
  }

  trigger(): void {
    const entries = [...this.observed].map((target) => ({ target }) as ResizeObserverEntry);
    this.callback(entries, this as unknown as ResizeObserver);
  }
}
