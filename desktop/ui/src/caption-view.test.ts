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

  unobserve(target: Element): void {
    this.observed.delete(target);
  }

  disconnect(): void {
    this.observed.clear();
  }

  trigger(): void {
    const entries = [...this.observed].map((target) => ({ target }) as ResizeObserverEntry);
    this.callback(entries, this as unknown as ResizeObserver);
  }
}
