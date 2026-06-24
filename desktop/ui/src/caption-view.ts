import { languageTag } from "./languages.js";
import { isInteger } from "./runtime-guards.js";
import type { SubtitleDocument, SubtitleLine } from "./subtitle-document.js";
import { formatClock } from "./time-format.js";

const HISTORY_VIRTUALIZATION_MIN_LINES = 80;
const HISTORY_VIRTUAL_OVERSCAN_ROWS = 8;
const HISTORY_ROW_ESTIMATE_PX = 64;
const HISTORY_ROW_GAP_PX = 9;
const HISTORY_DEFAULT_VIEWPORT_PX = HISTORY_ROW_ESTIMATE_PX * 8;

interface CaptionViewElements {
  currentSource: HTMLElement;
  currentTranslation: HTMLElement;
  historyList: HTMLElement;
}

interface VirtualHistoryState {
  readonly lines: readonly SubtitleLine[];
  readonly translationEnabled: boolean;
  readonly translationLanguage: string;
}

interface VirtualScrollAnchor {
  readonly identity: string;
  readonly fallbackIndex: number;
  readonly offsetPx: number;
}

interface DeferredHistoryRender {
  readonly document: SubtitleDocument;
  readonly historyVisible: boolean;
  readonly translationLanguage: string;
}

export class CaptionView {
  private renderedHistoryLines: readonly SubtitleLine[] = [];
  private renderedHistoryTranslationEnabled: boolean | null = null;
  private renderedHistoryTranslationLanguage = "";
  private renderedHistoryVersion: number | null = null;
  private readonly historyItemHeightByKey = new Map<string, number>();
  private deferredHistoryRender: DeferredHistoryRender | null = null;
  private deferredVirtualHistoryAnchor: VirtualScrollAnchor | null = null;
  private deferredVirtualHistoryAnchorLatest = false;
  private deferredVirtualHistoryWindowRender = false;
  private historyAutoScrollToLatestPending = false;
  private historyPinnedToLatest = false;
  private historySelectionPointerActive = false;
  private virtualHistory: VirtualHistoryState | null = null;
  private virtualHistoryPinnedToLatest = false;

  constructor(private readonly elements: CaptionViewElements) {
    observeCurrentCaptionLayout(elements.currentSource, elements.currentTranslation);
    observeHistoryListLayout(elements.historyList, () => this.handleHistoryLayoutChange());
    elements.historyList.addEventListener("scroll", () => this.handleHistoryScroll());
    elements.historyList.addEventListener("wheel", () => this.handleHistoryUserScrollIntent());
    elements.historyList.addEventListener("touchmove", () => this.handleHistoryUserScrollIntent());
    elements.historyList.addEventListener("pointerdown", (event) => this.handleHistoryPointerDown(event));
    elements.historyList.addEventListener("pointerup", () => this.finishHistorySelectionPointer());
    elements.historyList.addEventListener("pointercancel", () => this.finishHistorySelectionPointer());
    if (typeof document !== "undefined" && typeof document.addEventListener === "function") {
      document.addEventListener("selectionchange", () => this.releaseHistorySelectionIfInactive());
    }
    if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
      window.addEventListener("pointerup", () => this.finishHistorySelectionPointer());
      window.addEventListener("pointercancel", () => this.finishHistorySelectionPointer());
      window.addEventListener("blur", () => this.cancelHistorySelectionPointer());
    }
  }

  render(
    document: SubtitleDocument,
    { historyVisible, translationLanguage = "" }: { historyVisible: boolean; translationLanguage?: string },
  ): void {
    const windowState = document.window();
    renderCaptionLine(
      windowState.current,
      this.elements.currentSource,
      this.elements.currentTranslation,
      translationLanguage,
    );
    this.renderHistory(document, historyVisible, translationLanguage);
  }

  reset(): void {
    this.elements.historyList.replaceChildren();
    this.elements.historyList.classList.remove("is-virtualized");
    this.renderedHistoryLines = [];
    this.renderedHistoryTranslationEnabled = null;
    this.renderedHistoryTranslationLanguage = "";
    this.renderedHistoryVersion = null;
    this.deferredHistoryRender = null;
    this.deferredVirtualHistoryAnchor = null;
    this.deferredVirtualHistoryAnchorLatest = false;
    this.deferredVirtualHistoryWindowRender = false;
    this.historyItemHeightByKey.clear();
    this.historyAutoScrollToLatestPending = false;
    this.historyPinnedToLatest = false;
    this.historySelectionPointerActive = false;
    this.virtualHistory = null;
    this.virtualHistoryPinnedToLatest = false;
  }

  scrollHistoryToLatest(behavior: ScrollBehavior): void {
    if (this.historySelectionIsActive()) {
      return;
    }
    this.historyPinnedToLatest = true;
    this.historyAutoScrollToLatestPending = true;
    if (this.virtualHistory) {
      this.renderVirtualHistoryWindow({ anchorLatest: true });
    }
    const target = latestHistoryItem(this.elements.historyList);
    if (typeof HTMLElement === "undefined" || !(target instanceof HTMLElement)) {
      return;
    }
    const scroll = (): void => target.scrollIntoView({ behavior, block: "end" });
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(scroll);
      return;
    }
    scroll();
  }

  releaseHistorySelection(): void {
    this.flushDeferredHistoryRenderIfInactive();
  }

  private finishHistorySelectionPointer(): void {
    this.historySelectionPointerActive = false;
    this.scheduleHistorySelectionReleaseCheck();
  }

  private cancelHistorySelectionPointer(): void {
    this.historySelectionPointerActive = false;
    this.releaseHistorySelectionIfInactive();
  }

  private scheduleHistorySelectionReleaseCheck(): void {
    const releaseIfInactive = (): void => this.releaseHistorySelectionIfInactive();
    if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
      window.setTimeout(releaseIfInactive, 0);
      return;
    }
    releaseIfInactive();
  }

  private releaseHistorySelectionIfInactive(): void {
    if (this.historySelectionPointerActive || historyListHasActiveSelection(this.elements.historyList)) {
      return;
    }
    this.flushDeferredHistoryRender();
    this.flushDeferredVirtualHistoryWindowRender();
  }

  private flushDeferredHistoryRenderIfInactive(): void {
    if (this.historySelectionPointerActive || historyListHasActiveSelection(this.elements.historyList)) {
      return;
    }
    this.flushDeferredHistoryRender();
    this.flushDeferredVirtualHistoryWindowRender();
  }

  private flushDeferredHistoryRender(): void {
    const deferred = this.deferredHistoryRender;
    if (!deferred) {
      return;
    }
    this.deferredHistoryRender = null;
    this.renderHistory(deferred.document, deferred.historyVisible, deferred.translationLanguage);
  }

  private flushDeferredVirtualHistoryWindowRender(): void {
    if (!this.deferredVirtualHistoryWindowRender || !this.virtualHistory) {
      this.deferredVirtualHistoryWindowRender = false;
      this.deferredVirtualHistoryAnchor = null;
      this.deferredVirtualHistoryAnchorLatest = false;
      return;
    }
    this.deferredVirtualHistoryWindowRender = false;
    const anchor = this.deferredVirtualHistoryAnchor;
    const anchorLatest = this.deferredVirtualHistoryAnchorLatest;
    this.deferredVirtualHistoryAnchor = null;
    this.deferredVirtualHistoryAnchorLatest = false;
    this.renderVirtualHistoryWindow({ anchor: anchorLatest ? null : anchor, anchorLatest });
  }

  private renderHistory(document: SubtitleDocument, historyVisible: boolean, translationLanguage: string): void {
    const historyVersion = document.stableRenderVersion;
    if (
      this.renderedHistoryVersion === historyVersion &&
      this.renderedHistoryTranslationEnabled === document.translationEnabled &&
      this.renderedHistoryTranslationLanguage === translationLanguage
    ) {
      return;
    }

    const lines = document.stableLines;
    if (this.historySelectionIsActive()) {
      this.deferredHistoryRender = { document, historyVisible, translationLanguage };
      return;
    }
    this.deferredHistoryRender = null;
    const useDeferredVirtualWindow = this.deferredVirtualHistoryWindowRender;
    const deferredAnchorLatest = useDeferredVirtualWindow && this.deferredVirtualHistoryAnchorLatest;
    const deferredAnchor = useDeferredVirtualWindow && !deferredAnchorLatest ? this.deferredVirtualHistoryAnchor : null;
    this.deferredVirtualHistoryWindowRender = false;
    this.deferredVirtualHistoryAnchor = null;
    this.deferredVirtualHistoryAnchorLatest = false;
    const shouldVirtualize = lines.length > HISTORY_VIRTUALIZATION_MIN_LINES;
    const hadRenderedHistory = this.renderedHistoryLines.length > 0;
    const translationChanged = this.renderedHistoryTranslationEnabled !== document.translationEnabled;
    const translationLanguageChanged = this.renderedHistoryTranslationLanguage !== translationLanguage;
    const wasVirtualized = this.virtualHistory !== null;
    const wasHistoryAtLatest =
      !hadRenderedHistory ||
      (wasVirtualized
        ? this.virtualHistoryPinnedToLatest
        : this.historyPinnedToLatest ||
          this.historyAutoScrollToLatestPending ||
          historyListScrolledToLatest(this.elements.historyList));
    const anchorLatest = shouldVirtualize && (!historyVisible || wasHistoryAtLatest || deferredAnchorLatest);
    const virtualAnchor =
      shouldVirtualize && !anchorLatest && historyVisible
        ? deferredAnchor || (wasVirtualized ? this.currentVirtualScrollAnchor() : this.currentFullHistoryScrollAnchor())
        : null;
    const fullAnchor =
      !shouldVirtualize && wasVirtualized && !wasHistoryAtLatest && historyVisible
        ? this.currentVirtualScrollAnchor()
        : null;

    if (shouldVirtualize) {
      this.elements.historyList.classList.add("is-virtualized");
      this.virtualHistory = {
        lines: lines.slice(),
        translationEnabled: document.translationEnabled,
        translationLanguage,
      };
      this.renderVirtualHistoryWindow({ anchor: virtualAnchor, anchorLatest });
    } else {
      this.virtualHistory = null;
      this.virtualHistoryPinnedToLatest = false;
      this.elements.historyList.classList.remove("is-virtualized");
      if (wasVirtualized) {
        this.elements.historyList.replaceChildren();
      }
      this.renderFullHistory(
        lines,
        document.translationEnabled,
        translationLanguage,
        translationChanged,
        translationLanguageChanged,
        wasVirtualized,
      );
      if (fullAnchor) {
        this.anchorFullHistoryScroll(fullAnchor, lines);
      }
    }
    this.renderedHistoryLines = lines.slice();
    this.renderedHistoryTranslationEnabled = document.translationEnabled;
    this.renderedHistoryTranslationLanguage = translationLanguage;
    this.renderedHistoryVersion = historyVersion;
    if (historyVisible && wasHistoryAtLatest) {
      this.scrollHistoryToLatest("smooth");
    }
  }

  private historySelectionIsActive(): boolean {
    return this.historySelectionPointerActive || historyListHasActiveSelection(this.elements.historyList);
  }

  private renderFullHistory(
    lines: readonly SubtitleLine[],
    translationEnabled: boolean,
    translationLanguage: string,
    translationChanged: boolean,
    translationLanguageChanged: boolean,
    forceUpdate = false,
  ): void {
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (!line) {
        continue;
      }
      const item = this.elements.historyList.children[index] as HTMLElement | undefined;
      const historyItem = item || createHistoryItem();
      if (!item) {
        this.elements.historyList.append(historyItem);
      }
      if (
        forceUpdate ||
        translationChanged ||
        translationLanguageChanged ||
        this.renderedHistoryLines[index] !== line
      ) {
        updateHistoryItem(
          historyItem,
          line,
          translationEnabled,
          this.renderedHistoryLines[index] || null,
          translationLanguage,
        );
      }
      historyItem.classList.toggle("is-latest", index === lines.length - 1);
    }
    trimHistoryItems(this.elements.historyList, lines.length);
  }

  private renderVirtualHistoryWindow(
    {
      anchor = null,
      anchorLatest = false,
    }: {
      anchor?: VirtualScrollAnchor | null;
      anchorLatest?: boolean;
    } = {},
    measurementPass = 0,
  ): void {
    const state = this.virtualHistory;
    if (!state) {
      return;
    }

    const offsets = historyLineOffsets(state.lines, this.historyItemHeightByKey, state.translationEnabled);
    const totalHeight = offsets.at(-1) || 0;
    const viewportHeight = historyViewportHeight(this.elements.historyList);
    const scrollTop = anchorLatest
      ? Math.max(0, totalHeight - viewportHeight)
      : anchor
        ? anchoredHistoryScrollTop(anchor, state.lines, offsets, totalHeight, viewportHeight)
        : Math.max(0, this.elements.historyList.scrollTop);
    if (anchorLatest || anchor) {
      this.elements.historyList.scrollTop = scrollTop;
    }
    const visibleStart = historyIndexAtOffset(offsets, scrollTop);
    const visibleEnd = historyIndexAtOffset(offsets, scrollTop + viewportHeight) + 1;
    const start = Math.max(0, visibleStart - HISTORY_VIRTUAL_OVERSCAN_ROWS);
    const end = Math.min(state.lines.length, visibleEnd + HISTORY_VIRTUAL_OVERSCAN_ROWS);

    const children: HTMLElement[] = [];
    const topHeight = offsets[start] || 0;
    if (topHeight > 0) {
      children.push(createHistorySpacer(topHeight));
    }
    const renderedItems: Array<{ index: number; item: HTMLElement }> = [];
    for (let index = start; index < end; index += 1) {
      const line = state.lines[index];
      if (!line) {
        continue;
      }
      const item = createHistoryItem();
      updateHistoryItem(
        item,
        line,
        state.translationEnabled,
        this.renderedHistoryLines[index] || null,
        state.translationLanguage,
      );
      item.classList.toggle("is-latest", index === state.lines.length - 1);
      children.push(item);
      renderedItems.push({ index, item });
    }
    const bottomHeight = Math.max(0, totalHeight - (offsets[end] || totalHeight));
    if (bottomHeight > 0) {
      children.push(createHistorySpacer(bottomHeight));
    }
    this.elements.historyList.replaceChildren(...children);
    const heightsChanged = this.measureVirtualHistoryItems(state.lines, renderedItems, state.translationEnabled);
    if (heightsChanged && (anchorLatest || anchor) && measurementPass < 1) {
      this.renderVirtualHistoryWindow({ anchor, anchorLatest }, measurementPass + 1);
      return;
    }
    this.virtualHistoryPinnedToLatest = this.isVirtualHistoryScrolledToLatest();
  }

  private measureVirtualHistoryItems(
    lines: readonly SubtitleLine[],
    renderedItems: Array<{ index: number; item: HTMLElement }>,
    translationEnabled: boolean,
  ): boolean {
    let changed = false;
    for (const { index, item } of renderedItems) {
      const line = lines[index];
      if (!line) {
        continue;
      }
      const height = measuredHistoryItemHeight(item);
      if (height > 0) {
        const key = historyLineKey(line, index, translationEnabled);
        changed ||= this.historyItemHeightByKey.get(key) !== height;
        this.historyItemHeightByKey.set(key, height);
      }
    }
    return changed;
  }

  private currentVirtualScrollAnchor(): VirtualScrollAnchor | null {
    const state = this.virtualHistory;
    if (!state) {
      return null;
    }
    const offsets = historyLineOffsets(state.lines, this.historyItemHeightByKey, state.translationEnabled);
    const scrollTop = Math.max(0, this.elements.historyList.scrollTop);
    const index = historyIndexAtOffset(offsets, scrollTop);
    const line = state.lines[index];
    if (!line) {
      return null;
    }
    return {
      identity: historyLineIdentity(line, index),
      fallbackIndex: index,
      offsetPx: Math.max(0, scrollTop - (offsets[index] || 0)),
    };
  }

  private currentFullHistoryScrollAnchor(): VirtualScrollAnchor | null {
    if (this.renderedHistoryLines.length <= 0) {
      return null;
    }
    const offsets = renderedHistoryLineOffsets(this.renderedHistoryLines.length, this.elements.historyList);
    const scrollTop = Math.max(0, this.elements.historyList.scrollTop);
    const index = historyIndexAtOffset(offsets, scrollTop);
    const line = this.renderedHistoryLines[index];
    if (!line) {
      return null;
    }
    return {
      identity: historyLineIdentity(line, index),
      fallbackIndex: index,
      offsetPx: Math.max(0, scrollTop - (offsets[index] || 0)),
    };
  }

  private isVirtualHistoryScrolledToLatest(): boolean {
    const state = this.virtualHistory;
    if (!state) {
      return false;
    }
    const offsets = historyLineOffsets(state.lines, this.historyItemHeightByKey, state.translationEnabled);
    const totalHeight = offsets.at(-1) || 0;
    const maxScrollTop = Math.max(0, totalHeight - historyViewportHeight(this.elements.historyList));
    return Math.max(0, this.elements.historyList.scrollTop) >= maxScrollTop - 2;
  }

  private handleHistoryPointerDown(event: Event): void {
    const targetItem = historyItemFromEventTarget(event.target);
    if (targetItem) {
      this.historySelectionPointerActive = true;
    }
  }

  private anchorFullHistoryScroll(anchor: VirtualScrollAnchor, lines: readonly SubtitleLine[]): void {
    const offsets = renderedHistoryLineOffsets(lines.length, this.elements.historyList);
    const totalHeight = offsets.at(-1) || 0;
    const viewportHeight = historyViewportHeight(this.elements.historyList);
    this.elements.historyList.scrollTop = anchoredHistoryScrollTop(anchor, lines, offsets, totalHeight, viewportHeight);
  }

  private handleHistoryLayoutChange(): void {
    if (!this.virtualHistory) {
      return;
    }
    if (this.historySelectionIsActive()) {
      const anchorLatest = this.virtualHistoryPinnedToLatest;
      this.deferredVirtualHistoryAnchor = anchorLatest ? null : this.currentVirtualScrollAnchor();
      this.historyItemHeightByKey.clear();
      this.deferredVirtualHistoryAnchorLatest = anchorLatest;
      this.deferredVirtualHistoryWindowRender = true;
      return;
    }
    const anchorLatest = this.virtualHistoryPinnedToLatest;
    const anchor = anchorLatest ? null : this.currentVirtualScrollAnchor();
    this.historyItemHeightByKey.clear();
    this.renderVirtualHistoryWindow({ anchor, anchorLatest });
  }

  private handleHistoryScroll(): void {
    if (this.virtualHistory) {
      if (this.historySelectionIsActive()) {
        this.deferredVirtualHistoryAnchor = null;
        this.deferredVirtualHistoryAnchorLatest = false;
        this.deferredVirtualHistoryWindowRender = true;
        this.virtualHistoryPinnedToLatest = false;
        return;
      }
      this.renderVirtualHistoryWindow();
      if (!this.virtualHistoryPinnedToLatest) {
        this.historyAutoScrollToLatestPending = false;
        this.historyPinnedToLatest = false;
      }
      return;
    }
    if (this.historyAutoScrollToLatestPending) {
      if (historyListScrolledToLatest(this.elements.historyList)) {
        this.historyAutoScrollToLatestPending = false;
        this.historyPinnedToLatest = true;
      }
      return;
    }
    this.historyPinnedToLatest = historyListScrolledToLatest(this.elements.historyList);
  }

  private handleHistoryUserScrollIntent(): void {
    this.historyAutoScrollToLatestPending = false;
    this.historyPinnedToLatest = historyListScrolledToLatest(this.elements.historyList);
  }
}

function visibleTranslation(line: SubtitleLine): string | null {
  return line.translation || line.translationMessage;
}

function renderCaptionLine(
  line: SubtitleLine | null,
  sourceElement: HTMLElement,
  translationElement: HTMLElement,
  translationLanguage: string,
): void {
  applyLineLanguage(sourceElement, line?.language);
  applyLineLanguage(translationElement, translationLanguage);
  setCaptionText(sourceElement, line?.text || "");
  setCaptionText(translationElement, line ? visibleTranslation(line) || "" : "");
}

function createHistoryItem(): HTMLElement {
  const item = document.createElement("article");
  item.className = "history-item";

  const time = document.createElement("div");
  time.className = "history-time";

  const source = createHistoryText("history-source");
  const translation = createHistoryText("history-translation");

  item.append(time, source, translation);
  return item;
}

function createHistorySpacer(height: number): HTMLElement {
  const spacer = document.createElement("div");
  spacer.className = "history-spacer";
  spacer.setAttribute("aria-hidden", "true");
  spacer.style.setProperty("height", `${Math.max(0, Math.round(height))}px`);
  return spacer;
}

function createHistoryText(className: string): HTMLElement {
  const element = document.createElement("div");
  element.className = className;
  element.setAttribute("dir", "auto");
  return element;
}

function updateHistoryItem(
  item: HTMLElement,
  line: SubtitleLine,
  translationEnabled: boolean,
  previousLine: SubtitleLine | null,
  translationLanguage: string,
): void {
  const [time, source, translation] = Array.from(item.children) as HTMLElement[];
  if (previousLine && !isSameHistoryLine(previousLine, line)) {
    setTextIfChanged(source, "");
    setTextIfChanged(translation, "");
  }
  if (source) {
    applyLineLanguage(source, line.language);
  }
  if (translation) {
    applyLineLanguage(translation, translationLanguage);
  }
  setTextIfChanged(time, formatRange(line.startMs, line.endMs, line.timingStatus));
  setTextIfChanged(source, line.text);
  setTextIfChanged(translation, translationEnabled ? visibleTranslation(line) || "" : "");
}

function setTextIfChanged(element: HTMLElement | undefined, value: string): boolean {
  if (element && element.textContent !== value) {
    element.textContent = value;
    return true;
  }
  return false;
}

function trimHistoryItems(historyList: HTMLElement, lineCount: number): void {
  if (historyList.children.length <= lineCount) {
    return;
  }
  const children = Array.from(historyList.children) as HTMLElement[];
  historyList.replaceChildren(...children.slice(0, lineCount));
}

function historyLineOffsets(
  lines: readonly SubtitleLine[],
  heightByKey: ReadonlyMap<string, number>,
  translationEnabled: boolean,
): number[] {
  const offsets = [0];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const measuredHeight = line ? heightByKey.get(historyLineKey(line, index, translationEnabled)) : null;
    const rowHeight = measuredHeight && measuredHeight > 0 ? measuredHeight : HISTORY_ROW_ESTIMATE_PX;
    const gap = index === lines.length - 1 ? 0 : HISTORY_ROW_GAP_PX;
    offsets.push((offsets[index] ?? 0) + rowHeight + gap);
  }
  return offsets;
}

function renderedHistoryLineOffsets(lineCount: number, historyList: HTMLElement): number[] {
  const offsets = [0];
  for (let index = 0; index < lineCount; index += 1) {
    const item = historyList.children[index] as HTMLElement | undefined;
    const rowHeight = item ? measuredHistoryItemHeight(item) : HISTORY_ROW_ESTIMATE_PX;
    const gap = index === lineCount - 1 ? 0 : HISTORY_ROW_GAP_PX;
    offsets.push((offsets[index] ?? 0) + rowHeight + gap);
  }
  return offsets;
}

function historyIndexAtOffset(offsets: readonly number[], offset: number): number {
  let low = 0;
  let high = Math.max(0, offsets.length - 2);
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if ((offsets[mid + 1] || 0) <= offset) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function historyViewportHeight(historyList: HTMLElement): number {
  return historyList.clientHeight > 0 ? historyList.clientHeight : HISTORY_DEFAULT_VIEWPORT_PX;
}

function historyListScrolledToLatest(historyList: HTMLElement): boolean {
  const maxScrollTop = Math.max(0, historyList.scrollHeight - Math.max(0, historyList.clientHeight));
  return Math.max(0, historyList.scrollTop) >= maxScrollTop - 2;
}

function historyLineKey(line: SubtitleLine, fallbackIndex: number, translationEnabled: boolean): string {
  const visibleParts = [
    line.text,
    line.startMs ?? "",
    line.endMs ?? "",
    line.timingStatus ?? "",
    translationEnabled ? visibleTranslation(line) || "" : "",
  ].join("\u001f");
  return `${historyLineIdentity(line, fallbackIndex)}:${visibleParts}`;
}

function historyLineIdentity(line: SubtitleLine, fallbackIndex: number): string {
  if (line.id) {
    return `id:${line.id}`;
  }
  if (isInteger(line.index)) {
    return `index:${line.index}`;
  }
  return `offset:${fallbackIndex}`;
}

function anchoredHistoryScrollTop(
  anchor: VirtualScrollAnchor,
  lines: readonly SubtitleLine[],
  offsets: readonly number[],
  totalHeight: number,
  viewportHeight: number,
): number {
  const index = historyAnchorIndex(anchor, lines);
  const maxScrollTop = Math.max(0, totalHeight - viewportHeight);
  return Math.min(maxScrollTop, Math.max(0, (offsets[index] || 0) + anchor.offsetPx));
}

function historyAnchorIndex(anchor: VirtualScrollAnchor, lines: readonly SubtitleLine[]): number {
  const index = lines.findIndex((line, lineIndex) => historyLineIdentity(line, lineIndex) === anchor.identity);
  if (index >= 0) {
    return index;
  }
  return Math.min(Math.max(0, anchor.fallbackIndex), Math.max(0, lines.length - 1));
}

function measuredHistoryItemHeight(item: HTMLElement): number {
  const rectHeight = typeof item.getBoundingClientRect === "function" ? item.getBoundingClientRect().height : 0;
  const height = rectHeight || item.offsetHeight || 0;
  return height > 0 ? height : HISTORY_ROW_ESTIMATE_PX;
}

function latestHistoryItem(historyList: HTMLElement): HTMLElement | null {
  const children = Array.from(historyList.children) as HTMLElement[];
  for (let index = children.length - 1; index >= 0; index -= 1) {
    const child = children[index];
    if (child?.className.split(/\s+/).includes("history-item")) {
      return child;
    }
  }
  return null;
}

function historyItemFromEventTarget(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof HTMLElement)) {
    return null;
  }
  let element: HTMLElement | null = target;
  while (element) {
    if (isHistoryItem(element)) {
      return element;
    }
    element = element.parentElement;
  }
  return null;
}

function isHistoryItem(element: HTMLElement): boolean {
  return element.className.split(/\s+/).includes("history-item");
}

function historyListHasActiveSelection(historyList: HTMLElement): boolean {
  if (typeof document === "undefined" || typeof document.getSelection !== "function") {
    return false;
  }
  const selection = document.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount <= 0 || selection.toString() === "") {
    return false;
  }
  if (nodeIsWithinElement(selection.anchorNode, historyList) || nodeIsWithinElement(selection.focusNode, historyList)) {
    return true;
  }
  for (let index = 0; index < selection.rangeCount; index += 1) {
    const range = selection.getRangeAt(index);
    if (typeof range.intersectsNode === "function" && range.intersectsNode(historyList)) {
      return true;
    }
    if (nodeIsWithinElement(range.commonAncestorContainer, historyList)) {
      return true;
    }
  }
  return false;
}

function nodeIsWithinElement(node: Node | null, element: HTMLElement): boolean {
  let current: Node | null = node;
  while (current) {
    if (current === element) {
      return true;
    }
    current = current.parentNode ?? ((current as unknown as { parentElement?: Node | null }).parentElement || null);
  }
  return false;
}

function setCaptionText(element: HTMLElement, value: string): void {
  if (setTextIfChanged(element, value)) {
    anchorCaptionTail(element);
  }
}

function observeCurrentCaptionLayout(...elements: HTMLElement[]): void {
  if (typeof ResizeObserver === "undefined") {
    return;
  }
  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      if (entry.target instanceof HTMLElement) {
        anchorCaptionTail(entry.target);
      }
    }
  });
  for (const element of elements) {
    observer.observe(element);
  }
}

function observeHistoryListLayout(element: HTMLElement, onResize: () => void): void {
  if (typeof ResizeObserver === "undefined") {
    return;
  }
  const observer = new ResizeObserver(onResize);
  observer.observe(element);
}

function anchorCaptionTail(element: HTMLElement): void {
  element.scrollTop = element.scrollHeight;
}

// Tag the element with the line's language (as a BCP-47 tag) so screen readers
// pick the right voice. Display names are mapped to tags; unknown values become
// "" rather than misleading assistive tech.
function applyLineLanguage(element: HTMLElement, language: string | undefined): void {
  element.setAttribute("lang", languageTag(language));
}

function isSameHistoryLine(left: SubtitleLine, right: SubtitleLine): boolean {
  if (isInteger(left.index) && isInteger(right.index)) {
    return left.index === right.index;
  }
  if (left.id || right.id) {
    return left.id === right.id;
  }
  return left === right;
}

function formatRange(startMs: number | null, endMs: number | null, status: string | null): string {
  const prefix = isInteger(startMs) && isInteger(endMs) ? `${formatClock(startMs)} - ${formatClock(endMs)}` : "pending";
  return status ? `${prefix} ${status}` : prefix;
}
