import { SubtitleDocument, type SubtitleLine } from "./subtitle-document.js";

interface CaptionViewElements {
  previousSource: HTMLElement;
  previousTranslation: HTMLElement;
  currentSource: HTMLElement;
  currentTranslation: HTMLElement;
  historyList: HTMLElement;
}

export class CaptionView {
  private renderedHistoryLines: readonly SubtitleLine[] = [];
  private renderedHistoryTranslationEnabled: boolean | null = null;

  constructor(private readonly elements: CaptionViewElements) {}

  render(document: SubtitleDocument, { historyVisible }: { historyVisible: boolean }): void {
    const windowState = document.window();
    renderCaptionLine(windowState.previous, this.elements.previousSource, this.elements.previousTranslation);
    renderCaptionLine(windowState.current, this.elements.currentSource, this.elements.currentTranslation);
    this.renderHistory(document, historyVisible);
  }

  reset(): void {
    this.elements.historyList.replaceChildren();
    this.renderedHistoryLines = [];
    this.renderedHistoryTranslationEnabled = null;
  }

  scrollHistoryToLatest(behavior: ScrollBehavior): void {
    const target = this.elements.historyList.lastElementChild;
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

  private renderHistory(document: SubtitleDocument, historyVisible: boolean): void {
    const lines = document.stableLines;
    const hadNewLine = lines.length > this.renderedHistoryLines.length;
    const translationChanged = this.renderedHistoryTranslationEnabled !== document.translationEnabled;

    if (lines.length < this.renderedHistoryLines.length) {
      this.elements.historyList.replaceChildren();
      this.renderedHistoryLines = [];
    }

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
      if (translationChanged || this.renderedHistoryLines[index] !== line) {
        updateHistoryItem(historyItem, line, document.translationEnabled);
      }
      historyItem.classList.toggle("is-latest", index === lines.length - 1);
    }
    this.renderedHistoryLines = lines.slice();
    this.renderedHistoryTranslationEnabled = document.translationEnabled;
    if (historyVisible && hadNewLine) {
      this.scrollHistoryToLatest("smooth");
    }
  }
}

function renderCaptionLine(
  line: SubtitleLine | null,
  sourceElement: HTMLElement,
  translationElement: HTMLElement,
): void {
  setTextIfChanged(sourceElement, line?.text || "");
  setTextIfChanged(translationElement, line?.translation || "");
}

function createHistoryItem(): HTMLElement {
  const item = document.createElement("article");
  item.className = "history-item";

  const time = document.createElement("div");
  time.className = "history-time";

  const source = document.createElement("div");
  source.className = "history-source";

  const translation = document.createElement("div");
  translation.className = "history-translation";

  item.append(time, source, translation);
  return item;
}

function updateHistoryItem(item: HTMLElement, line: SubtitleLine, translationEnabled: boolean): void {
  const [time, source, translation] = Array.from(item.children) as HTMLElement[];
  if (time) {
    setTextIfChanged(time, formatRange(line.startMs, line.endMs, line.timingStatus));
  }
  if (source) {
    setTextIfChanged(source, line.text);
  }
  if (translation) {
    setTextIfChanged(
      translation,
      translationEnabled ? line.translation || line.translationMessage || "" : "",
    );
  }
}

function setTextIfChanged(element: HTMLElement, value: string): void {
  if (element.textContent !== value) {
    element.textContent = value;
  }
}

function formatRange(startMs: number | null, endMs: number | null, status: string | null): string {
  const prefix = isInteger(startMs) && isInteger(endMs)
    ? `${formatClock(startMs)} - ${formatClock(endMs)}`
    : "pending";
  return status ? `${prefix} ${status}` : prefix;
}

function formatClock(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const millis = Math.floor(ms % 1000);
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function isInteger(value: unknown): value is number {
  return Number.isInteger(value);
}
