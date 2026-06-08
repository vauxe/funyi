import { languageTag } from "./languages.js";
import { isInteger } from "./runtime-guards.js";
import type { SubtitleDocument, SubtitleLine } from "./subtitle-document.js";
import { formatClock } from "./time-format.js";

interface CaptionViewElements {
  currentSource: HTMLElement;
  currentTranslation: HTMLElement;
  historyList: HTMLElement;
}

export class CaptionView {
  private renderedHistoryLines: readonly SubtitleLine[] = [];
  private renderedHistoryTranslationEnabled: boolean | null = null;

  constructor(private readonly elements: CaptionViewElements) {}

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

  private renderHistory(document: SubtitleDocument, historyVisible: boolean, translationLanguage: string): void {
    const lines = document.stableLines;
    const hadNewLine = lines.length > this.renderedHistoryLines.length;
    const translationChanged = this.renderedHistoryTranslationEnabled !== document.translationEnabled;

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
        updateHistoryItem(
          historyItem,
          line,
          document.translationEnabled,
          this.renderedHistoryLines[index] || null,
          translationLanguage,
        );
      }
      historyItem.classList.toggle("is-latest", index === lines.length - 1);
    }
    trimHistoryItems(this.elements.historyList, lines.length);
    this.renderedHistoryLines = lines.slice();
    this.renderedHistoryTranslationEnabled = document.translationEnabled;
    if (historyVisible && hadNewLine) {
      this.scrollHistoryToLatest("smooth");
    }
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

function setTextIfChanged(element: HTMLElement | undefined, value: string): void {
  if (element && element.textContent !== value) {
    element.textContent = value;
  }
}

function trimHistoryItems(historyList: HTMLElement, lineCount: number): void {
  const children = Array.from(historyList.children) as HTMLElement[];
  if (children.length > lineCount) {
    historyList.replaceChildren(...children.slice(0, lineCount));
  }
}

function setCaptionText(element: HTMLElement, value: string): void {
  setTextIfChanged(element, value);
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
