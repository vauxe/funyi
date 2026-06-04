import { languageTag } from "./languages.js";
import { isInteger } from "./runtime-guards.js";
import type { SubtitleDocument, SubtitleLine } from "./subtitle-document.js";
import { formatClock } from "./time-format.js";
import type { TranscriptLine } from "./transcript-export.js";

interface CaptionViewElements {
  currentSource: HTMLElement;
  currentTranslation: HTMLElement;
  historyList: HTMLElement;
  announcer: HTMLElement;
}

// Cap the live-region log so a long session does not grow it without bound.
const MAX_ANNOUNCED_LINES = 40;

export class CaptionView {
  private readonly announcedKeys = new Set<string>();
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
    this.elements.announcer.replaceChildren();
    this.announcedKeys.clear();
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

  // Build the export transcript from the rendered history so a user's inline edits
  // (contenteditable rows flagged data-user-edited) are what gets copied, not the
  // raw model text.
  collectTranscriptLines(): TranscriptLine[] {
    return this.renderedHistoryLines.map((line, index) => {
      const cells = Array.from(this.elements.historyList.children[index]?.children ?? []) as HTMLElement[];
      return {
        startMs: line.startMs,
        text: editedValue(cells[1]) ?? line.text,
        translation: editedValue(cells[2]) ?? line.translation,
      };
    });
  }

  private renderHistory(document: SubtitleDocument, historyVisible: boolean, translationLanguage: string): void {
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
    this.announceStableLines(lines, document.translationEnabled, translationLanguage);
    this.renderedHistoryLines = lines.slice();
    this.renderedHistoryTranslationEnabled = document.translationEnabled;
    if (historyVisible && hadNewLine) {
      this.scrollHistoryToLatest("smooth");
    }
  }

  // Announce only stabilized lines through the polite log region. The visible
  // current line is rewritten on every partial, so announcing it would flood the
  // screen reader; committed segments give one announcement each. Lines are keyed
  // so a transcript_final rebuild (which replaces the list wholesale) never
  // re-announces or skips an already-spoken line.
  private announceStableLines(
    lines: readonly SubtitleLine[],
    translationEnabled: boolean,
    translationLanguage: string,
  ): void {
    let appended = false;
    for (const line of lines) {
      if (!line) {
        continue;
      }
      const source = line.text.trim();
      const translation = translationEnabled ? (line.translation || line.translationMessage || "").trim() : "";
      if (!source && !translation) {
        continue;
      }
      const key = line.id || `${line.index ?? ""}:${source}`;
      const sourceKey = `${key}:source:${source}`;
      const translationKey = `${key}:translation:${translation}`;
      const announceSource = Boolean(source) && !this.announcedKeys.has(sourceKey);
      const announceTranslation = Boolean(translation) && !this.announcedKeys.has(translationKey);
      if (!announceSource && !announceTranslation) {
        continue;
      }
      if (announceSource) {
        this.announcedKeys.add(sourceKey);
      }
      if (announceTranslation) {
        this.announcedKeys.add(translationKey);
      }
      // Source and translation are different languages, so each goes in its own
      // span with its own `lang` for correct screen-reader pronunciation.
      const entry = document.createElement("div");
      entry.setAttribute("dir", "auto");
      if (announceSource) {
        entry.append(announceSpan(source, line.language));
      }
      if (announceTranslation) {
        entry.append(announceSpan(`${announceSource ? " — " : ""}${translation}`, translationLanguage));
      }
      this.elements.announcer.append(entry);
      appended = true;
    }
    if (appended) {
      this.trimAnnouncer();
    }
  }

  private trimAnnouncer(): void {
    const children = Array.from(this.elements.announcer.children);
    if (children.length > MAX_ANNOUNCED_LINES) {
      this.elements.announcer.replaceChildren(...children.slice(children.length - MAX_ANNOUNCED_LINES));
    }
  }
}

// Returns the inline-edited text of a history cell, or null when the user has not
// touched it (so the caller falls back to the model value).
function editedValue(element: HTMLElement | undefined): string | null {
  return element?.dataset.userEdited === "true" ? (element.textContent ?? "") : null;
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
  setCaptionText(translationElement, line?.translation || line?.translationMessage || "");
}

function announceSpan(text: string, language: string | undefined): HTMLElement {
  const span = document.createElement("span");
  applyLineLanguage(span, language);
  span.textContent = text;
  return span;
}

function createHistoryItem(): HTMLElement {
  const item = document.createElement("article");
  item.className = "history-item";

  const time = document.createElement("div");
  time.className = "history-time";

  const source = createEditableHistoryText("history-source", "Source transcript");
  const translation = createEditableHistoryText("history-translation", "Translation");

  item.append(time, source, translation);
  return item;
}

function createEditableHistoryText(className: string, label: string): HTMLElement {
  const element = document.createElement("div");
  element.className = className;
  element.setAttribute("contenteditable", "plaintext-only");
  element.setAttribute("role", "textbox");
  element.setAttribute("aria-label", label);
  element.setAttribute("aria-multiline", "true");
  element.setAttribute("spellcheck", "false");
  element.setAttribute("dir", "auto");
  element.setAttribute("tabindex", "0");
  element.addEventListener("input", () => {
    element.dataset.userEdited = "true";
  });
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
    delete source?.dataset.userEdited;
    delete translation?.dataset.userEdited;
  }
  if (source) {
    applyLineLanguage(source, line.language);
  }
  if (translation) {
    applyLineLanguage(translation, translationLanguage);
  }
  setTextIfChanged(time, formatRange(line.startMs, line.endMs, line.timingStatus));
  setEditableTextIfChanged(source, line.text);
  setEditableTextIfChanged(translation, translationEnabled ? line.translation || line.translationMessage || "" : "");
}

function setTextIfChanged(element: HTMLElement | undefined, value: string): void {
  if (element && element.textContent !== value) {
    element.textContent = value;
  }
}

function setEditableTextIfChanged(element: HTMLElement | undefined, value: string): void {
  if (!element || element.dataset.userEdited === "true") {
    return;
  }
  setTextIfChanged(element, value);
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
  if (left.id || right.id) {
    return left.id === right.id;
  }
  if (isInteger(left.index) || isInteger(right.index)) {
    return left.index === right.index;
  }
  return left === right;
}

function formatRange(startMs: number | null, endMs: number | null, status: string | null): string {
  const prefix = isInteger(startMs) && isInteger(endMs) ? `${formatClock(startMs)} - ${formatClock(endMs)}` : "pending";
  return status ? `${prefix} ${status}` : prefix;
}
