export type SubtitleEvent = Record<string, unknown> & {
  type?: unknown;
};

interface SubtitleLineInit {
  id?: string | null;
  index?: number | null;
  startMs?: number | null;
  endMs?: number | null;
  text?: string;
  language?: string;
  sourceRevision?: number | null;
  timingStatus?: string | null;
  translation?: string | null;
  translationStatus?: string | null;
  translationMessage?: string | null;
}

export interface SubtitleWindow {
  previous: SubtitleLine | null;
  current: SubtitleLine | null;
}

export class SubtitleLine {
  readonly id: string | null;
  readonly index: number | null;
  readonly startMs: number | null;
  readonly endMs: number | null;
  readonly text: string;
  readonly language: string;
  readonly sourceRevision: number | null;
  readonly timingStatus: string | null;
  readonly translation: string | null;
  readonly translationStatus: string | null;
  readonly translationMessage: string | null;

  constructor({
    id = null,
    index = null,
    startMs = null,
    endMs = null,
    text = "",
    language = "",
    sourceRevision = null,
    timingStatus = null,
    translation = null,
    translationStatus = null,
    translationMessage = null,
  }: SubtitleLineInit = {}) {
    this.id = id;
    this.index = index;
    this.startMs = startMs;
    this.endMs = endMs;
    this.text = text;
    this.language = language;
    this.sourceRevision = sourceRevision;
    this.timingStatus = timingStatus;
    this.translation = translation;
    this.translationStatus = translationStatus;
    this.translationMessage = translationMessage;
  }

  withPatch(patch: SubtitleLineInit): SubtitleLine {
    return new SubtitleLine({ ...this, ...patch });
  }
}

export class SubtitleDocument {
  translationEnabled: boolean;
  revision: number;
  stableLines: SubtitleLine[];
  current: SubtitleLine | null;

  constructor({ translationEnabled = true }: { translationEnabled?: boolean } = {}) {
    this.translationEnabled = Boolean(translationEnabled);
    this.revision = 0;
    this.stableLines = [];
    this.current = null;
  }

  setTranslationEnabled(enabled: boolean): void {
    this.translationEnabled = Boolean(enabled);
  }

  applyEvent(event: SubtitleEvent): void {
    const eventType = String(event?.type || "");
    if (eventType === "transcript_update") {
      this.applyTranscriptUpdate(event);
    } else if (eventType === "transcript_timing_update") {
      this.applyTranscriptTimingUpdate(event);
    } else if (eventType === "transcript_final") {
      this.applyTranscriptFinal(event);
    } else if (eventType === "translation_stable") {
      this.applyStableTranslation(event);
    } else if (eventType === "translation_preview") {
      this.applyPreviewTranslation(event);
    } else if (eventType === "translation_status") {
      this.applyTranslationStatus(event);
    }
  }

  window({ includeTranslation = this.translationEnabled }: { includeTranslation?: boolean } = {}): SubtitleWindow {
    return {
      previous: renderLine(this.stableLines.at(-1) || null, includeTranslation),
      current: renderLine(this.current, includeTranslation),
    };
  }

  toSrt({ includeTranslation = this.translationEnabled }: { includeTranslation?: boolean } = {}): string {
    const blocks = [];
    let number = 1;
    for (const line of this.stableLines) {
      if (!isInteger(line.startMs) || !isInteger(line.endMs)) {
        continue;
      }
      const textLines = [line.text];
      if (includeTranslation && line.translation) {
        textLines.push(line.translation);
      }
      blocks.push(
        [
          String(number),
          `${formatSrtTime(line.startMs)} --> ${formatSrtTime(line.endMs)}`,
          ...textLines,
        ].join("\n"),
      );
      number += 1;
    }
    return blocks.length ? `${blocks.join("\n\n")}\n` : "";
  }

  private applyTranscriptUpdate(event: SubtitleEvent): void {
    const stableBase = toInt(event.stable_base, 0);
    const stableAppends = Array.isArray(event.stable_appends) ? event.stable_appends : [];
    const stableCount = toInt(event.stable_count, 0);
    if (stableBase !== this.stableLines.length) {
      throw new Error(`stable cursor mismatch: stable_base=${stableBase}, local_count=${this.stableLines.length}`);
    }
    if (stableBase + stableAppends.length !== stableCount) {
      throw new Error(
        `stable count mismatch: stable_base=${stableBase}, appends=${stableAppends.length}, stable_count=${stableCount}`,
      );
    }

    const revision = toInt(event.revision, this.revision);
    const previousCurrent = this.current;
    const resetCurrentPreview = stableAppends.length > 0;
    for (const segment of stableAppends) {
      if (isRecord(segment)) {
        this.stableLines.push(preserveStableTranslation(lineFromSegment(segment, revision), previousCurrent));
      }
    }

    const nextCurrent = isRecord(event.partial)
      ? lineFromSegment(event.partial, revision)
      : null;
    this.current = resetCurrentPreview
      ? nextCurrent
      : preserveCurrentTranslation(nextCurrent, previousCurrent);
    this.revision = revision;
  }

  private applyTranscriptTimingUpdate(event: SubtitleEvent): void {
    const index = this.stableIndex(event);
    if (index === null) {
      return;
    }
    this.patchStableLine(index, {
      startMs: optionalInt(event.start_ms),
      endMs: optionalInt(event.end_ms),
      timingStatus: stringOrNull(event.timing_status),
    });
  }

  private applyTranscriptFinal(event: SubtitleEvent): void {
    const existing = new Map(this.stableLines.filter((line) => line.id).map((line) => [line.id as string, line]));
    const revision = toInt(event.revision, this.revision);
    const lines = [];
    for (const segment of Array.isArray(event.segments) ? event.segments : []) {
      if (!isRecord(segment)) {
        continue;
      }
      let line = lineFromSegment(segment, revision);
      const previous = line.id ? existing.get(line.id) : undefined;
      if (previous) {
        line = line.withPatch({
          translation: previous.translation,
          translationStatus: previous.translationStatus,
          translationMessage: previous.translationMessage,
        });
      }
      lines.push(line);
    }
    this.stableLines = lines;
    this.current = null;
    this.revision = revision;
  }

  private applyStableTranslation(event: SubtitleEvent): void {
    const index = this.stableIndex(event);
    const text = String(event.text || "").trim();
    if (index === null || !text) {
      return;
    }
    this.patchStableLine(index, {
      translation: text,
      translationStatus: null,
      translationMessage: null,
    });
  }

  private applyPreviewTranslation(event: SubtitleEvent): void {
    if (!this.current) {
      return;
    }
    const sourceRevision = toInt(event.source_revision, 0);
    const text = String(event.text || "").trim();
    if (text && this.current.sourceRevision === sourceRevision) {
      this.current = this.current.withPatch({ translation: text });
    }
  }

  private applyTranslationStatus(event: SubtitleEvent): void {
    const index = this.stableIndex(event);
    if (index === null) {
      return;
    }
    this.patchStableLine(index, {
      translationStatus: String(event.code || ""),
      translationMessage: String(event.message || ""),
    });
  }

  private stableIndex(event: SubtitleEvent): number | null {
    const segmentId = String(event.source_segment_id || "");
    if (segmentId) {
      const index = this.stableLines.findIndex((line) => line.id === segmentId);
      return index >= 0 ? index : null;
    }
    const segmentIndex = optionalInt(event.source_segment_index);
    if (!segmentIndex || segmentIndex <= 0) {
      return null;
    }
    const index = this.stableLines.findIndex((line) => line.index === segmentIndex);
    return index >= 0 ? index : null;
  }

  private patchStableLine(index: number, patch: SubtitleLineInit): void {
    const line = this.stableLines[index];
    if (line) {
      this.stableLines[index] = line.withPatch(patch);
    }
  }
}

function lineFromSegment(segment: Record<string, unknown>, revision: number): SubtitleLine {
  return new SubtitleLine({
    id: stringOrNull(segment.id),
    index: optionalInt(segment.index),
    startMs: optionalInt(segment.start_ms),
    endMs: optionalInt(segment.end_ms),
    text: String(segment.text || "").trim(),
    language: String(segment.language || ""),
    sourceRevision: revision,
    timingStatus: stringOrNull(segment.timing_status),
  });
}

function renderLine(line: SubtitleLine | null, includeTranslation: boolean): SubtitleLine | null {
  if (!line || includeTranslation) {
    return line;
  }
  return line.withPatch({
    translation: null,
    translationStatus: null,
    translationMessage: null,
  });
}

function preserveCurrentTranslation(
  next: SubtitleLine | null,
  previous: SubtitleLine | null,
): SubtitleLine | null {
  if (!next || !previous?.translation || !isSamePartialLine(next, previous)) {
    return next;
  }
  return next.withPatch({
    translation: previous.translation,
    translationStatus: previous.translationStatus,
    translationMessage: previous.translationMessage,
  });
}

function preserveStableTranslation(
  next: SubtitleLine,
  previous: SubtitleLine | null,
): SubtitleLine {
  if (!previous?.translation || !isSamePartialLine(next, previous)) {
    return next;
  }
  return next.withPatch({
    translation: previous.translation,
    translationStatus: previous.translationStatus,
    translationMessage: previous.translationMessage,
  });
}

function isSamePartialLine(left: SubtitleLine, right: SubtitleLine): boolean {
  if (left.id && right.id) {
    return left.id === right.id;
  }
  if (isInteger(left.index) && isInteger(right.index)) {
    return left.index === right.index;
  }

  const leftText = left.text.trim();
  const rightText = right.text.trim();
  return Boolean(leftText && rightText)
    && (leftText === rightText || leftText.startsWith(rightText) || rightText.startsWith(leftText));
}

function formatSrtTime(ms: number): string {
  let totalMs = Math.max(0, Math.trunc(ms));
  const hours = Math.trunc(totalMs / 3600000);
  totalMs %= 3600000;
  const minutes = Math.trunc(totalMs / 60000);
  totalMs %= 60000;
  const seconds = Math.trunc(totalMs / 1000);
  const millis = totalMs % 1000;
  return `${pad2(hours)}:${pad2(minutes)}:${pad2(seconds)},${pad3(millis)}`;
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function pad3(value: number): string {
  return String(value).padStart(3, "0");
}

function optionalInt(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function toInt(value: unknown, fallback: number): number {
  return optionalInt(value) ?? fallback;
}

function stringOrNull(value: unknown): string | null {
  const text = String(value || "");
  return text ? text : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function isInteger(value: unknown): value is number {
  return Number.isInteger(value);
}
