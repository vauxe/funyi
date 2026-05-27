import type { RealtimeEvent } from "./realtime-events.js";
import { optionalRecord, recordArray } from "./runtime-guards.js";

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

export interface SubtitleLine {
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
}

class SubtitleLineRecord implements SubtitleLine {
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

  withPatch(patch: SubtitleLineInit): SubtitleLineRecord {
    return new SubtitleLineRecord({ ...this, ...patch });
  }
}

export class SubtitleDocument {
  private currentLine: SubtitleLineRecord | null;
  private revision: number;
  private stableLineList: SubtitleLineRecord[];
  private translationEnabledValue: boolean;

  constructor({ translationEnabled = true }: { translationEnabled?: boolean } = {}) {
    this.translationEnabledValue = Boolean(translationEnabled);
    this.revision = 0;
    this.stableLineList = [];
    this.currentLine = null;
  }

  get stableLines(): readonly SubtitleLine[] {
    return this.stableLineList;
  }

  get translationEnabled(): boolean {
    return this.translationEnabledValue;
  }

  setTranslationEnabled(enabled: boolean): void {
    this.translationEnabledValue = Boolean(enabled);
  }

  applyEvent(event: RealtimeEvent): void {
    switch (event.type) {
      case "transcript_update":
        this.applyTranscriptUpdate(event);
        return;
      case "transcript_timing_update":
        this.applyTranscriptTimingUpdate(event);
        return;
      case "transcript_final":
        this.applyTranscriptFinal(event);
        return;
      case "translation_stable":
        this.applyStableTranslation(event);
        return;
      case "translation_preview":
        this.applyPreviewTranslation(event);
        return;
      case "translation_status":
        this.applyTranslationStatus(event);
        return;
      default:
        return;
    }
  }

  window({ includeTranslation = this.translationEnabledValue }: { includeTranslation?: boolean } = {}): SubtitleWindow {
    return {
      previous: renderLine(this.stableLineList.at(-1) || null, includeTranslation),
      current: renderLine(this.currentLine, includeTranslation),
    };
  }

  toSrt({ includeTranslation = this.translationEnabledValue }: { includeTranslation?: boolean } = {}): string {
    const blocks = [];
    let number = 1;
    for (const line of this.stableLineList) {
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

  private applyTranscriptUpdate(event: RealtimeEvent): void {
    const stableBase = toInt(event.stable_base, 0);
    const stableAppends = recordArray(event.stable_appends, "stable_appends");
    const stableCount = toInt(event.stable_count, 0);
    if (stableBase !== this.stableLineList.length) {
      throw new Error(`stable cursor mismatch: stable_base=${stableBase}, local_count=${this.stableLineList.length}`);
    }
    if (stableBase + stableAppends.length !== stableCount) {
      throw new Error(
        `stable count mismatch: stable_base=${stableBase}, appends=${stableAppends.length}, stable_count=${stableCount}`,
      );
    }

    const revision = toInt(event.revision, this.revision);
    const previousCurrent = this.currentLine;
    const resetCurrentPreview = stableAppends.length > 0;
    for (const segment of stableAppends) {
      this.stableLineList.push(preserveStableTranslation(lineFromSegment(segment, revision), previousCurrent));
    }

    const partial = optionalRecord(event.partial, "partial");
    const nextCurrent = partial ? lineFromSegment(partial, revision) : null;
    this.currentLine = resetCurrentPreview
      ? nextCurrent
      : preserveCurrentTranslation(nextCurrent, previousCurrent);
    this.revision = revision;
  }

  private applyTranscriptTimingUpdate(event: RealtimeEvent): void {
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

  private applyTranscriptFinal(event: RealtimeEvent): void {
    const existing = new Map(this.stableLineList.filter((line) => line.id).map((line) => [line.id as string, line]));
    const revision = toInt(event.revision, this.revision);
    const lines = [];
    for (const segment of recordArray(event.segments, "segments")) {
      let line = lineFromSegment(segment, revision);
      const previous = line.id ? existing.get(line.id) : undefined;
      if (previous) {
        line = line.withPatch(translationPatch(previous));
      }
      lines.push(line);
    }
    this.stableLineList = lines;
    this.currentLine = null;
    this.revision = revision;
  }

  private applyStableTranslation(event: RealtimeEvent): void {
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

  private applyPreviewTranslation(event: RealtimeEvent): void {
    if (!this.currentLine) {
      return;
    }
    const sourceRevision = toInt(event.source_revision, 0);
    const text = String(event.text || "").trim();
    if (text && this.currentLine.sourceRevision === sourceRevision) {
      this.currentLine = this.currentLine.withPatch({ translation: text });
    }
  }

  private applyTranslationStatus(event: RealtimeEvent): void {
    const index = this.stableIndex(event);
    if (index === null) {
      return;
    }
    this.patchStableLine(index, {
      translationStatus: String(event.code || ""),
      translationMessage: String(event.message || ""),
    });
  }

  private stableIndex(event: RealtimeEvent): number | null {
    const segmentId = String(event.source_segment_id || "");
    if (segmentId) {
      const index = this.stableLineList.findIndex((line) => line.id === segmentId);
      return index >= 0 ? index : null;
    }
    const segmentIndex = optionalInt(event.source_segment_index);
    if (!segmentIndex || segmentIndex <= 0) {
      return null;
    }
    const index = this.stableLineList.findIndex((line) => line.index === segmentIndex);
    return index >= 0 ? index : null;
  }

  private patchStableLine(index: number, patch: SubtitleLineInit): void {
    const line = this.stableLineList[index];
    if (line) {
      this.stableLineList[index] = line.withPatch(patch);
    }
  }
}

function lineFromSegment(segment: Record<string, unknown>, revision: number): SubtitleLineRecord {
  return new SubtitleLineRecord({
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

function renderLine(line: SubtitleLineRecord | null, includeTranslation: boolean): SubtitleLine | null {
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
  next: SubtitleLineRecord | null,
  previous: SubtitleLineRecord | null,
): SubtitleLineRecord | null {
  if (!next || !previous?.translation || !isSamePartialLine(next, previous)) {
    return next;
  }
  return next.withPatch(translationPatch(previous));
}

function preserveStableTranslation(
  next: SubtitleLineRecord,
  previous: SubtitleLineRecord | null,
): SubtitleLineRecord {
  if (!previous?.translation || !isSamePartialLine(next, previous)) {
    return next;
  }
  return next.withPatch(translationPatch(previous));
}

function translationPatch(line: SubtitleLine): SubtitleLineInit {
  return {
    translation: line.translation,
    translationStatus: line.translationStatus,
    translationMessage: line.translationMessage,
  };
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

function isInteger(value: unknown): value is number {
  return Number.isInteger(value);
}
