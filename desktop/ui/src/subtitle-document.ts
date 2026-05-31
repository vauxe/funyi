import type { RealtimeEvent } from "./realtime-events.js";
import { isInteger, optionalRecord, recordArray } from "./runtime-guards.js";

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

export class SubtitleDocument {
  private currentLine: SubtitleLine | null;
  private revision: number;
  private showLatestStableAsCurrent: boolean;
  private stableLineList: SubtitleLine[];
  private translationEnabledValue: boolean;

  constructor({ translationEnabled = true }: { translationEnabled?: boolean } = {}) {
    this.translationEnabledValue = translationEnabled;
    this.revision = 0;
    this.showLatestStableAsCurrent = false;
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
    this.translationEnabledValue = enabled;
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

  window(): SubtitleWindow {
    const includeTranslation = this.translationEnabledValue;
    const currentLine =
      this.currentLine || (this.showLatestStableAsCurrent ? this.stableLineList.at(-1) || null : null);
    return {
      current: renderLine(currentLine, includeTranslation),
    };
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
    if (nextCurrent) {
      this.currentLine = resetCurrentPreview ? nextCurrent : preserveCurrentTranslation(nextCurrent, previousCurrent);
      this.showLatestStableAsCurrent = false;
    } else if (stableAppends.length > 0) {
      this.currentLine = null;
      this.showLatestStableAsCurrent = true;
    } else if (!this.showLatestStableAsCurrent) {
      this.currentLine = null;
    }
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
    // An unbounded session's transcript_final may OMIT `segments` entirely, which
    // means "keep the stable history already replayed from transcript_update" — only
    // a present `segments` array is a full-snapshot rebuild (an empty array still
    // legitimately clears). The key-absent vs key-present distinction is the
    // contract; letting recordArray() collapse both to [] would erase the whole
    // transcript on the terminal event of a streaming session. Mirror the Python
    // model, which guards on `"segments" not in event` (events are JSON, so a
    // missing key reads as undefined here).
    if (event.segments === undefined) {
      this.currentLine = null;
      this.showLatestStableAsCurrent = false;
      this.revision = toInt(event.revision, this.revision);
      return;
    }

    const existing = new Map(this.stableLineList.filter((line) => line.id).map((line) => [line.id as string, line]));
    const revision = toInt(event.revision, this.revision);
    const lines = [];
    for (const segment of recordArray(event.segments, "segments")) {
      let line = lineFromSegment(segment, revision);
      const previous = line.id ? existing.get(line.id) : undefined;
      if (previous) {
        line = patchLine(line, translationPatch(previous));
      }
      lines.push(line);
    }
    this.stableLineList = lines;
    this.currentLine = null;
    this.showLatestStableAsCurrent = false;
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
      this.currentLine = patchLine(this.currentLine, { translation: text });
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
      if (index >= 0) {
        return index;
      }
      // Fall through to index-based lookup: a transcript_final rebuild can reassign
      // ids while the 1-based index still matches (parity with the Python model).
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
      this.stableLineList[index] = patchLine(line, patch);
    }
  }
}

function lineFromSegment(segment: Record<string, unknown>, revision: number): SubtitleLine {
  return createSubtitleLine({
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
  return patchLine(line, {
    translation: null,
    translationStatus: null,
    translationMessage: null,
  });
}

function preserveCurrentTranslation(next: SubtitleLine | null, previous: SubtitleLine | null): SubtitleLine | null {
  if (!next || !previous?.translation || !isSamePartialLine(next, previous)) {
    return next;
  }
  return patchLine(next, translationPatch(previous));
}

function preserveStableTranslation(next: SubtitleLine, previous: SubtitleLine | null): SubtitleLine {
  if (!previous?.translation || !isSamePartialLine(next, previous)) {
    return next;
  }
  return patchLine(next, translationPatch(previous));
}

function createSubtitleLine({
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
}: SubtitleLineInit = {}): SubtitleLine {
  return {
    id,
    index,
    startMs,
    endMs,
    text,
    language,
    sourceRevision,
    timingStatus,
    translation,
    translationStatus,
    translationMessage,
  };
}

function patchLine(line: SubtitleLine, patch: SubtitleLineInit): SubtitleLine {
  return createSubtitleLine({ ...line, ...patch });
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
  if (isInteger(left.startMs) && left.startMs === right.startMs) {
    return true;
  }
  return (
    Boolean(leftText && rightText) &&
    (leftText === rightText || leftText.startsWith(rightText) || rightText.startsWith(leftText))
  );
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
