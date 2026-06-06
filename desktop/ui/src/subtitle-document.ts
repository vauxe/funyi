import type { RealtimeEvent } from "./realtime-events.js";
import { isInteger, optionalRecord, recordArray } from "./runtime-guards.js";
import type {
  TranscriptDocumentSnapshot,
  TranscriptSegmentSnapshot,
  TranscriptTranslationUnitSnapshot,
} from "./transcription-document.js";
import { parseTranscriptDocumentSnapshot } from "./transcription-document.js";

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
  private pendingStableUnitIds: string[];
  private translationEnabledValue: boolean;

  constructor({ translationEnabled = true }: { translationEnabled?: boolean } = {}) {
    this.translationEnabledValue = translationEnabled;
    this.revision = 0;
    this.showLatestStableAsCurrent = false;
    this.stableLineList = [];
    this.pendingStableUnitIds = [];
    this.currentLine = null;
  }

  get stableLines(): readonly SubtitleLine[] {
    return this.stableDisplayLines();
  }

  get translationEnabled(): boolean {
    return this.translationEnabledValue;
  }

  setTranslationEnabled(enabled: boolean): void {
    this.translationEnabledValue = enabled;
  }

  replaceSnapshot(snapshot: TranscriptDocumentSnapshot): void {
    this.revision += 1;
    this.currentLine = null;
    this.showLatestStableAsCurrent = snapshot.segments.length > 0;
    this.pendingStableUnitIds = [];
    this.stableLineList = linesFromSnapshot(snapshot, this.revision);
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
      this.currentDisplayLine() || (this.showLatestStableAsCurrent ? this.stableDisplayLines().at(-1) || null : null);
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
    const partial = optionalRecord(event.partial, "partial");
    for (const segment of stableAppends) {
      const line = preserveStableTranslation(lineFromSegment(segment, revision), previousCurrent);
      this.stableLineList.push(line);
      if (partial && line.id) {
        this.pendingStableUnitIds.push(line.id);
      }
    }

    const nextCurrent = partial ? lineFromSegment(partial, revision) : null;
    if (nextCurrent) {
      this.currentLine = resetCurrentPreview ? nextCurrent : preserveCurrentTranslation(nextCurrent, previousCurrent);
      this.showLatestStableAsCurrent = false;
    } else if (stableAppends.length > 0) {
      this.currentLine = null;
      this.showLatestStableAsCurrent = true;
      this.pendingStableUnitIds = [];
    } else if (!this.showLatestStableAsCurrent) {
      this.currentLine = null;
      this.pendingStableUnitIds = [];
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
      this.pendingStableUnitIds = [];
      this.revision = toInt(event.revision, this.revision);
      return;
    }

    const existingById = new Map(
      this.stableLineList.filter((line) => line.id).map((line) => [line.id as string, line]),
    );
    const existingByIndex = new Map(
      this.stableLineList
        .filter((line): line is SubtitleLine & { index: number } => isInteger(line.index))
        .map((line) => [line.index, line]),
    );
    const revision = toInt(event.revision, this.revision);
    const lines = [];
    for (const segment of recordArray(event.segments, "segments")) {
      let line = lineFromSegment(segment, revision);
      const previous = previousStableLine(line, existingById, existingByIndex);
      if (previous && !segmentHasTranslationState(segment)) {
        line = patchLine(line, translationPatch(previous));
      }
      lines.push(line);
    }
    const documentPayload = optionalRecord(event.document, "document");
    if (documentPayload) {
      applyTranslationUnits(lines, parseTranscriptDocumentSnapshot(documentPayload).translationUnits);
    }
    this.stableLineList = lines;
    this.pendingStableUnitIds = [];
    this.currentLine = null;
    this.showLatestStableAsCurrent = false;
    this.revision = revision;
  }

  private applyStableTranslation(event: RealtimeEvent): void {
    const text = String(event.text || "").trim();
    if (!text) {
      return;
    }
    if (this.clearTranslationCoveragePending(event)) {
      this.clearCurrentTranslationPreview();
    }
    this.patchStableAnchor(event, {
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
    const patch = {
      translation: null,
      translationStatus: String(event.code || ""),
      translationMessage: String(event.message || ""),
    };
    if (this.clearTranslationCoveragePending(event)) {
      this.clearCurrentTranslationPreview();
    }
    this.patchStableAnchor(event, patch);
  }

  private patchStableAnchor(event: RealtimeEvent, patch: SubtitleLineInit): void {
    const index = this.stableIndex(event);
    if (index === null) {
      return;
    }
    this.clearPendingStableLine(index);
    this.patchStableLine(index, patch);
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

  private clearPendingStableLine(index: number): void {
    const id = this.stableLineList[index]?.id;
    if (id) {
      this.pendingStableUnitIds = this.pendingStableUnitIds.filter((pendingId) => pendingId !== id);
    }
  }

  private clearTranslationCoveragePending(event: RealtimeEvent): boolean {
    const pendingIds = this.translationCoverageIds(event);
    const previousCount = this.pendingStableUnitIds.length;
    if (pendingIds.size === 0 || previousCount === 0) {
      return false;
    }
    this.pendingStableUnitIds = this.pendingStableUnitIds.filter((id) => !pendingIds.has(id));
    return this.pendingStableUnitIds.length < previousCount;
  }

  private translationCoverageIds(event: RealtimeEvent): Set<string> {
    const ids = new Set(stringArray(event.source_segment_ids));
    const anchorId = String(event.source_segment_id || "");
    if (anchorId) {
      ids.add(anchorId);
    }
    const indices = new Set(intArray(event.source_segment_indices));
    const anchorIndex = optionalInt(event.source_segment_index);
    if (anchorIndex) {
      indices.add(anchorIndex);
    }
    if (indices.size > 0) {
      for (const line of this.stableLineList) {
        if (line.id && line.index !== null && indices.has(line.index)) {
          ids.add(line.id);
        }
      }
    }
    return ids;
  }

  private clearCurrentTranslationPreview(): void {
    if (!this.currentLine?.translation) {
      return;
    }
    this.currentLine = patchLine(this.currentLine, {
      translation: null,
      translationStatus: null,
      translationMessage: null,
    });
  }

  private currentDisplayLine(): SubtitleLine | null {
    if (!this.currentLine) {
      return null;
    }
    if (!this.translationEnabledValue || !this.currentLine.translation) {
      return this.currentLine;
    }
    const pendingLines = this.pendingStableLines();
    if (pendingLines.length === 0) {
      return this.currentLine;
    }
    const unitLine = combinedSourceLine([...pendingLines, this.currentLine], {
      id: this.currentLine.id,
      index: this.currentLine.index,
      translation: this.currentLine.translation,
      translationStatus: this.currentLine.translationStatus,
      translationMessage: this.currentLine.translationMessage,
    });
    return patchLine(unitLine, { sourceRevision: this.currentLine.sourceRevision });
  }

  private stableDisplayLines(): readonly SubtitleLine[] {
    if (!this.translationEnabledValue) {
      return this.stableLineList;
    }
    const pendingIds = new Set(this.currentLine?.translation ? this.pendingStableUnitIds : []);
    return this.stableLineList.filter((line) => !(line.id && pendingIds.has(line.id)));
  }

  private pendingStableLines(): SubtitleLine[] {
    return this.pendingStableUnitIds
      .map((id) => this.stableLineList.find((line) => line.id === id))
      .filter((line): line is SubtitleLine => Boolean(line));
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
    translation: stringOrNull(segment.translation),
    translationStatus: stringOrNull(segment.translation_status),
    translationMessage: stringOrNull(segment.translation_message),
  });
}

function lineFromSnapshotSegment(
  segment: TranscriptSegmentSnapshot,
  revision: number,
  fallbackIndex: number,
): SubtitleLine {
  return createSubtitleLine({
    id: segment.id,
    index: segment.index ?? fallbackIndex,
    startMs: segment.startMs,
    endMs: segment.endMs,
    text: segment.text,
    language: segment.language,
    sourceRevision: revision,
    timingStatus: segment.timingStatus,
    translation: segment.translation,
    translationStatus: segment.translationStatus ?? null,
    translationMessage: segment.translationMessage ?? null,
  });
}

function linesFromSnapshot(snapshot: TranscriptDocumentSnapshot, revision: number): SubtitleLine[] {
  const lines = snapshot.segments.map((segment, offset) => lineFromSnapshotSegment(segment, revision, offset + 1));
  applyTranslationUnits(lines, snapshot.translationUnits);
  return lines;
}

function applyTranslationUnits(lines: SubtitleLine[], units: readonly TranscriptTranslationUnitSnapshot[]): void {
  for (const unit of units) {
    const text = unit.text.trim();
    if (!text) {
      continue;
    }
    const anchorOffset = snapshotTranslationUnitAnchorOffset(lines, unit);
    if (anchorOffset === null) {
      continue;
    }
    const anchor = lines[anchorOffset];
    if (anchor) {
      lines[anchorOffset] = patchLine(anchor, {
        translation: text,
        translationStatus: null,
        translationMessage: null,
      });
    }
  }
}

function snapshotTranslationUnitAnchorOffset(
  segments: readonly Pick<SubtitleLine, "id" | "index">[],
  unit: TranscriptTranslationUnitSnapshot,
): number | null {
  const coverageLength = Math.max(unit.sourceSegmentIds.length, unit.sourceSegmentIndices.length);
  for (let coverageOffset = coverageLength - 1; coverageOffset >= 0; coverageOffset -= 1) {
    const id = unit.sourceSegmentIds[coverageOffset];
    if (id) {
      const offset = segments.findIndex((segment) => segment.id === id);
      if (offset >= 0) {
        return offset;
      }
    }
    const index = unit.sourceSegmentIndices[coverageOffset];
    if (index !== undefined) {
      const offset = segments.findIndex((segment) => segment.index === index);
      if (offset >= 0) {
        return offset;
      }
    }
  }
  return null;
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

function previousStableLine(
  line: SubtitleLine,
  byId: ReadonlyMap<string, SubtitleLine>,
  byIndex: ReadonlyMap<number, SubtitleLine>,
): SubtitleLine | undefined {
  if (line.id) {
    const previous = byId.get(line.id);
    if (previous) {
      return previous;
    }
  }
  return isInteger(line.index) ? byIndex.get(line.index) : undefined;
}

function combinedSourceLine(sourceLines: readonly SubtitleLine[], patch: SubtitleLineInit = {}): SubtitleLine {
  const first = sourceLines[0] || null;
  const last = sourceLines.at(-1) || null;
  return createSubtitleLine({
    id: patch.id ?? last?.id ?? null,
    index: patch.index ?? last?.index ?? null,
    startMs: first?.startMs ?? null,
    endMs: last?.endMs ?? first?.endMs ?? null,
    text: joinSourceTexts(sourceLines.map((line) => line.text)),
    language: first?.language || "",
    sourceRevision: maxNullableInt(sourceLines.map((line) => line.sourceRevision)),
    timingStatus: combinedTimingStatus(sourceLines),
    translation: patch.translation ?? null,
    translationStatus: patch.translationStatus ?? null,
    translationMessage: patch.translationMessage ?? null,
  });
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

function joinSourceTexts(texts: readonly string[]): string {
  let joined = "";
  for (const rawText of texts) {
    const text = String(rawText || "").trim();
    if (!text) {
      continue;
    }
    if (!joined) {
      joined = text;
      continue;
    }
    if (needsAsciiWordSeparator(joined.at(-1) || "", text[0] || "")) {
      joined += " ";
    }
    joined += text;
  }
  return joined.trim();
}

function needsAsciiWordSeparator(left: string, right: string): boolean {
  return Boolean(left && right && isAsciiAlphaNum(left) && isAsciiAlphaNum(right));
}

function isAsciiAlphaNum(value: string): boolean {
  return /^[0-9A-Za-z]$/u.test(value);
}

function combinedTimingStatus(lines: readonly SubtitleLine[]): string | null {
  const statuses = lines.map((line) => line.timingStatus).filter((status): status is string => Boolean(status));
  if (statuses.includes("failed")) {
    return "failed";
  }
  return statuses.length === lines.length && statuses.every((status) => status === "aligned") ? "aligned" : null;
}

function maxNullableInt(values: readonly (number | null)[]): number | null {
  const integers = values.filter((value): value is number => isInteger(value));
  return integers.length > 0 ? Math.max(...integers) : null;
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

function segmentHasTranslationState(segment: Record<string, unknown>): boolean {
  return "translation" in segment || "translation_status" in segment || "translation_message" in segment;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item || "").trim()).filter((item) => item.length > 0);
}

function intArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => optionalInt(item)).filter((item): item is number => item !== null);
}
