import type { RealtimeEvent } from "./realtime-events.js";
import { isInteger, optionalRecord, recordArray } from "./runtime-guards.js";
import type { TranscriptDocumentSnapshot, TranscriptSegmentSnapshot } from "./transcription-document.js";
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

interface StableTranslationUnit {
  readonly sourceSegmentIds: readonly string[];
  readonly sourceSegmentIndices: readonly number[];
  readonly text: string | null;
  readonly translationStatus: string | null;
  readonly translationMessage: string | null;
}

interface SourceLineIndex {
  readonly offsetById: ReadonlyMap<string, number>;
  readonly offsetByIndex: ReadonlyMap<number, number>;
}

export class SubtitleDocument {
  private currentLine: SubtitleLine | null;
  private revision: number;
  private showLatestStableAsCurrent: boolean;
  private stableLineList: SubtitleLine[];
  private stableLineOffsetById: Map<string, number>;
  private stableLineOffsetByIndex: Map<number, number>;
  private stableProjection: SubtitleLine[] | null;
  private stableRenderRevision: number;
  private stableTranslationUnitIndex: Map<string, number>;
  private stableTranslationUnits: StableTranslationUnit[];
  private pendingStableUnitIds: string[];
  private translationEnabledValue: boolean;

  constructor({ translationEnabled = true }: { translationEnabled?: boolean } = {}) {
    this.translationEnabledValue = translationEnabled;
    this.revision = 0;
    this.showLatestStableAsCurrent = false;
    this.stableLineList = [];
    this.stableLineOffsetById = new Map();
    this.stableLineOffsetByIndex = new Map();
    this.stableProjection = null;
    this.stableRenderRevision = 0;
    this.stableTranslationUnitIndex = new Map();
    this.stableTranslationUnits = [];
    this.pendingStableUnitIds = [];
    this.currentLine = null;
  }

  get stableLines(): readonly SubtitleLine[] {
    return this.projectedStableLines();
  }

  get stableRenderVersion(): number {
    return this.stableRenderRevision;
  }

  exportLines(): readonly SubtitleLine[] {
    return this.projectedStableLines();
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
    this.rebuildStableLineIndex();
    this.replaceStableTranslationUnits(translationUnitsFromSnapshot(snapshot));
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
      this.currentDisplayLine() || (this.showLatestStableAsCurrent ? this.projectedStableLines().at(-1) || null : null);
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
      const line = lineFromSegment(segment, revision);
      const offset = this.stableLineList.length;
      this.stableLineList.push(line);
      this.indexStableLine(offset, line);
      if (partial && line.id) {
        this.pendingStableUnitIds.push(line.id);
      }
    }
    if (stableAppends.length > 0) {
      this.invalidateStableProjection();
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

    const revision = toInt(event.revision, this.revision);
    const segmentRecords = recordArray(event.segments, "segments");
    const lines = [];
    const segmentTranslationUnits = [];
    const explicitlyTranslatedSegmentIds = new Set<string>();
    const explicitlyTranslatedSegmentIndices = new Set<number>();
    for (const segment of segmentRecords) {
      const line = lineFromSegment(segment, revision);
      if (segmentHasTranslationState(segment)) {
        if (line.id) {
          explicitlyTranslatedSegmentIds.add(line.id);
        }
        if (line.index !== null) {
          explicitlyTranslatedSegmentIndices.add(line.index);
        }
        const unit = translationUnitFromSegment(segment);
        if (unit) {
          segmentTranslationUnits.push(unit);
        }
      }
      lines.push(line);
    }
    this.stableLineList = lines;
    this.rebuildStableLineIndex();
    const documentPayload = optionalRecord(event.document, "document");
    if (documentPayload) {
      this.replaceStableTranslationUnits(
        translationUnitsFromSnapshot(parseTranscriptDocumentSnapshot(documentPayload)),
      );
    } else if (explicitlyTranslatedSegmentIds.size > 0 || explicitlyTranslatedSegmentIndices.size > 0) {
      this.replaceStableTranslationUnits(
        this.stableTranslationUnits
          .filter(
            (unit) =>
              !translationUnitTouchesCoverage(unit, explicitlyTranslatedSegmentIds, explicitlyTranslatedSegmentIndices),
          )
          .concat(segmentTranslationUnits),
      );
    }
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
    this.upsertStableTranslationUnit(translationUnitFromEvent(event, { text }));
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
      translationStatus: String(event.code || ""),
      translationMessage: String(event.message || ""),
    };
    if (this.clearTranslationCoveragePending(event)) {
      this.clearCurrentTranslationPreview();
    }
    this.upsertStableTranslationUnit(translationUnitFromEvent(event, patch));
  }

  private stableIndex(event: RealtimeEvent): number | null {
    const segmentId = String(event.source_segment_id || "");
    if (segmentId) {
      const index = this.stableLineOffsetById.get(segmentId);
      if (index !== undefined) {
        return index;
      }
      // Fall through to index-based lookup: a transcript_final rebuild can reassign
      // ids while the 1-based index still matches (parity with the Python model).
    }
    const segmentIndex = optionalInt(event.source_segment_index);
    if (!segmentIndex || segmentIndex <= 0) {
      return null;
    }
    return this.stableLineOffsetByIndex.get(segmentIndex) ?? null;
  }

  private patchStableLine(index: number, patch: SubtitleLineInit): void {
    const line = this.stableLineList[index];
    if (line) {
      this.stableLineList[index] = patchLine(line, patch);
      this.invalidateStableProjection();
    }
  }

  private clearTranslationCoveragePending(event: RealtimeEvent): boolean {
    const previousCount = this.pendingStableUnitIds.length;
    if (previousCount === 0) {
      return false;
    }
    const pendingIds = this.translationCoverageIds(event);
    if (pendingIds.size === 0) {
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
      for (const index of indices) {
        const offset = this.stableLineOffsetByIndex.get(index);
        const line = offset === undefined ? undefined : this.stableLineList[offset];
        if (line?.id) {
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

  private projectedStableLines(): readonly SubtitleLine[] {
    if (!this.translationEnabledValue) {
      return this.stableLineList;
    }
    if (!this.stableProjection) {
      this.stableProjection = projectStableLines(this.stableLineList, this.stableTranslationUnits);
    }
    return this.stableProjection;
  }

  private rebuildStableLineIndex(): void {
    this.stableLineOffsetById = new Map();
    this.stableLineOffsetByIndex = new Map();
    this.stableLineList.forEach((line, offset) => {
      this.indexStableLine(offset, line);
    });
    this.invalidateStableProjection();
  }

  private indexStableLine(offset: number, line: SubtitleLine): void {
    if (line.id) {
      this.stableLineOffsetById.set(line.id, offset);
    }
    if (line.index !== null) {
      this.stableLineOffsetByIndex.set(line.index, offset);
    }
  }

  private invalidateStableProjection(): void {
    this.stableRenderRevision += 1;
    this.stableProjection = null;
  }

  private tryPatchStableProjection(unit: StableTranslationUnit): boolean {
    if (!this.stableProjection || this.stableProjection.length !== this.stableLineList.length) {
      return false;
    }
    const offset = this.stableOffsetForSingleTranslationUnit(unit);
    const sourceLine = offset === null ? undefined : this.stableLineList[offset];
    if (offset === null || !sourceLine) {
      return false;
    }
    this.stableProjection[offset] = patchLine(sourceLine, {
      translation: unit.text,
      translationStatus: unit.translationStatus,
      translationMessage: unit.translationMessage,
    });
    this.stableRenderRevision += 1;
    return true;
  }

  private stableOffsetForSingleTranslationUnit(unit: StableTranslationUnit): number | null {
    if (translationCoverageLength(unit) !== 1) {
      return null;
    }
    const index = unit.sourceSegmentIndices[0];
    if (index !== undefined) {
      return this.stableLineOffsetByIndex.get(index) ?? null;
    }
    const id = unit.sourceSegmentIds[0];
    return id ? (this.stableLineOffsetById.get(id) ?? null) : null;
  }

  private pendingStableLines(): SubtitleLine[] {
    return this.pendingStableUnitIds
      .map((id) => {
        const offset = this.stableLineOffsetById.get(id);
        return offset === undefined ? undefined : this.stableLineList[offset];
      })
      .filter((line): line is SubtitleLine => Boolean(line));
  }

  private upsertStableTranslationUnit(next: StableTranslationUnit): void {
    const key = translationCoverageKey(next);
    if (!key) {
      return;
    }
    const index = this.stableTranslationUnitIndex.get(key);
    if (index === undefined) {
      this.stableTranslationUnitIndex.set(key, this.stableTranslationUnits.length);
      this.stableTranslationUnits.push(next);
      if (!this.tryPatchStableProjection(next)) {
        this.invalidateStableProjection();
      }
      return;
    }
    const previous = this.stableTranslationUnits[index];
    if (!previous) {
      return;
    }
    const merged = {
      ...next,
      text: next.text ?? previous.text,
    };
    this.stableTranslationUnits[index] = merged;
    if (!this.tryPatchStableProjection(merged)) {
      this.invalidateStableProjection();
    }
  }

  private replaceStableTranslationUnits(units: StableTranslationUnit[]): void {
    this.stableTranslationUnits = units;
    this.stableTranslationUnitIndex = translationUnitIndex(units);
    this.invalidateStableProjection();
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
  });
}

function linesFromSnapshot(snapshot: TranscriptDocumentSnapshot, revision: number): SubtitleLine[] {
  return snapshot.segments.map((segment, offset) => lineFromSnapshotSegment(segment, revision, offset + 1));
}

function translationUnitsFromSnapshot(snapshot: TranscriptDocumentSnapshot): StableTranslationUnit[] {
  return [
    ...snapshot.translationUnits
      .map((unit) => ({
        sourceSegmentIds: unit.sourceSegmentIds,
        sourceSegmentIndices: unit.sourceSegmentIndices,
        text: unit.text.trim() || null,
        translationStatus: unit.translationStatus,
        translationMessage: unit.translationMessage,
      }))
      .filter(
        (unit) =>
          (unit.text || unit.translationStatus || unit.translationMessage) &&
          (unit.sourceSegmentIds.length > 0 || unit.sourceSegmentIndices.length > 0),
      ),
    ...snapshot.segments.map(translationUnitFromSnapshotSegment).filter(isStableTranslationUnit),
  ];
}

function translationUnitFromSnapshotSegment(segment: TranscriptSegmentSnapshot): StableTranslationUnit | null {
  if (!segment.translation && !segment.translationStatus && !segment.translationMessage) {
    return null;
  }
  return {
    sourceSegmentIds: segment.id ? [segment.id] : [],
    sourceSegmentIndices: isInteger(segment.index) ? [segment.index] : [],
    text: segment.translation,
    translationStatus: segment.translationStatus ?? null,
    translationMessage: segment.translationMessage ?? null,
  };
}

function translationUnitFromSegment(segment: Record<string, unknown>): StableTranslationUnit | null {
  if (!segmentHasTranslationState(segment)) {
    return null;
  }
  const id = stringOrNull(segment.id);
  const index = optionalInt(segment.index);
  if (!id && !index) {
    return null;
  }
  return {
    sourceSegmentIds: id ? [id] : [],
    sourceSegmentIndices: index ? [index] : [],
    text: stringOrNull(segment.translation),
    translationStatus: stringOrNull(segment.translation_status),
    translationMessage: stringOrNull(segment.translation_message),
  };
}

function isStableTranslationUnit(unit: StableTranslationUnit | null): unit is StableTranslationUnit {
  return Boolean(unit);
}

function translationUnitFromEvent(
  event: RealtimeEvent,
  patch: { text?: string; translationStatus?: string; translationMessage?: string },
): StableTranslationUnit {
  const sourceSegmentIds = stringArray(event.source_segment_ids);
  const sourceSegmentIndices = intArray(event.source_segment_indices);
  const anchorId = stringOrNull(event.source_segment_id);
  const anchorIndex = optionalInt(event.source_segment_index);
  return {
    sourceSegmentIds: sourceSegmentIds.length > 0 ? sourceSegmentIds : anchorId ? [anchorId] : [],
    sourceSegmentIndices: sourceSegmentIndices.length > 0 ? sourceSegmentIndices : anchorIndex ? [anchorIndex] : [],
    text: patch.text ?? null,
    translationStatus: patch.translationStatus ?? null,
    translationMessage: patch.translationMessage ?? null,
  };
}

function projectStableLines(
  sourceLines: readonly SubtitleLine[],
  translationUnits: readonly StableTranslationUnit[],
): SubtitleLine[] {
  const sourceIndex = buildSourceLineIndex(sourceLines);
  const resolvedByStartOffset = new Map<number, { unit: StableTranslationUnit; offsets: number[] }>();
  for (const unit of translationUnits) {
    const offsets = resolveTranslationUnitOffsets(sourceIndex, unit);
    const startOffset = offsets?.[0];
    if (offsets && startOffset !== undefined && !resolvedByStartOffset.has(startOffset)) {
      resolvedByStartOffset.set(startOffset, { unit, offsets });
    }
  }

  const projected: SubtitleLine[] = [];
  for (let offset = 0; offset < sourceLines.length; ) {
    const line = sourceLines[offset];
    if (!line) {
      offset += 1;
      continue;
    }
    const match = resolvedByStartOffset.get(offset);
    if (match) {
      const coveredLines: SubtitleLine[] = [];
      for (const coveredOffset of match.offsets) {
        const coveredLine = sourceLines[coveredOffset];
        if (coveredLine) {
          coveredLines.push(coveredLine);
        }
      }
      const lastOffset = Math.max(...match.offsets);
      projected.push(
        combinedSourceLine(coveredLines, {
          id: coveredLines.at(-1)?.id ?? null,
          index: coveredLines.at(-1)?.index ?? null,
          translation: match.unit.text,
          translationStatus: match.unit.translationStatus,
          translationMessage: match.unit.translationMessage,
        }),
      );
      offset = Math.max(offset + 1, lastOffset + 1);
      continue;
    }
    projected.push(line);
    offset += 1;
  }
  return projected;
}

function resolveTranslationUnitOffsets(sourceIndex: SourceLineIndex, unit: StableTranslationUnit): number[] | null {
  const offsets: number[] = [];
  const coverageLength = translationCoverageLength(unit);
  if (coverageLength === 0) {
    return null;
  }
  for (let coverageOffset = 0; coverageOffset < coverageLength; coverageOffset += 1) {
    const id = unit.sourceSegmentIds[coverageOffset];
    let offset = id ? (sourceIndex.offsetById.get(id) ?? -1) : -1;
    const index = unit.sourceSegmentIndices[coverageOffset];
    if (offset < 0 && index !== undefined) {
      offset = sourceIndex.offsetByIndex.get(index) ?? -1;
    }
    if (offset < 0) {
      return null;
    }
    const previousOffset = offsets.at(-1);
    if (previousOffset !== undefined && offset !== previousOffset + 1) {
      return null;
    }
    offsets.push(offset);
  }
  return offsets;
}

function buildSourceLineIndex(sourceLines: readonly SubtitleLine[]): SourceLineIndex {
  const offsetById = new Map<string, number>();
  const offsetByIndex = new Map<number, number>();
  sourceLines.forEach((line, offset) => {
    if (line.id) {
      offsetById.set(line.id, offset);
    }
    if (line.index !== null) {
      offsetByIndex.set(line.index, offset);
    }
  });
  return { offsetById, offsetByIndex };
}

function translationUnitTouchesCoverage(
  unit: StableTranslationUnit,
  segmentIds: ReadonlySet<string>,
  segmentIndices: ReadonlySet<number>,
): boolean {
  return (
    unit.sourceSegmentIds.some((id) => segmentIds.has(id)) ||
    unit.sourceSegmentIndices.some((index) => segmentIndices.has(index))
  );
}

function translationUnitIndex(units: readonly StableTranslationUnit[]): Map<string, number> {
  const index = new Map<string, number>();
  units.forEach((unit, offset) => {
    const key = translationCoverageKey(unit);
    if (key && !index.has(key)) {
      index.set(key, offset);
    }
  });
  return index;
}

function translationCoverageKey(unit: StableTranslationUnit): string | null {
  if (unit.sourceSegmentIndices.length > 0) {
    return `index:${unit.sourceSegmentIndices.join(",")}`;
  }
  if (unit.sourceSegmentIds.length > 0) {
    return `id:${unit.sourceSegmentIds.join("\u001f")}`;
  }
  return null;
}

function translationCoverageLength(unit: StableTranslationUnit): number {
  return Math.max(unit.sourceSegmentIds.length, unit.sourceSegmentIndices.length);
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

function combinedSourceLine(sourceLines: readonly SubtitleLine[], patch: SubtitleLineInit = {}): SubtitleLine {
  const first = sourceLines[0] || null;
  const last = sourceLines.at(-1) || null;
  return createSubtitleLine({
    id: patch.id ?? last?.id ?? null,
    index: patch.index ?? last?.index ?? null,
    startMs: first?.startMs ?? null,
    endMs: last?.endMs ?? null,
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
  return Boolean(left && right && isAsciiAlphaNum(right) && (isAsciiAlphaNum(left) || ",.!?;:".includes(left)));
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
  const numbers = values.filter((value): value is number => isInteger(value));
  return numbers.length > 0 ? Math.max(...numbers) : null;
}

function stringOrNull(value: unknown): string | null {
  const text = typeof value === "string" ? value.trim() : "";
  return text || null;
}

function optionalInt(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numberValue = Number(value);
  return Number.isInteger(numberValue) ? numberValue : null;
}

function toInt(value: unknown, fallback: number): number {
  return optionalInt(value) ?? fallback;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item || "").trim()).filter(Boolean);
}

function intArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map(optionalInt).filter((item): item is number => item !== null);
}

function segmentHasTranslationState(segment: Record<string, unknown>): boolean {
  return "translation" in segment || "translation_status" in segment || "translation_message" in segment;
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
