import { isInteger, isRecord, recordArray } from "./runtime-guards.js";

export interface TranscriptSegmentSnapshot {
  readonly id: string | null;
  readonly index: number | null;
  readonly startMs: number | null;
  readonly endMs: number | null;
  readonly text: string;
  readonly language: string;
  readonly timingStatus: string | null;
  readonly translation: string | null;
  readonly translationStatus: string | null;
  readonly translationMessage: string | null;
}

export interface TranscriptTranslationUnitSnapshot {
  readonly text: string;
  readonly targetLanguage: string;
  readonly sourceSegmentIds: readonly string[];
  readonly sourceSegmentIndices: readonly number[];
  readonly translationStatus: string | null;
  readonly translationMessage: string | null;
}

export interface TranscriptDocumentSnapshot {
  readonly schemaVersion: number;
  readonly durationMs: number | null;
  readonly language: string;
  readonly text: string;
  readonly segments: readonly TranscriptSegmentSnapshot[];
  readonly translationUnits: readonly TranscriptTranslationUnitSnapshot[];
}

export function parseTranscriptDocumentSnapshot(payload: unknown): TranscriptDocumentSnapshot {
  if (!isRecord(payload)) {
    throw new Error("transcript document must be an object");
  }
  const schemaVersion = integerOrDefault(payload.schemaVersion, 1);
  if (schemaVersion !== 1) {
    throw new Error(`unsupported transcript document schema: ${schemaVersion}`);
  }
  return {
    schemaVersion,
    durationMs: optionalInteger(payload.durationMs),
    language: stringOrEmpty(payload.language),
    text: stringOrEmpty(payload.text),
    segments: recordArray(payload.segments, "segments").map(segmentFromRecord),
    translationUnits: recordArray(payload.translationUnits, "translationUnits").map(translationUnitFromRecord),
  };
}

function segmentFromRecord(segment: Record<string, unknown>): TranscriptSegmentSnapshot {
  return {
    id: optionalString(segment.id),
    index: optionalInteger(segment.index),
    startMs: optionalInteger(segment.startMs),
    endMs: optionalInteger(segment.endMs),
    text: stringOrEmpty(segment.text).trim(),
    language: stringOrEmpty(segment.language),
    timingStatus: optionalString(segment.timingStatus),
    translation: optionalString(segment.translation),
    translationStatus: optionalString(segment.translationStatus),
    translationMessage: optionalString(segment.translationMessage),
  };
}

function translationUnitFromRecord(unit: Record<string, unknown>): TranscriptTranslationUnitSnapshot {
  return {
    text: stringOrEmpty(unit.text).trim(),
    targetLanguage: stringOrEmpty(unit.targetLanguage),
    sourceSegmentIds: stringArray(unit.sourceSegmentIds),
    sourceSegmentIndices: integerArray(unit.sourceSegmentIndices),
    translationStatus: optionalString(unit.translationStatus),
    translationMessage: optionalString(unit.translationMessage),
  };
}

function integerOrDefault(value: unknown, fallback: number): number {
  return isInteger(value) ? value : fallback;
}

function optionalInteger(value: unknown): number | null {
  return isInteger(value) ? value : null;
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

function stringOrEmpty(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => stringOrEmpty(item).trim()).filter((item) => item.length > 0);
}

function integerArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is number => isInteger(item) && item > 0);
}
