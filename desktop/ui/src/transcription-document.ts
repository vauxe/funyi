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
}

export interface TranscriptDocumentSnapshot {
  readonly schemaVersion: number;
  readonly durationMs: number | null;
  readonly language: string;
  readonly text: string;
  readonly segments: readonly TranscriptSegmentSnapshot[];
}

export function parseTranscriptDocumentSnapshot(payload: unknown): TranscriptDocumentSnapshot {
  if (!isRecord(payload)) {
    throw new Error("transcript document must be an object");
  }
  const schemaVersion = integerOrDefault(payload.schemaVersion, 1);
  if (schemaVersion !== 1) {
    throw new Error(`unsupported transcript document schema: ${schemaVersion}`);
  }
  const segments = recordArray(payload.segments, "segments").map(segmentFromRecord);
  return {
    schemaVersion,
    durationMs: optionalInteger(payload.durationMs),
    language: stringOrEmpty(payload.language),
    text: stringOrEmpty(payload.text),
    segments,
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
