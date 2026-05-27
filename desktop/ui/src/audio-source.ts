import { parseAudioSourceKind, type AudioSourceKind } from "./audio-source-kind.js";
import { requiredBoolean, requiredRecord, requiredString } from "./runtime-guards.js";

export interface AudioSource {
  id: string;
  name: string;
  kind: AudioSourceKind;
  isAvailable: boolean;
  detail: string;
}

export function parseAudioSources(value: unknown): AudioSource[] {
  if (!Array.isArray(value)) {
    throw new Error("audio sources payload must be an array");
  }
  return value.map((source, index) => parseAudioSource(source, index));
}

function parseAudioSource(value: unknown, index: number): AudioSource {
  const source = requiredRecord(value, `audio source ${index}`);
  return {
    id: requiredString(source.id, `audio source ${index}.id`),
    name: requiredString(source.name, `audio source ${index}.name`),
    kind: parseAudioSourceKind(source.kind, `audio source ${index}.kind`),
    isAvailable: requiredBoolean(source.isAvailable, `audio source ${index}.isAvailable`),
    detail: requiredString(source.detail, `audio source ${index}.detail`),
  };
}
