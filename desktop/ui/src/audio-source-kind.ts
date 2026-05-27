export type AudioSourceKind = "system" | "microphone";
export type SilentAudioHealthStatus = "systemSilent" | "microphoneSilent";

export function parseAudioSourceKind(value: unknown, fieldName: string): AudioSourceKind {
  if (value === "system" || value === "microphone") {
    return value;
  }
  throw new Error(`${fieldName} must be system or microphone`);
}

export function audioSourceShortLabel(kind: AudioSourceKind): string {
  return kind === "microphone" ? "Mic" : "Sys";
}

export function audioSourceDefaultName(kind: AudioSourceKind): string {
  return kind === "microphone" ? "Microphone" : "Audio";
}

export function silentAudioSourceStatus(kind: AudioSourceKind): string {
  return `${audioSourceShortLabel(kind)} silent`;
}

export function silentAudioHealthStatus(kind: AudioSourceKind): SilentAudioHealthStatus {
  return kind === "microphone" ? "microphoneSilent" : "systemSilent";
}

export function audioSourceKindFromAudioHealthStatus(status: SilentAudioHealthStatus | ""): AudioSourceKind | null {
  if (status === "microphoneSilent") {
    return "microphone";
  }
  if (status === "systemSilent") {
    return "system";
  }
  return null;
}
