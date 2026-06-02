export type SessionState = "idle" | "connecting" | "running" | "paused" | "finishing";

export function isActiveSessionState(state: SessionState): boolean {
  return state !== "idle";
}

export function isLanguageConfigurationLocked(state: SessionState): boolean {
  return state === "connecting" || state === "paused" || state === "finishing";
}

export function isAudioSourceConfigurationLocked(state: SessionState): boolean {
  return state === "connecting" || state === "finishing";
}
