export type SessionState = "idle" | "connecting" | "running" | "finishing";

export function isActiveSessionState(state: SessionState): boolean {
  return state !== "idle";
}

export function isSessionConfigurationLocked(state: SessionState): boolean {
  return state === "connecting" || state === "finishing";
}
