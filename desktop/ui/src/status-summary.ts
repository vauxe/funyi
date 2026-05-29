import { audioStatsState, type AudioLevelState, type AudioStatsState } from "./audio-level.js";
import { audioSourceKindFromAudioHealthStatus, type AudioSourceKind } from "./audio-source-kind.js";
import type { SessionState } from "./session-state.js";
import { FINAL_TRANSCRIPT_CANCELLED_MESSAGE, NO_AUDIO_SOURCE_MESSAGE, type StatusValues } from "./session-status.js";

export type StatusTone = "idle" | "active" | "warn" | "error";

export interface StatusSummary {
  text: string;
  tone: StatusTone;
  level?: AudioLevelState;
  volume?: number;
}

interface UserFacingErrorRule {
  message: string;
  matches(value: string): boolean;
}

const USER_FACING_ERROR_RULES: UserFacingErrorRule[] = [
  {
    matches: (value) => /Another realtime session is active/i.test(value),
    message: "Previous session closing",
  },
  {
    matches: (value) => value === NO_AUDIO_SOURCE_MESSAGE,
    message: "No audio source available.",
  },
  {
    matches: (value) => /^(?:WS error|WebSocket connection failed)/i.test(value),
    message: "Connection failed.",
  },
  {
    matches: (value) => /WebSocket closed before start/i.test(value),
    message: "Connection closed.",
  },
  {
    matches: (value) => /WebSocket closed/i.test(value),
    message: "Connection closed.",
  },
  {
    matches: (value) => /Timed out waiting for transcript_final/i.test(value),
    message: "Finish timed out.",
  },
  {
    matches: (value) => /invalid event/i.test(value),
    message: "Service sent an invalid response.",
  },
];

export function summarizeStatus(statusValues: StatusValues, sessionState: SessionState): StatusSummary {
  const error = currentUserVisibleError(statusValues, sessionState);
  if (sessionState !== "running") {
    if (error) {
      return { text: userFacingError(error), tone: "error" };
    }
    if (sessionState === "connecting") {
      return { text: "Connecting...", tone: "active" };
    }
    if (sessionState === "finishing") {
      return { text: "Finishing...", tone: "active" };
    }
    return { text: "", tone: "idle" };
  }

  const audioStats = audioStatsState(statusValues.audioStats);
  if (error) {
    return withAudioStats({ text: userFacingError(error), tone: "error" }, audioStats);
  }
  if (audioStats.hasDroppedFrames) {
    return withAudioStats({ text: "Audio lagging", tone: "warn" }, audioStats);
  }
  const silentSourceKind = audioSourceKindFromAudioHealthStatus(statusValues.audioHealth);
  if (silentSourceKind) {
    return withAudioStats({ text: silentCaptureSummary(silentSourceKind), tone: "warn" }, audioStats);
  }

  return withAudioStats({ text: "", tone: "idle" }, audioStats);
}

function withAudioStats(summary: StatusSummary, audioStats: AudioStatsState): StatusSummary {
  return { ...summary, level: audioStats.level, volume: audioStats.volume };
}

function currentUserVisibleError(statusValues: StatusValues, sessionState: SessionState): string {
  const captureStatus = statusValues.captureStatus;
  if (statusTextHasError(captureStatus)) {
    return captureStatus;
  }

  const connectionStatus = statusValues.connectionStatus;
  if (!connectionStatus || isLowLevelConnectionStatus(connectionStatus)) {
    return "";
  }
  if (statusTextHasError(connectionStatus)) {
    return connectionStatus;
  }
  return sessionState === "idle" && connectionStatus !== FINAL_TRANSCRIPT_CANCELLED_MESSAGE ? connectionStatus : "";
}

function isLowLevelConnectionStatus(value: string): boolean {
  return /^WS(?:\.\.\.| OK| closed)?$/i.test(value);
}

function userFacingError(value: string): string {
  return USER_FACING_ERROR_RULES.find((rule) => rule.matches(value))?.message ?? value;
}

function silentCaptureSummary(kind: AudioSourceKind): string {
  return kind === "microphone" ? "No mic audio" : "No system audio";
}

function statusTextHasError(value: string): boolean {
  return /error|failed|closed|timeout|timed out|lost|unavailable|unsupported|invalid|permission|denied|no native|no audio/i.test(
    value,
  );
}
