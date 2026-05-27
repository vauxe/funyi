import { parseAudioStatsState, type AudioLevelState } from "./audio-level.js";
import { audioSourceKindFromAudioHealthStatus, type AudioSourceKind } from "./audio-source-kind.js";
import type { SessionState } from "./session-state.js";
import {
  FINAL_TRANSCRIPT_CANCELLED_MESSAGE,
  NO_AUDIO_SOURCE_MESSAGE,
  type StatusValues,
} from "./session-status.js";

export type StatusTone = "idle" | "active" | "warn" | "error";

export interface StatusSummary {
  text: string;
  tone: StatusTone;
  level?: AudioLevelState;
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
    matches: (value) => /^WS error|WebSocket connection failed/i.test(value),
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

export function summarizeStatus(
  statusValues: StatusValues,
  sessionState: SessionState,
): StatusSummary {
  const error = currentUserVisibleError(statusValues, sessionState);
  if (error) {
    return { text: userFacingError(error), tone: "error" };
  }
  if (sessionState === "connecting") {
    return { text: "Connecting...", tone: "active" };
  }
  if (sessionState === "finishing") {
    return { text: "Finishing...", tone: "active" };
  }
  if (sessionState !== "running") {
    return { text: "", tone: "idle" };
  }

  const audioStats = parseAudioStatsState(statusValues.audioStats);
  if (audioStats.hasDroppedFrames) {
    return { text: "Audio lagging", tone: "warn", level: audioStats.level };
  }
  const silentSourceKind = audioSourceKindFromAudioHealthStatus(statusValues.audioHealth);
  if (silentSourceKind) {
    return { text: silentCaptureSummary(silentSourceKind), tone: "warn", level: audioStats.level };
  }

  return { text: "", tone: "idle", level: audioStats.level };
}

function currentUserVisibleError(
  statusValues: StatusValues,
  sessionState: SessionState,
): string {
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
  return sessionState === "idle" && connectionStatus !== FINAL_TRANSCRIPT_CANCELLED_MESSAGE
    ? connectionStatus
    : "";
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
  return /error|failed|closed|timeout|timed out|lost|unavailable|unsupported|invalid|permission|denied|no native|no audio/i.test(value);
}
