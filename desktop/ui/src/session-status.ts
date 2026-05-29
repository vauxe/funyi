import type { AudioStats } from "./audio-level.js";
import type { SilentAudioHealthStatus } from "./audio-source-kind.js";

export const NO_AUDIO_SOURCE_MESSAGE = "No native audio source available.";
export const FINAL_TRANSCRIPT_CANCELLED_MESSAGE = "Final transcript cancelled.";

export type AudioHealthStatus = "" | SilentAudioHealthStatus;

export interface StatusValues {
  audioHealth: AudioHealthStatus;
  audioStats: AudioStats;
  captureStatus: string;
  connectionStatus: string;
}

export type StatusKey = keyof StatusValues;
export type StatusValue<K extends StatusKey> = StatusValues[K];
