import { isRecord } from "./runtime-guards.js";

export const AUDIO_FRAME_EVENT = "audio-frame";
export const AUDIO_CAPTURE_ERROR_EVENT = "audio-capture-error";
export const AUDIO_CAPTURE_FAILED_MESSAGE = "Audio capture failed.";

export type Unlisten = () => void;

export interface AudioCaptureError {
  message: string;
}

export function parseAudioCaptureError(value: unknown): AudioCaptureError {
  if (isRecord(value) && typeof value.message === "string" && value.message) {
    return { message: value.message };
  }
  return { message: AUDIO_CAPTURE_FAILED_MESSAGE };
}
