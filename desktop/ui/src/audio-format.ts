import { isRecord } from "./runtime-guards.js";

export const AUDIO_SAMPLE_RATE = 16000;
export const AUDIO_FORMAT = "pcm_s16le";

export function decodeBase64Pcm(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export function isExpectedAudioFrame(
  frame: unknown,
): frame is { dataBase64: string; format: typeof AUDIO_FORMAT; sampleRate: typeof AUDIO_SAMPLE_RATE } {
  return (
    isRecord(frame) &&
    frame.sampleRate === AUDIO_SAMPLE_RATE &&
    frame.format === AUDIO_FORMAT &&
    typeof frame.dataBase64 === "string"
  );
}
