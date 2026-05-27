import type { AudioCaptureError, Unlisten } from "./audio-capture-events.js";
import type { AudioSource } from "./audio-source.js";

export interface AudioAdapter {
  decodePcm(base64: string): Uint8Array;
  listSources(): Promise<AudioSource[]>;
  listenCaptureErrors(handler: (payload: AudioCaptureError) => void): Promise<Unlisten>;
  listenFrames(handler: (frame: unknown) => void): Promise<Unlisten>;
  startCapture(sourceId: string): Promise<void>;
  stopCapture(): Promise<void>;
}
