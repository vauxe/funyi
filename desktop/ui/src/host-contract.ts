import type { AudioCaptureError, Unlisten } from "./audio-capture-events.js";
import type { AudioSource } from "./audio-source.js";
import type { ResizeDirection } from "./overlay-contract.js";

export interface AudioCaptureHost {
  listAudioSources(): Promise<AudioSource[]>;
  startAudioCapture(sourceId: string): Promise<void>;
  stopAudioCapture(): Promise<void>;
  listenAudioFrames(handler: (frame: unknown) => void): Promise<Unlisten>;
  listenAudioCaptureErrors(handler: (error: AudioCaptureError) => void): Promise<Unlisten>;
}

export interface OverlayHost {
  startOverlayDrag(): Promise<void>;
  updateOverlayDrag(): Promise<void>;
  endOverlayDrag(): Promise<void>;
  startOverlayResize(direction: ResizeDirection): Promise<void>;
  updateOverlayResize(): Promise<void>;
  endOverlayResize(): Promise<void>;
  minimizeOverlay(): Promise<void>;
  closeOverlay(): Promise<void>;
}

export type DesktopHost = AudioCaptureHost & OverlayHost;
