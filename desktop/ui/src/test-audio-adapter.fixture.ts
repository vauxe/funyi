import type { AudioAdapter } from "./audio-adapter.js";
import type { AudioCaptureError, Unlisten } from "./audio-capture-events.js";
import type { AudioSource } from "./audio-source.js";

interface FakeAudioAdapterOptions {
  decodeError?: Error | null;
  listSourcesError?: Error | null;
  sources?: AudioSource[];
  startError?: Error | null;
}

export interface FakeAudioAdapter extends AudioAdapter {
  captureErrorHandler: ((payload: AudioCaptureError) => void) | null;
  frameHandler: ((frame: unknown) => void) | null;
  startCalls: string[];
  stopCalls: number;
  unlistenCaptureErrors: number;
  unlistenFrames: number;
}

export function createFakeAudioAdapter({
  decodeError = null,
  listSourcesError = null,
  sources = [],
  startError = null,
}: FakeAudioAdapterOptions = {}): FakeAudioAdapter {
  const audio = {
    captureErrorHandler: null as ((payload: AudioCaptureError) => void) | null,
    frameHandler: null as ((frame: unknown) => void) | null,
    startCalls: [] as string[],
    stopCalls: 0,
    unlistenCaptureErrors: 0,
    unlistenFrames: 0,
    decodePcm: (base64: string) => {
      if (decodeError) {
        throw decodeError;
      }
      return new Uint8Array([base64.length]);
    },
    listenCaptureErrors: async (handler: (payload: AudioCaptureError) => void): Promise<Unlisten> => {
      audio.captureErrorHandler = handler;
      return () => {
        audio.captureErrorHandler = null;
        audio.unlistenCaptureErrors += 1;
      };
    },
    listenFrames: async (handler: (frame: unknown) => void): Promise<Unlisten> => {
      audio.frameHandler = handler;
      return () => {
        audio.frameHandler = null;
        audio.unlistenFrames += 1;
      };
    },
    listSources: async () => {
      if (listSourcesError) {
        throw listSourcesError;
      }
      return sources;
    },
    startCapture: async (sourceId: string) => {
      audio.startCalls.push(sourceId);
      if (startError) {
        throw startError;
      }
    },
    stopCapture: async () => {
      audio.stopCalls += 1;
    },
  } satisfies FakeAudioAdapter;

  return audio;
}
