export const AUDIO_FRAME_EVENT = "audio-frame";
export const AUDIO_CAPTURE_ERROR_EVENT = "audio-capture-error";

export interface AudioSource {
  id: string;
  name: string;
  kind: string;
  isAvailable: boolean;
  detail: string;
}

export interface AudioFrame {
  seq?: number;
  sampleRate: number;
  format: string;
  dataBase64: string;
}

export interface AudioCaptureError {
  message: string;
}

export type Unlisten = () => void;

interface TauriEvent<TPayload> {
  payload: TPayload;
}

interface TauriRuntime {
  core: {
    invoke<TResult>(command: string, args?: Record<string, unknown>): Promise<TResult>;
  };
  event: {
    listen<TPayload>(
      event: string,
      handler: (event: TauriEvent<TPayload>) => void,
    ): Promise<Unlisten>;
  };
}

declare global {
  interface Window {
    __TAURI__?: TauriRuntime;
  }
}

export function isTauriRuntime(): boolean {
  return tauriRuntime() !== null;
}

export async function listAudioSources(): Promise<AudioSource[]> {
  const runtime = tauriRuntime();
  if (!runtime) {
    return [
      {
        id: "browser-dev",
        name: "Native system audio is available only in Tauri",
        kind: "system",
        isAvailable: false,
        detail: "Run this UI with pnpm run dev from desktop/.",
      },
    ];
  }
  return runtime.core.invoke<AudioSource[]>("list_audio_sources");
}

export async function startAudioCapture(sourceId: string): Promise<void> {
  const runtime = tauriRuntime();
  if (!runtime) {
    throw new Error("Native audio capture requires Tauri.");
  }
  await runtime.core.invoke("start_audio_capture", { sourceId });
}

export async function stopAudioCapture(): Promise<void> {
  const runtime = tauriRuntime();
  if (!runtime) {
    return;
  }
  await runtime.core.invoke("stop_audio_capture");
}

export async function listenAudioFrames(handler: (frame: AudioFrame) => void): Promise<Unlisten> {
  const runtime = tauriRuntime();
  if (!runtime) {
    return () => {};
  }
  return runtime.event.listen<AudioFrame>(AUDIO_FRAME_EVENT, (event) => handler(event.payload));
}

export async function listenAudioCaptureErrors(handler: (error: AudioCaptureError) => void): Promise<Unlisten> {
  const runtime = tauriRuntime();
  if (!runtime) {
    return () => {};
  }
  return runtime.event.listen<AudioCaptureError>(
    AUDIO_CAPTURE_ERROR_EVENT,
    (event) => handler(event.payload),
  );
}

export function decodeBase64Pcm(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function tauriRuntime(): TauriRuntime | null {
  const maybeWindow = (globalThis as typeof globalThis & { window?: Window }).window;
  const runtime = maybeWindow?.__TAURI__;
  return runtime ?? null;
}
