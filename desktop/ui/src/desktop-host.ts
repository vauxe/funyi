import {
  AUDIO_CAPTURE_ERROR_EVENT,
  AUDIO_FRAME_EVENT,
  parseAudioCaptureError,
  type AudioCaptureError,
  type Unlisten,
} from "./audio-capture-events.js";
import { parseAudioSources, type AudioSource } from "./audio-source.js";
import type { DesktopHost } from "./host-contract.js";
import type { ResizeDirection } from "./overlay-contract.js";
import { tauriRuntime } from "./tauri-runtime.js";

export const DESKTOP_COMMANDS = {
  closeOverlay: "close_overlay",
  endOverlayDrag: "end_overlay_drag",
  endOverlayResize: "end_overlay_resize",
  listAudioSources: "list_audio_sources",
  minimizeOverlay: "minimize_overlay",
  startAudioCapture: "start_audio_capture",
  startOverlayDrag: "start_overlay_drag",
  startOverlayResize: "start_overlay_resize",
  stopAudioCapture: "stop_audio_capture",
  updateOverlayDrag: "update_overlay_drag",
  updateOverlayResize: "update_overlay_resize",
} as const;

export const desktopHost: DesktopHost = {
  async listAudioSources(): Promise<AudioSource[]> {
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
    return parseAudioSources(await runtime.core.invoke<unknown>(DESKTOP_COMMANDS.listAudioSources));
  },

  async startAudioCapture(sourceId: string): Promise<void> {
    await invokeRequired(DESKTOP_COMMANDS.startAudioCapture, { sourceId }, "Native audio capture requires Tauri.");
  },

  async stopAudioCapture(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.stopAudioCapture);
  },

  async listenAudioFrames(handler: (frame: unknown) => void): Promise<Unlisten> {
    return listenOptional(AUDIO_FRAME_EVENT, handler);
  },

  async listenAudioCaptureErrors(handler: (error: AudioCaptureError) => void): Promise<Unlisten> {
    return listenOptional(AUDIO_CAPTURE_ERROR_EVENT, (payload) => handler(parseAudioCaptureError(payload)));
  },

  async startOverlayDrag(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.startOverlayDrag);
  },

  async updateOverlayDrag(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.updateOverlayDrag);
  },

  async endOverlayDrag(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.endOverlayDrag);
  },

  async startOverlayResize(direction: ResizeDirection): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.startOverlayResize, { direction });
  },

  async updateOverlayResize(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.updateOverlayResize);
  },

  async endOverlayResize(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.endOverlayResize);
  },

  async minimizeOverlay(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.minimizeOverlay);
  },

  async closeOverlay(): Promise<void> {
    await invokeOptional(DESKTOP_COMMANDS.closeOverlay);
  },
};

async function invokeRequired<TResult>(
  command: string,
  args: Record<string, unknown> | undefined,
  unavailableMessage: string,
): Promise<TResult> {
  const runtime = tauriRuntime();
  if (!runtime) {
    throw new Error(unavailableMessage);
  }
  return runtime.core.invoke<TResult>(command, args);
}

async function invokeOptional<TResult>(command: string, args?: Record<string, unknown>): Promise<TResult | undefined> {
  const runtime = tauriRuntime();
  if (!runtime) {
    return undefined;
  }
  return runtime.core.invoke<TResult>(command, args);
}

async function listenOptional<TPayload>(eventName: string, handler: (payload: TPayload) => void): Promise<Unlisten> {
  const runtime = tauriRuntime();
  if (!runtime) {
    return () => {};
  }
  return runtime.event.listen<TPayload>(eventName, (event) => handler(event.payload));
}
