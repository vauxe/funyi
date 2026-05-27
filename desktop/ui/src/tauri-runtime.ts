import type { Unlisten } from "./audio-capture-events.js";

export interface TauriEvent<TPayload> {
  payload: TPayload;
}

export interface TauriRuntime {
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

export function tauriRuntime(): TauriRuntime | null {
  const maybeWindow = (globalThis as typeof globalThis & { window?: Window }).window;
  return maybeWindow?.__TAURI__ ?? null;
}
