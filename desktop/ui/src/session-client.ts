import type { LanguageConfigUpdate, RealtimeEvent, RealtimeStartPayload } from "./realtime-events.js";

export interface LiveSessionClient {
  close(): void | Promise<void>;
  connect(startPayload: RealtimeStartPayload): Promise<void>;
  finish(): void;
  setLanguageConfig(config: LanguageConfigUpdate): void;
  sendPcm(bytes: Uint8Array): boolean;
}

export interface LiveSessionClientCallbacks {
  url: string;
  onClose: (event: CloseEvent, source: LiveSessionClient) => void | Promise<void>;
  onError: (event: Event, source: LiveSessionClient) => void;
  onEvent: (event: RealtimeEvent, source: LiveSessionClient) => void | Promise<void>;
  onStatus: (status: string, source: LiveSessionClient) => void;
}
