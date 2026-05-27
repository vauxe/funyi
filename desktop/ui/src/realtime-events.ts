import { isRecord } from "./runtime-guards.js";

export interface RealtimeEvent extends Record<string, unknown> {
  type?: string;
}

export interface LanguageConfigUpdate {
  language?: string | null;
  target_language?: string | null;
}

export interface RealtimeStartPayload extends Record<string, unknown> {
  type: "start";
  session_id?: string;
  sample_rate: number;
  audio_format?: string;
  language?: string;
  target_language?: string;
}

export function parseRealtimeEventMessage(message: string): RealtimeEvent {
  const event = JSON.parse(message);
  if (!isRecord(event)) {
    throw new Error("event payload must be an object");
  }
  if (event.type !== undefined && typeof event.type !== "string") {
    throw new Error("event type must be a string");
  }
  return event;
}

export function readyEventTranslationEnabled(event: RealtimeEvent): boolean {
  const translation = event.translation;
  return isRecord(translation) && translation.enabled === true;
}
