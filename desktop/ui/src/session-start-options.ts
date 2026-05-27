import { AUDIO_FORMAT, AUDIO_SAMPLE_RATE } from "./audio-format.js";
import type { AudioSourceKind } from "./audio-source-kind.js";
import type { RealtimeStartPayload } from "./realtime-events.js";

const INVALID_AUDIO_SOURCE_MESSAGE = "Selected audio source is invalid.";

export interface SessionStartInput {
  url: string;
  audioSourceId: string;
  audioSourceKind: AudioSourceKind | null;
  asrLanguage: string | null;
  targetLanguage: string;
  now?: () => number;
}

export interface LiveSessionStartOptions {
  url: string;
  startPayload: RealtimeStartPayload;
  audioSourceId: string;
  audioSourceKind: AudioSourceKind;
}

export type SessionStartResult =
  | { ok: true; options: LiveSessionStartOptions }
  | { ok: false; message: string };

export function buildSessionStartOptions({
  url,
  audioSourceId,
  audioSourceKind,
  asrLanguage,
  targetLanguage,
  now = Date.now,
}: SessionStartInput): SessionStartResult {
  if (!audioSourceKind) {
    return { ok: false, message: INVALID_AUDIO_SOURCE_MESSAGE };
  }

  const startPayload: RealtimeStartPayload = {
    type: "start",
    session_id: `desktop-${now()}`,
    sample_rate: AUDIO_SAMPLE_RATE,
    audio_format: AUDIO_FORMAT,
  };

  const language = asrLanguage?.trim();
  if (language) {
    startPayload.language = language;
  }

  const target = targetLanguage.trim();
  if (target) {
    startPayload.target_language = target;
  }

  return {
    ok: true,
    options: {
      url: url.trim(),
      audioSourceId,
      audioSourceKind,
      startPayload,
    },
  };
}
