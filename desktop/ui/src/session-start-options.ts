import { AUDIO_FORMAT, AUDIO_SAMPLE_RATE } from "./audio-format.js";
import type { AudioSourceKind } from "./audio-source-kind.js";
import type { RealtimeStartPayload } from "./realtime-events.js";

const INVALID_AUDIO_SOURCE_MESSAGE = "Selected audio source is invalid.";
const INVALID_SERVER_URL_MESSAGE = "Server URL must be a ws:// address on the local machine.";

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[/u, "").replace(/\]$/u, "").toLowerCase();
  if (host === "localhost" || host === "::1") {
    return true;
  }
  // Exact dotted-quad in 127.0.0.0/8 only. Matching the raw hostname by prefix
  // would wrongly accept attacker-resolvable names like "127.0.0.1.evil.com".
  const octets = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/u.exec(host);
  if (!octets) {
    return false;
  }
  const parts = octets.slice(1).map(Number);
  return parts[0] === 127 && parts.every((value) => value <= 255);
}

// The transcript stream carries microphone/system audio, so the client must only
// ever connect to a loopback ws backend. This mirrors the Tauri CSP connect-src
// allowlist (ws:// loopback only) as defense in depth.
function validateServerUrl(rawUrl: string): { ok: true; url: string } | { ok: false; message: string } {
  const url = rawUrl.trim();
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { ok: false, message: INVALID_SERVER_URL_MESSAGE };
  }
  if (parsed.protocol !== "ws:") {
    return { ok: false, message: INVALID_SERVER_URL_MESSAGE };
  }
  if (parsed.username || parsed.password) {
    return { ok: false, message: INVALID_SERVER_URL_MESSAGE };
  }
  if (!isLoopbackHost(parsed.hostname)) {
    return { ok: false, message: INVALID_SERVER_URL_MESSAGE };
  }
  // Return the canonical, parsed form so the validated value and the value handed
  // to new WebSocket() are byte-identical (no parser differential).
  return { ok: true, url: parsed.href };
}

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

export type SessionStartResult = { ok: true; options: LiveSessionStartOptions } | { ok: false; message: string };

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

  const server = validateServerUrl(url);
  if (!server.ok) {
    return { ok: false, message: server.message };
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
      url: server.url,
      audioSourceId,
      audioSourceKind,
      startPayload,
    },
  };
}
