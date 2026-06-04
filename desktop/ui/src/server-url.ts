export const INVALID_SERVER_URL_MESSAGE = "Server URL must be a ws:// address on the local machine.";

export type UrlValidationResult = { ok: true; url: string } | { ok: false; message: string };

function isCspAllowedLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[/u, "").replace(/\]$/u, "").toLowerCase();
  if (host === "localhost") {
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

// The transcript paths carry microphone/system/file audio, so the client must
// only ever connect to a loopback backend. This mirrors the Tauri CSP
// connect-src allowlist as defense in depth.
export function validateRealtimeServerUrl(rawUrl: string): UrlValidationResult {
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
  if (!isCspAllowedLoopbackHost(parsed.hostname)) {
    return { ok: false, message: INVALID_SERVER_URL_MESSAGE };
  }
  // Return the canonical, parsed form so the validated value and the value handed
  // to new WebSocket() are byte-identical (no parser differential).
  return { ok: true, url: parsed.href };
}

export function transcriptionUrlFromRealtimeUrl(rawUrl: string): UrlValidationResult {
  const realtime = validateRealtimeServerUrl(rawUrl);
  if (!realtime.ok) {
    return realtime;
  }
  const url = new URL(realtime.url);
  url.protocol = "http:";
  url.pathname = "/api/transcriptions";
  url.search = "";
  url.hash = "";
  return { ok: true, url: url.href };
}

export function transcriptionStreamUrlFromRealtimeUrl(rawUrl: string): UrlValidationResult {
  const endpoint = transcriptionUrlFromRealtimeUrl(rawUrl);
  if (!endpoint.ok) {
    return endpoint;
  }
  const url = new URL(endpoint.url);
  url.pathname = "/api/transcriptions/stream";
  return { ok: true, url: url.href };
}
