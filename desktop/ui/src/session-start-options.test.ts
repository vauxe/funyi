import test from "node:test";
import assert from "node:assert/strict";

import { buildSessionStartOptions } from "./session-start-options.js";

test("builds deterministic live session start options", () => {
  const result = buildSessionStartOptions({
    url: " ws://127.0.0.1:8000/ws/asr ",
    audioSourceId: "system_default",
    audioSourceKind: "system",
    asrLanguage: " Chinese ",
    targetLanguage: " Japanese ",
    now: () => 123,
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.deepEqual(result.options, {
    url: "ws://127.0.0.1:8000/ws/asr",
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: {
      type: "start",
      session_id: "desktop-123",
      sample_rate: 16000,
      audio_format: "pcm_s16le",
      language: "Chinese",
      target_language: "Japanese",
    },
  });
});

test("omits empty optional language fields", () => {
  const result = buildSessionStartOptions({
    url: "ws://127.0.0.1:8000/ws/asr",
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
    asrLanguage: null,
    targetLanguage: "",
    now: () => 456,
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.deepEqual(result.options.startPayload, {
    type: "start",
    session_id: "desktop-456",
    sample_rate: 16000,
    audio_format: "pcm_s16le",
  });
});

test("rejects unknown selected audio source kind", () => {
  const result = buildSessionStartOptions({
    url: "ws://127.0.0.1:8000/ws/asr",
    audioSourceId: "missing",
    audioSourceKind: null,
    asrLanguage: null,
    targetLanguage: "",
  });

  assert.deepEqual(result, {
    ok: false,
    message: "Selected audio source is invalid.",
  });
});

const INVALID_SERVER_URL = "Server URL must be a ws:// address on the local machine.";

function buildWithUrl(url: string): ReturnType<typeof buildSessionStartOptions> {
  return buildSessionStartOptions({
    url,
    audioSourceId: "system_default",
    audioSourceKind: "system",
    asrLanguage: null,
    targetLanguage: "",
  });
}

test("accepts ws loopback server urls", () => {
  for (const url of ["ws://127.0.0.1:8000/ws/asr", "ws://localhost:8000/ws/asr", "ws://127.0.0.5:9000/ws"]) {
    const result = buildWithUrl(url);
    assert.equal(result.ok, true, url);
    if (result.ok) {
      assert.equal(result.options.url, url);
    }
  }
});

for (const url of [
  "wss://localhost:8443/ws/asr",
  "http://127.0.0.1:8000/ws/asr",
  "ws://evil.example.com:8000/ws/asr",
  "ws://127.0.0.1.evil.com:8000/ws/asr",
  "ws://127.evil.com:8000/ws/asr",
  "ws://0.0.0.0:8000/ws/asr",
  "ws://127.0.0.300:8000/ws/asr",
  "ws://[::1]:8000/ws/asr",
  "ws://[::2]:8000/ws/asr",
  "ws://user:pass@127.0.0.1:8000/ws/asr",
  "not a url",
]) {
  test(`rejects non-loopback ws url: ${url}`, () => {
    assert.deepEqual(buildWithUrl(url), { ok: false, message: INVALID_SERVER_URL });
  });
}
