import test from "node:test";
import assert from "node:assert/strict";

import {
  INVALID_SERVER_URL_MESSAGE,
  transcriptionUrlFromRealtimeUrl,
  validateRealtimeServerUrl,
} from "./server-url.js";

test("validates loopback realtime websocket urls", () => {
  for (const url of ["ws://127.0.0.1:8000/ws/asr", "ws://localhost:8000/ws/asr", "ws://127.0.0.5:9000/ws"]) {
    assert.deepEqual(validateRealtimeServerUrl(url), { ok: true, url }, url);
  }
});

test("derives the offline transcription endpoint from the realtime endpoint", () => {
  assert.deepEqual(transcriptionUrlFromRealtimeUrl(" ws://127.0.0.1:8000/ws/asr?debug=1#x "), {
    ok: true,
    url: "http://127.0.0.1:8000/api/transcriptions",
  });
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
  test(`rejects non-loopback realtime url: ${url}`, () => {
    assert.deepEqual(validateRealtimeServerUrl(url), { ok: false, message: INVALID_SERVER_URL_MESSAGE });
    assert.deepEqual(transcriptionUrlFromRealtimeUrl(url), { ok: false, message: INVALID_SERVER_URL_MESSAGE });
  });
}
