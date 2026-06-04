import test from "node:test";
import assert from "node:assert/strict";

import { transcribeFile } from "./transcription-client.js";

test("posts a raw file to the offline transcription endpoint and parses the snapshot", async () => {
  const file = namedBlob("clip.wav", "audio/wav");
  const seen: { init: RequestInit | null; url: string } = { init: null, url: "" };
  const restore = stubFetch(async (url, init) => {
    seen.url = String(url);
    seen.init = init ?? null;
    return jsonResponse({
      schemaVersion: 1,
      durationMs: 1200,
      language: "Chinese",
      text: "你好",
      segments: [
        {
          id: "seg_000001",
          index: 1,
          startMs: 0,
          endMs: 1200,
          text: "你好",
          language: "Chinese",
          translation: "hello",
        },
      ],
    });
  });

  try {
    const snapshot = await transcribeFile({
      file,
      language: " Chinese ",
      realtimeUrl: "ws://127.0.0.1:8000/ws/asr",
      targetLanguage: " English ",
    });

    assert.equal(
      seen.url,
      "http://127.0.0.1:8000/api/transcriptions?language=Chinese&targetLanguage=English&filename=clip.wav",
    );
    assert.equal(seen.init?.method, "POST");
    assert.equal(seen.init?.body, file);
    assert.deepEqual(seen.init?.headers, { "Content-Type": "audio/wav" });
    assert.equal(snapshot.segments[0]?.text, "你好");
    assert.equal(snapshot.segments[0]?.translation, "hello");
  } finally {
    restore();
  }
});

test("surfaces service error messages", async () => {
  const restore = stubFetch(async () =>
    jsonResponse({ error: { code: "busy", message: "Another transcription session is active." } }, false, 409),
  );

  try {
    await assert.rejects(
      () =>
        transcribeFile({
          file: namedBlob("clip.wav", "audio/wav"),
          language: null,
          realtimeUrl: "ws://127.0.0.1:8000/ws/asr",
          targetLanguage: "",
        }),
      /Another transcription session is active/,
    );
  } finally {
    restore();
  }
});

function namedBlob(name: string, type: string): Blob {
  const blob = new Blob(["audio"], { type });
  Object.defineProperty(blob, "name", { configurable: true, value: name });
  return blob;
}

function jsonResponse(payload: unknown, ok = true, status = 200): Response {
  return {
    json: async () => payload,
    ok,
    status,
  } as Response;
}

function stubFetch(implementation: typeof fetch): () => void {
  const previous = globalThis.fetch;
  Object.defineProperty(globalThis, "fetch", {
    configurable: true,
    value: implementation,
    writable: true,
  });
  return () => {
    Object.defineProperty(globalThis, "fetch", {
      configurable: true,
      value: previous,
      writable: true,
    });
  };
}
