import test from "node:test";
import assert from "node:assert/strict";

import { transcribeFile } from "./transcription-client.js";

test("posts a raw file to the offline transcription endpoint and parses the snapshot", async () => {
  const file = namedBlob("clip.wav", "audio/wav");
  const seen: { init: RequestInit | null; url: string } = { init: null, url: "" };
  const events: string[] = [];
  const restore = stubFetch(async (url, init) => {
    seen.url = String(url);
    seen.init = init ?? null;
    return ndjsonResponse([
      {
        type: "transcript_update",
        revision: 1,
        stable_base: 0,
        stable_count: 1,
        stable_appends: [
          {
            id: "seg_000001",
            index: 1,
            start_ms: 0,
            end_ms: 1200,
            text: "你好",
            language: "Chinese",
            timing_status: "aligned",
          },
        ],
        partial: null,
      },
      {
        type: "translation_stable",
        source_revision: 1,
        source_segment_id: "seg_000001",
        source_segment_index: 1,
        source_segment_ids: ["seg_000001"],
        source_segment_indices: [1],
        text: "hello",
        target_language: "English",
      },
      {
        type: "transcript_final",
        revision: 1,
        final_revision: 1,
        stable_count: 1,
        document: {
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
              translationStatus: "ok",
            },
          ],
          translationUnits: [
            {
              text: "hello",
              targetLanguage: "English",
              sourceSegmentIds: ["seg_000001"],
              sourceSegmentIndices: [1],
            },
          ],
        },
      },
    ]);
  });

  try {
    const snapshot = await transcribeFile({
      file,
      language: " Chinese ",
      realtimeUrl: "ws://127.0.0.1:8000/ws/asr",
      onEvent: (event) => events.push(String(event.type || "")),
      targetLanguage: " English ",
    });

    assert.equal(
      seen.url,
      "http://127.0.0.1:8000/api/transcriptions/stream?language=Chinese&targetLanguage=English&filename=clip.wav",
    );
    assert.equal(seen.init?.method, "POST");
    assert.equal(seen.init?.body, file);
    assert.deepEqual(seen.init?.headers, { Accept: "application/x-ndjson", "Content-Type": "audio/wav" });
    assert.deepEqual(events, ["transcript_update", "translation_stable", "transcript_final"]);
    assert.equal(snapshot.segments[0]?.text, "你好");
    assert.equal(snapshot.segments[0]?.translation, "hello");
    assert.equal(snapshot.segments[0]?.translationStatus, "ok");
    assert.equal(snapshot.translationUnits[0]?.text, "hello");
    assert.deepEqual(snapshot.translationUnits[0]?.sourceSegmentIds, ["seg_000001"]);
  } finally {
    restore();
  }
});

test("cancels the stream reader when a stream error event is received", async () => {
  let cancelled = false;
  const restore = stubFetch(async () =>
    errorStreamResponse({ message: "model failed" }, () => {
      cancelled = true;
    }),
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
      /model failed/,
    );
    assert.equal(cancelled, true);
  } finally {
    restore();
  }
});

test("parses streamed NDJSON across chunk and UTF-8 boundaries without a trailing newline", async () => {
  const update = {
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: 1,
    stable_appends: [
      {
        id: "seg_000001",
        index: 1,
        start_ms: 0,
        end_ms: 1200,
        text: "你好",
        language: "Chinese",
      },
    ],
    partial: null,
  };
  const final = {
    type: "transcript_final",
    revision: 1,
    final_revision: 1,
    stable_count: 1,
    document: {
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
        },
      ],
    },
  };
  const streamText = `${JSON.stringify(update)}\n${JSON.stringify(final)}`;
  const encoder = new TextEncoder();
  const firstChineseByte = encoder.encode(streamText.slice(0, streamText.indexOf("你好"))).length;
  const encodedLength = encoder.encode(streamText).length;
  const restore = stubFetch(async () =>
    chunkedTextResponse(streamText, [firstChineseByte + 1, firstChineseByte + 4, encodedLength - 2]),
  );
  const events: string[] = [];

  try {
    const snapshot = await transcribeFile({
      file: namedBlob("clip.wav", "audio/wav"),
      language: null,
      realtimeUrl: "ws://127.0.0.1:8000/ws/asr",
      onEvent: (event) => events.push(String(event.type || "")),
      targetLanguage: "",
    });

    assert.deepEqual(events, ["transcript_update", "transcript_final"]);
    assert.equal(snapshot.segments[0]?.text, "你好");
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

function ndjsonResponse(payloads: readonly unknown[]): Response {
  const encoder = new TextEncoder();
  return {
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const payload of payloads) {
          controller.enqueue(encoder.encode(`${JSON.stringify(payload)}\n`));
        }
        controller.close();
      },
    }),
    ok: true,
    status: 200,
  } as Response;
}

function chunkedTextResponse(text: string, splitPoints: readonly number[]): Response {
  const encoded = new TextEncoder().encode(text);
  return {
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        let offset = 0;
        for (const splitPoint of splitPoints) {
          controller.enqueue(encoded.slice(offset, splitPoint));
          offset = splitPoint;
        }
        controller.enqueue(encoded.slice(offset));
        controller.close();
      },
    }),
    ok: true,
    status: 200,
  } as Response;
}

function errorStreamResponse(error: unknown, onCancel: () => void): Response {
  const encoder = new TextEncoder();
  return {
    body: new ReadableStream<Uint8Array>({
      cancel() {
        onCancel();
      },
      start(controller) {
        controller.enqueue(encoder.encode(`${JSON.stringify({ type: "error", error, fatal: true })}\n`));
      },
    }),
    ok: true,
    status: 200,
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
