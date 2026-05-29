import test from "node:test";
import assert from "node:assert/strict";

import { LiveAudioCapture } from "./live-audio-capture.js";
import type { StatusKey } from "./session-status.js";
import { createFakeAudioAdapter } from "./test-audio-adapter.fixture.js";

test("starts capture, forwards valid pcm, and cleans listeners on stop", async () => {
  const harness = createHarness();
  await harness.capture.start({
    sourceId: "system_default",
    sourceKind: "system",
    sendPcm: (bytes) => {
      harness.sentPcm.push(bytes);
      return true;
    },
  });

  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  harness.audio.frameHandler?.({ sampleRate: 48000, format: "pcm_s16le", dataBase64: "abcd" });
  await harness.capture.stop();

  assert.deepEqual(harness.audio.startCalls, ["system_default"]);
  assert.equal(harness.statuses.get("captureStatus"), "Sys");
  assert.deepEqual([...harness.sentPcm[0]!], [4]);
  assert.deepEqual(harness.statuses.get("audioStats"), { levelDb: null, droppedFrames: 0 });
  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
  assert.equal(harness.audio.unlistenFrames, 1);
  assert.equal(harness.audio.unlistenCaptureErrors, 1);
});

test("reports silent microphone capture and dropped frames", async () => {
  const harness = createHarness();
  await harness.capture.start({
    sourceId: "mic_default",
    sourceKind: "microphone",
    sendPcm: () => false,
  });

  for (let index = 0; index < 30; index += 1) {
    harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  }

  assert.equal(harness.statuses.get("captureStatus"), "Mic silent");
  assert.equal(harness.statuses.get("audioHealth"), "microphoneSilent");
  assert.deepEqual(harness.statuses.get("audioStats"), { levelDb: null, droppedFrames: 30 });
});

test("capture errors surface status and request abort", async () => {
  const harness = createHarness();
  await harness.capture.start({
    sourceId: "system_default",
    sourceKind: "system",
    sendPcm: () => true,
  });

  harness.audio.captureErrorHandler?.({ message: "device lost" });

  assert.equal(harness.statuses.get("captureStatus"), "device lost");
  assert.deepEqual(harness.abortMessages, ["device lost"]);
});

test("decode failures surface status and request abort without throwing from the frame handler", async () => {
  const harness = createHarness({ decodeError: new Error("invalid base64") });
  await harness.capture.start({
    sourceId: "system_default",
    sourceKind: "system",
    sendPcm: (bytes) => {
      harness.sentPcm.push(bytes);
      return true;
    },
  });

  assert.doesNotThrow(() => {
    harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "not-base64" });
  });

  assert.equal(harness.statuses.get("captureStatus"), "invalid base64");
  assert.deepEqual(harness.abortMessages, ["invalid base64"]);
  assert.deepEqual(harness.sentPcm, []);
});

test("start failure clears listeners before propagating", async () => {
  const harness = createHarness({ startError: new Error("permission denied") });

  await assert.rejects(
    harness.capture.start({
      sourceId: "system_default",
      sourceKind: "system",
      sendPcm: () => true,
    }),
    /permission denied/,
  );

  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
  assert.equal(harness.audio.unlistenFrames, 1);
  assert.equal(harness.audio.unlistenCaptureErrors, 1);
});

function createHarness({
  decodeError = null,
  startError = null,
}: {
  decodeError?: Error | null;
  startError?: Error | null;
} = {}) {
  const statuses = new Map<StatusKey, unknown>();
  const abortMessages: string[] = [];
  const sentPcm: Uint8Array[] = [];
  const audio = createFakeAudioAdapter({ decodeError, startError });
  const capture = new LiveAudioCapture({
    audio,
    onAbort: (message) => abortMessages.push(message),
    onStatus: (key, value) => statuses.set(key, value),
  });
  return { abortMessages, audio, capture, sentPcm, statuses };
}
