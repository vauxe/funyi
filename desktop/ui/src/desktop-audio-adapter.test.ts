import test from "node:test";
import assert from "node:assert/strict";

import { createDesktopAudioAdapter } from "./desktop-audio-adapter.js";
import type { AudioCaptureError } from "./audio-capture-events.js";
import type { AudioSource } from "./audio-source.js";
import type { AudioCaptureHost } from "./host-contract.js";

type ErrorHandler = (error: AudioCaptureError) => void;
type FrameHandler = (frame: unknown) => void;

test("adapts AudioCaptureHost methods to LiveSession audio adapter", async () => {
  const calls: string[] = [];
  const handlers: { error?: ErrorHandler; frame?: FrameHandler } = {};
  const sources: AudioSource[] = [{
    detail: "available",
    id: "system_default",
    isAvailable: true,
    kind: "system",
    name: "Audio",
  }];
  const host = fakeHost({
    listAudioSources: async () => {
      calls.push("list");
      return sources;
    },
    listenAudioCaptureErrors: async (handler) => {
      handlers.error = handler;
      return () => calls.push("unlisten-error");
    },
    listenAudioFrames: async (handler) => {
      handlers.frame = handler;
      return () => calls.push("unlisten-frame");
    },
    startAudioCapture: async (sourceId) => {
      calls.push(`start:${sourceId}`);
    },
    stopAudioCapture: async () => {
      calls.push("stop");
    },
  });
  const adapter = createDesktopAudioAdapter(host);
  const frames: unknown[] = [];
  const errors: AudioCaptureError[] = [];

  assert.equal(await adapter.listSources(), sources);
  const unlistenFrame = await adapter.listenFrames((frame) => frames.push(frame));
  const unlistenError = await adapter.listenCaptureErrors((error) => errors.push(error));
  await adapter.startCapture("system_default");
  handlers.frame?.({
    sampleRate: 16000,
    format: "pcm_s16le",
    dataBase64: "AQID",
  });
  handlers.error?.({ message: "device lost" });
  await adapter.stopCapture();
  unlistenFrame();
  unlistenError();

  assert.deepEqual([...adapter.decodePcm("AQID")], [1, 2, 3]);
  assert.deepEqual(frames, [{ sampleRate: 16000, format: "pcm_s16le", dataBase64: "AQID" }]);
  assert.deepEqual(errors, [{ message: "device lost" }]);
  assert.deepEqual(calls, ["list", "start:system_default", "stop", "unlisten-frame", "unlisten-error"]);
});

function fakeHost(overrides: Partial<AudioCaptureHost>): AudioCaptureHost {
  const noop = async (): Promise<void> => {};
  return {
    listAudioSources: async () => [],
    listenAudioCaptureErrors: async () => () => {},
    listenAudioFrames: async () => () => {},
    startAudioCapture: noop,
    stopAudioCapture: noop,
    ...overrides,
  };
}
