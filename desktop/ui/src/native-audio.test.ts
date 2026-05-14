import test from "node:test";
import assert from "node:assert/strict";

import {
  AUDIO_CAPTURE_ERROR_EVENT,
  AUDIO_FRAME_EVENT,
  decodeBase64Pcm,
  isTauriRuntime,
  listAudioSources,
  listenAudioCaptureErrors,
  listenAudioFrames,
  startAudioCapture,
  stopAudioCapture,
  type AudioCaptureError,
  type AudioFrame,
  type AudioSource,
} from "./native-audio.js";

test.afterEach(() => {
  Reflect.deleteProperty(globalThis, "window");
});

test("reports unavailable native audio outside Tauri", async () => {
  Reflect.deleteProperty(globalThis, "window");

  const sources = await listAudioSources();

  assert.equal(isTauriRuntime(), false);
  assert.equal(sources.length, 1);
  assert.equal(sources[0]!.isAvailable, false);
  await assert.rejects(startAudioCapture("system_default"), /requires Tauri/);
  await stopAudioCapture();
});

test("uses Tauri commands and events for native capture", async () => {
  const invokes: Array<{ command: string; args?: Record<string, unknown> }> = [];
  const listeners = new Map<string, (event: { payload: unknown }) => void>();
  let unlistenCount = 0;

  installTauriWindow({
    __TAURI__: {
      core: {
        async invoke<TResult>(command: string, args?: Record<string, unknown>): Promise<TResult> {
          invokes.push({ command, args });
          if (command === "list_audio_sources") {
            return [
              {
                id: "pulse:output.monitor",
                name: "output",
                kind: "system",
                isAvailable: true,
                detail: "monitor",
              },
            ] as unknown as TResult;
          }
          return undefined as TResult;
        },
      },
      event: {
        async listen<TPayload>(
          event: string,
          handler: (event: { payload: TPayload }) => void,
        ): Promise<() => void> {
          listeners.set(event, handler as (event: { payload: unknown }) => void);
          return () => {
            unlistenCount += 1;
            listeners.delete(event);
          };
        },
      },
    },
  });

  const sources = await listAudioSources();
  await startAudioCapture("pulse:output.monitor");
  await stopAudioCapture();

  const frames: AudioFrame[] = [];
  const errors: AudioCaptureError[] = [];
  const unlistenFrame = await listenAudioFrames((frame) => frames.push(frame));
  const unlistenError = await listenAudioCaptureErrors((error) => errors.push(error));

  listeners.get(AUDIO_FRAME_EVENT)?.({
    payload: {
      seq: 1,
      sampleRate: 16000,
      format: "pcm_s16le",
      dataBase64: "AQID",
    },
  });
  listeners.get(AUDIO_CAPTURE_ERROR_EVENT)?.({
    payload: { message: "device lost" },
  });
  unlistenFrame();
  unlistenError();

  assert.equal(isTauriRuntime(), true);
  assert.equal((sources[0] as AudioSource).id, "pulse:output.monitor");
  assert.deepEqual(invokes, [
    { command: "list_audio_sources", args: undefined },
    { command: "start_audio_capture", args: { sourceId: "pulse:output.monitor" } },
    { command: "stop_audio_capture", args: undefined },
  ]);
  assert.deepEqual(frames, [
    { seq: 1, sampleRate: 16000, format: "pcm_s16le", dataBase64: "AQID" },
  ]);
  assert.deepEqual(errors, [{ message: "device lost" }]);
  assert.equal(unlistenCount, 2);
});

test("decodes base64 pcm bytes", () => {
  assert.deepEqual([...decodeBase64Pcm("AQID")], [1, 2, 3]);
});

function installTauriWindow(windowValue: { __TAURI__: NonNullable<Window["__TAURI__"]> }): void {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: windowValue,
    writable: true,
  });
}
