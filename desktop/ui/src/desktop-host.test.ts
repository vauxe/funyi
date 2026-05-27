import test from "node:test";
import assert from "node:assert/strict";

import { DESKTOP_COMMANDS, desktopHost } from "./desktop-host.js";
import type { AudioCaptureError } from "./audio-capture-events.js";
import { AUDIO_CAPTURE_ERROR_EVENT, AUDIO_FRAME_EVENT } from "./audio-capture-events.js";
import type { AudioSource } from "./audio-source.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { installFakeTauriRuntime } from "./test-tauri-runtime.fixture.js";

test.afterEach(() => {
  clearBrowserGlobals("window");
});

test("defines the Tauri command contract in one place", () => {
  assert.deepEqual(DESKTOP_COMMANDS, {
    closeOverlay: "close_overlay",
    endOverlayDrag: "end_overlay_drag",
    endOverlayResize: "end_overlay_resize",
    listAudioSources: "list_audio_sources",
    minimizeOverlay: "minimize_overlay",
    setOverlayMode: "set_overlay_mode",
    startAudioCapture: "start_audio_capture",
    startOverlayDrag: "start_overlay_drag",
    startOverlayResize: "start_overlay_resize",
    stopAudioCapture: "stop_audio_capture",
    updateOverlayDrag: "update_overlay_drag",
    updateOverlayResize: "update_overlay_resize",
  });
});

test("reports unavailable native audio outside Tauri", async () => {
  clearBrowserGlobals("window");

  const sources = await desktopHost.listAudioSources();

  assert.equal(sources.length, 1);
  assert.equal(sources[0]!.isAvailable, false);
  await assert.rejects(desktopHost.startAudioCapture("system_default"), /requires Tauri/);
  await desktopHost.stopAudioCapture();
});

test("uses Tauri commands and events for native capture", async () => {
  const runtime = installFakeTauriRuntime({
    invoke(command) {
      if (command === DESKTOP_COMMANDS.listAudioSources) {
        return [
          {
            id: "system_default",
            name: "output",
            kind: "system",
            isAvailable: true,
            detail: "monitor",
          },
        ];
      }
      return undefined;
    },
  });

  const sources = await desktopHost.listAudioSources();
  await desktopHost.startAudioCapture("system_default");
  await desktopHost.stopAudioCapture();

  const frames: unknown[] = [];
  const errors: AudioCaptureError[] = [];
  const unlistenFrame = await desktopHost.listenAudioFrames((frame) => frames.push(frame));
  const unlistenError = await desktopHost.listenAudioCaptureErrors((error) => errors.push(error));

  runtime.emit(AUDIO_FRAME_EVENT, {
    seq: 1,
    sampleRate: 16000,
    format: "pcm_s16le",
    dataBase64: "AQID",
  });
  runtime.emit(AUDIO_CAPTURE_ERROR_EVENT, { message: "device lost" });
  runtime.emit(AUDIO_CAPTURE_ERROR_EVENT, { message: "" });
  unlistenFrame();
  unlistenError();

  assert.equal((sources[0] as AudioSource).id, "system_default");
  assert.deepEqual(runtime.invocations, [
    { command: DESKTOP_COMMANDS.listAudioSources, args: undefined },
    { command: DESKTOP_COMMANDS.startAudioCapture, args: { sourceId: "system_default" } },
    { command: DESKTOP_COMMANDS.stopAudioCapture, args: undefined },
  ]);
  assert.deepEqual(frames, [
    { seq: 1, sampleRate: 16000, format: "pcm_s16le", dataBase64: "AQID" },
  ]);
  assert.deepEqual(errors, [{ message: "device lost" }, { message: "Audio capture failed." }]);
  assert.equal(runtime.unlistenCount, 2);
});

test("uses Tauri commands for overlay window operations", async () => {
  const runtime = installFakeTauriRuntime();

  await desktopHost.setOverlayMode("history");
  await desktopHost.startOverlayDrag();
  await desktopHost.updateOverlayDrag();
  await desktopHost.endOverlayDrag();
  await desktopHost.startOverlayResize("SouthEast");
  await desktopHost.updateOverlayResize();
  await desktopHost.endOverlayResize();
  await desktopHost.minimizeOverlay();
  await desktopHost.closeOverlay();

  assert.deepEqual(runtime.invocations, [
    { command: DESKTOP_COMMANDS.setOverlayMode, args: { mode: "history" } },
    { command: DESKTOP_COMMANDS.startOverlayDrag, args: undefined },
    { command: DESKTOP_COMMANDS.updateOverlayDrag, args: undefined },
    { command: DESKTOP_COMMANDS.endOverlayDrag, args: undefined },
    { command: DESKTOP_COMMANDS.startOverlayResize, args: { direction: "SouthEast" } },
    { command: DESKTOP_COMMANDS.updateOverlayResize, args: undefined },
    { command: DESKTOP_COMMANDS.endOverlayResize, args: undefined },
    { command: DESKTOP_COMMANDS.minimizeOverlay, args: undefined },
    { command: DESKTOP_COMMANDS.closeOverlay, args: undefined },
  ]);
});

test("keeps independent listeners for the same Tauri event", async () => {
  const runtime = installFakeTauriRuntime();
  const framesA: unknown[] = [];
  const framesB: unknown[] = [];

  const unlistenA = await desktopHost.listenAudioFrames((frame) => framesA.push(frame));
  const unlistenB = await desktopHost.listenAudioFrames((frame) => framesB.push(frame));

  runtime.emit(AUDIO_FRAME_EVENT, { seq: 1 });
  unlistenA();
  runtime.emit(AUDIO_FRAME_EVENT, { seq: 2 });
  unlistenB();
  runtime.emit(AUDIO_FRAME_EVENT, { seq: 3 });

  assert.deepEqual(framesA, [{ seq: 1 }]);
  assert.deepEqual(framesB, [{ seq: 1 }, { seq: 2 }]);
  assert.equal(runtime.unlistenCount, 2);
});

test("rejects invalid Tauri audio source payloads", async () => {
  installFakeTauriRuntime({
    invoke(command) {
      assert.equal(command, DESKTOP_COMMANDS.listAudioSources);
      return { id: "system_default" };
    },
  });

  await assert.rejects(desktopHost.listAudioSources(), /audio sources payload must be an array/);

  installFakeTauriRuntime({
    invoke(command) {
      assert.equal(command, DESKTOP_COMMANDS.listAudioSources);
      return [
        {
          id: "system_default",
          name: "output",
          kind: "speaker",
          isAvailable: true,
          detail: "monitor",
        },
      ];
    },
  });

  await assert.rejects(desktopHost.listAudioSources(), /audio source 0\.kind must be system or microphone/);
});
