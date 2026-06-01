import test from "node:test";
import assert from "node:assert/strict";
import { AsrClient } from "./asr-client.js";
import { createFunyiApp } from "./app-controller.js";
import { getAppElements } from "./app-dom.js";
import type { AudioSource } from "./audio-source.js";
import { ASR_LANGUAGE_OPTIONS, TRANSLATION_TARGET_LANGUAGE_OPTIONS } from "./languages.js";
import { MemoryKeyValueStore, PreferencesStore } from "./preferences.js";
import { installFakeAppDocument } from "./test-app-document.fixture.js";
import { nextTick } from "./test-async.fixture.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { type FakeElement, installedFakeDocument } from "./test-dom.fixture.js";
import { createFakeAudioAdapter } from "./test-audio-adapter.fixture.js";
import { createFakeOverlayHost, type FakeOverlayHost } from "./test-overlay-host.fixture.js";
import { pointerEvent } from "./test-pointer-event.fixture.js";
import { FakeWebSocket } from "./test-websocket.fixture.js";
import { installFakeWindowRuntime, type FakeWindowRuntime } from "./test-window.fixture.js";

test.beforeEach(() => {
  FakeWebSocket.install({ closeBehavior: "emit" });
});

test.afterEach(() => {
  clearBrowserGlobals("document", "Element", "HTMLElement", "window", "WebSocket");
});

test("window height switches history mode and inline settings drive start payload", async () => {
  const elements = installDocument();
  const { overlay, windowRuntime } = await bootApp();

  windowRuntime.setInnerHeight(320);
  windowRuntime.dispatch("resize", {});
  await nextTick();

  assert.equal(elements["app-shell"]!.attributes.get("data-overlay-mode"), "history");
  assert.equal("history-button" in elements, false);
  assert.deepEqual(overlay.invocations, []);
  assert.equal(elements["session-status"]!.textContent, "");
  assert.equal(elements["audio-source"]!.children[0]?.textContent, "Sys · Audio");
  assert.deepEqual(selectValues(elements["language"]!), ["", ...ASR_LANGUAGE_OPTIONS]);
  assert.deepEqual(selectValues(elements["translation-target-language"]!), [
    "",
    ...TRANSLATION_TARGET_LANGUAGE_OPTIONS,
  ]);
  assert.ok(selectValues(elements["translation-target-language"]!).includes("Traditional Chinese"));
  assert.equal(selectValues(elements["translation-target-language"]!).includes("Swedish"), false);

  elements["language"]!.value = "Chinese";
  elements["translation-target-language"]!.value = "Japanese";
  elements["session-button"]!.click();

  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  assert.equal(socket.url, "ws://127.0.0.1:8000/ws/asr");

  socket.open();

  const payload = JSON.parse(String(socket.sent[0]));
  assert.equal(payload.type, "start");
  assert.match(payload.session_id, /^desktop-\d+$/);
  assert.equal(payload.sample_rate, 16000);
  assert.equal(payload.audio_format, "pcm_s16le");
  assert.equal(payload.language, "Chinese");
  assert.equal("context" in payload, false);
  assert.equal(payload.target_language, "Japanese");
});

test("overlay listener setup does not block app boot", async () => {
  const elements = installDocument();
  const overlay = createFakeOverlayHost();
  overlay.listenOverlayDragFinished = () => deferred<() => void>().promise;

  await bootApp({ overlay });

  elements["session-button"]!.click();
  assert.equal(elements["session-status"]!.textContent, "Connecting...");
  assert.ok(FakeWebSocket.instances[0]);
});

test("empty translation target starts without translation request", async () => {
  const elements = installDocument();

  await bootApp();

  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  const payload = JSON.parse(String(socket.sent[0]));
  assert.equal("target_language" in payload, false);
  assert.equal(elements["session-status"]!.textContent, "Connecting...");
});

test("audio source listing failures keep the UI in a disabled idle state", async () => {
  const elements = installDocument();

  await bootApp({ listSourcesError: new Error("audio source probe failed") });

  assert.equal(elements["session-button"]!.disabled, true);
  assert.equal(elements["session-status"]!.textContent, "audio source probe failed");
});

test("invalid selected audio source is rejected before opening a websocket", async () => {
  const elements = installDocument();

  await bootApp();

  elements["audio-source"]!.value = "missing";
  elements["session-button"]!.click();

  assert.equal(FakeWebSocket.instances.length, 0);
  assert.equal(elements["session-status"]!.textContent, "Selected audio source is invalid.");
});

test("normal running sessions do not show redundant status text", async () => {
  const elements = installDocument();

  await bootApp();

  elements["translation-target-language"]!.value = "Japanese";
  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  socket.message({
    type: "ready",
    sample_rate: 16000,
    translation: { enabled: true, target_language: "Japanese" },
  });
  await nextTick();

  assert.equal(elements["session-status"]!.textContent, "");
  assert.equal(elements["app-shell"]!.dataset.statusActive, "false");

  elements["session-button"]!.click();
  await nextTick();

  assert.equal(elements["session-button"]!.title, "Cancel final transcript");
  assert.equal(elements["session-button"]!.attributes.get("aria-label"), "Cancel final transcript");

  socket.message({ type: "transcript_final", segments: [] });
  await nextTick();
});

test("active server session errors are shown as a retryable user status", async () => {
  const elements = installDocument();

  await bootApp();

  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  socket.message({ type: "error", error: "Another realtime session is active." });
  await nextTick();

  assert.equal(elements["session-status"]!.textContent, "Previous session closing");
});

test("language controls stay editable while running and send runtime updates", async () => {
  const elements = installDocument();

  await bootApp();

  elements["language"]!.value = "Chinese";
  elements["translation-target-language"]!.value = "English";
  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  socket.message({
    type: "ready",
    sample_rate: 16000,
    translation: { enabled: true, target_language: "English" },
  });
  await nextTick();

  assert.equal(elements["language"]!.disabled, false);
  assert.equal(elements["translation-target-language"]!.disabled, false);
  assert.equal(elements["server-url"]!.disabled, true);
  assert.equal(elements["audio-source"]!.disabled, false);

  elements["language"]!.value = "Japanese";
  elements["language"]!.dispatch("change", {});
  assert.deepEqual(JSON.parse(String(socket.sent.at(-1))), {
    type: "set_language",
    language: "Japanese",
  });

  elements["translation-target-language"]!.value = "";
  elements["translation-target-language"]!.dispatch("change", {});
  assert.deepEqual(JSON.parse(String(socket.sent.at(-1))), {
    type: "set_language",
    target_language: null,
  });
});

test("audio source changes hot-switch capture without reopening the websocket", async () => {
  const elements = installDocument();
  const { audio } = await bootApp({
    sources: [
      {
        id: "system_default",
        name: "Audio",
        kind: "system",
        isAvailable: true,
        detail: "available",
      },
      {
        id: "mic_default",
        name: "Studio Mic",
        kind: "microphone",
        isAvailable: true,
        detail: "available",
      },
    ],
  });

  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  await nextTick();

  elements["audio-source"]!.value = "mic_default";
  elements["audio-source"]!.dispatch("change", {});
  await nextTick();

  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(socket.sent.length, 1);
  assert.deepEqual(audio.startCalls, ["system_default", "mic_default"]);
  assert.equal(audio.stopCalls, 1);
  assert.equal(elements["session-status"]!.textContent, "");
});

test("invalid audio source changes are rejected while running", async () => {
  const elements = installDocument();

  await bootApp();

  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  await nextTick();

  elements["audio-source"]!.value = "missing";
  elements["audio-source"]!.dispatch("change", {});
  await nextTick();

  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(socket.sent.length, 1);
  assert.equal(elements["session-status"]!.textContent, "Selected audio source is invalid.");
});

test("recoverable command errors remain visible while running", async () => {
  const elements = installDocument();

  await bootApp();

  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  await nextTick();

  socket.message({ type: "error", error: "Unsupported target_language: Swedish." });
  await nextTick();

  assert.equal(elements["session-status"]!.textContent, "Unsupported target_language: Swedish.");
  assert.equal(elements["session-button"]!.title, "Stop");
});

test("native drag keeps the shell active until native finished event", async () => {
  const elements = installDocument();
  const { overlay, windowRuntime } = await bootApp();

  installedFakeDocument().activeElement = elements["server-url"]!;
  elements["caption-strip"]!.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();

  assert.equal(elements["server-url"]!.blurred, true);
  assert.equal(elements["app-shell"]!.className, "is-dragging");
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  await nextTick();

  const dragMethods = overlay.invocations
    .map((invocation) => invocation.method)
    .filter((method) => method.endsWith("OverlayDrag"));
  assert.deepEqual(dragMethods, ["startOverlayDrag"]);
  assert.equal(elements["app-shell"]!.className, "is-dragging");

  overlay.emitOverlayDragFinished(0);
  assert.equal(elements["app-shell"]!.className, "");
});

test("fallback drag uses start update end contract when native drag id is unavailable", async () => {
  const elements = installDocument();
  const { overlay, windowRuntime } = await bootApp();
  overlay.startOverlayDrag = async () => {
    overlay.invocations.push({ method: "startOverlayDrag" });
    return null;
  };

  elements["caption-strip"]!.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  await nextTick();

  const dragMethods = overlay.invocations
    .map((invocation) => invocation.method)
    .filter((method) => method.endsWith("OverlayDrag"));
  assert.deepEqual(dragMethods, ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"]);
  assert.equal(elements["app-shell"]!.className, "");
});

test("restores stored preferences into the controls on boot", async () => {
  const elements = installDocument();
  const preferences = new PreferencesStore(new MemoryKeyValueStore());
  preferences.save({ serverUrl: "ws://127.0.0.1:9001/ws/asr", asrLanguage: "Chinese", targetLanguage: "Japanese" });

  await bootApp({ preferences });

  assert.equal(elements["server-url"]!.value, "ws://127.0.0.1:9001/ws/asr");
  assert.equal(elements["language"]!.value, "Chinese");
  assert.equal(elements["translation-target-language"]!.value, "Japanese");
});

test("persists control changes for the next launch", async () => {
  const elements = installDocument();
  const { preferences } = await bootApp();

  elements["server-url"]!.value = "ws://127.0.0.1:9100/ws/asr";
  elements["server-url"]!.dispatch("change", {});
  elements["language"]!.value = "Chinese";
  elements["language"]!.dispatch("change", {});
  elements["translation-target-language"]!.value = "Japanese";
  elements["translation-target-language"]!.dispatch("change", {});
  elements["audio-source"]!.value = "system_default";
  elements["audio-source"]!.dispatch("change", {});

  assert.deepEqual(preferences.load(), {
    serverUrl: "ws://127.0.0.1:9100/ws/asr",
    asrLanguage: "Chinese",
    targetLanguage: "Japanese",
    audioSourceId: "system_default",
    captionOpacity: null,
  });
});

test("ignores stored options that no longer exist", async () => {
  const elements = installDocument();
  const preferences = new PreferencesStore(new MemoryKeyValueStore());
  preferences.save({ asrLanguage: "Klingon", audioSourceId: "ghost-device" });

  await bootApp({ preferences });

  assert.equal(elements["language"]!.value, "");
  assert.equal(elements["audio-source"]!.value, "system_default");
});

function installDocument(): Record<string, FakeElement> {
  const elements = installFakeAppDocument();
  elements["server-url"]!.value = "ws://127.0.0.1:8000/ws/asr";
  elements["language"]!.value = "";
  elements["translation-target-language"]!.value = "English";
  return elements;
}

function selectValues(element: FakeElement): string[] {
  return element.children.map((child) => child.value);
}

interface AppHarness {
  audio: ReturnType<typeof createFakeAudioAdapter>;
  overlay: FakeOverlayHost;
  windowRuntime: FakeWindowRuntime;
  preferences: PreferencesStore;
}

interface BootAppOptions {
  listSourcesError?: Error | null;
  overlay?: FakeOverlayHost;
  sources?: AudioSource[];
  preferences?: PreferencesStore;
}

async function bootApp({
  listSourcesError = null,
  overlay = createFakeOverlayHost(),
  sources = defaultAudioSources(),
  preferences = new PreferencesStore(new MemoryKeyValueStore()),
}: BootAppOptions = {}): Promise<AppHarness> {
  const audio = createFakeAudioAdapter({ listSourcesError, sources });
  const windowRuntime = installFakeWindowRuntime({ timerBehavior: "real" });
  const app = createFunyiApp({
    audio,
    dom: getAppElements(),
    overlay,
    preferences,
    createClient: ({ url, ...callbacks }) => new AsrClient({ url, ...callbacks }),
  });
  await app.boot();
  await nextTick();
  return { audio, overlay, windowRuntime, preferences };
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

function defaultAudioSources(): AudioSource[] {
  return [
    {
      id: "system_default",
      name: "Audio",
      kind: "system",
      isAvailable: true,
      detail: "available",
    },
  ];
}
