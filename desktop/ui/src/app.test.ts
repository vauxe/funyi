import test from "node:test";
import assert from "node:assert/strict";
import { AsrClient } from "./asr-client.js";
import { createFunyiApp } from "./app-controller.js";
import { getAppElements } from "./app-dom.js";
import type { AudioSource } from "./audio-source.js";
import { ASR_LANGUAGE_OPTIONS, TRANSLATION_TARGET_LANGUAGE_OPTIONS } from "./languages.js";
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
  assert.equal(elements["audio-source"]!.disabled, true);

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

test("drag uses one start/update/end contract across platforms", async () => {
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
  assert.deepEqual(dragMethods, ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"]);
  assert.equal(elements["app-shell"]!.className, "");
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
  overlay: FakeOverlayHost;
  windowRuntime: FakeWindowRuntime;
}

interface BootAppOptions {
  listSourcesError?: Error | null;
  sources?: AudioSource[];
}

async function bootApp({
  listSourcesError = null,
  sources = defaultAudioSources(),
}: BootAppOptions = {}): Promise<AppHarness> {
  const audio = createFakeAudioAdapter({ listSourcesError, sources });
  const overlay = createFakeOverlayHost();
  const windowRuntime = installFakeWindowRuntime({ timerBehavior: "real" });
  const app = createFunyiApp({
    audio,
    dom: getAppElements(),
    overlay,
    createClient: ({ url, ...callbacks }) => new AsrClient({ url, ...callbacks }),
  });
  await app.boot();
  return { overlay, windowRuntime };
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
