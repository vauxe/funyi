import test from "node:test";
import assert from "node:assert/strict";
import { AsrClient } from "./asr-client.js";
import { createFunyiApp, OFFLINE_FILE_SOURCE_ID } from "./app-controller.js";
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
  elements["transport-button"]!.click();

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

test("compact overlay keeps selectable history virtualized", async () => {
  const elements = installDocument();
  const { windowRuntime } = await bootApp();
  elements["transport-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  socket.message(longTranscriptUpdateEvent(120));
  await nextTick();

  windowRuntime.setInnerHeight(320);
  windowRuntime.dispatch("resize", {});
  await nextTick();
  assert.equal(elements["app-shell"]!.attributes.get("data-overlay-mode"), "history");
  assert.ok(historyItemsIn(elements["history-list"]!).length < 40);
  const visibleSource = historyItemsIn(elements["history-list"]!)[0]?.children[1];
  assert.ok(visibleSource);

  elements["history-list"]!.dispatch("pointerdown", { target: visibleSource });
  socket.message(appendTranscriptLineEvent(121));
  await nextTick();
  assert.equal(elements["current-source"]!.textContent, "line 121");
  assert.equal(historyItemsIn(elements["history-list"]!).at(-1)?.children[1]?.textContent, "line 120");
  assert.ok(elements["history-list"]!.className.split(/\s+/).includes("is-virtualized"));
  assert.ok(historyItemsIn(elements["history-list"]!).length < 40);

  windowRuntime.setInnerHeight(240);
  windowRuntime.dispatch("resize", {});
  await nextTick();

  assert.equal(elements["app-shell"]!.attributes.get("data-overlay-mode"), "compact");
  assert.equal(historyItemsIn(elements["history-list"]!).at(-1)?.children[1]?.textContent, "line 120");
  windowRuntime.dispatch("pointerup", {});
  await nextMacrotask();
  assert.equal(historyItemsIn(elements["history-list"]!).at(-1)?.children[1]?.textContent, "line 121");
  assert.ok(elements["history-list"]!.className.split(/\s+/).includes("is-virtualized"));
  assert.ok(historyItemsIn(elements["history-list"]!).length < 40);
});

test("overlay listener setup does not block app boot", async () => {
  const elements = installDocument();
  const overlay = createFakeOverlayHost();
  overlay.listenOverlayDragFinished = () => deferred<() => void>().promise;

  await bootApp({ overlay });

  elements["transport-button"]!.click();
  assert.equal(elements["session-status"]!.textContent, "Connecting...");
  assert.ok(FakeWebSocket.instances[0]);
});

test("empty translation target starts without translation request", async () => {
  const elements = installDocument();

  await bootApp();

  elements["transport-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  const payload = JSON.parse(String(socket.sent[0]));
  assert.equal("target_language" in payload, false);
  assert.equal(elements["session-status"]!.textContent, "Connecting...");
});

test("offline file source posts the selected file and renders the returned transcript", async () => {
  const elements = installDocument();
  const file = namedBlob("clip.wav", "audio/wav");
  const seen: { init: RequestInit | null; url: string } = { init: null, url: "" };
  const restore = stubFetch(async (url, init) => {
    seen.url = String(url);
    seen.init = init ?? null;
    return transcriptStreamResponse({ translation: "hello" });
  });

  try {
    await bootApp();

    elements["language"]!.value = "Chinese";
    elements["translation-target-language"]!.value = "English";
    elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
    elements["audio-source"]!.dispatch("change", {});
    elements["offline-file"]!.files = [file];
    elements["offline-file"]!.dispatch("change", {});

    assert.equal(elements["session-status"]!.textContent, "Start transcription. →");

    elements["transport-button"]!.click();
    await nextTick();

    assert.equal(FakeWebSocket.instances.length, 0);
    assert.equal(
      seen.url,
      "http://127.0.0.1:8000/api/transcriptions/stream?language=Chinese&targetLanguage=English&filename=clip.wav",
    );
    assert.equal(seen.init?.method, "POST");
    assert.equal(seen.init?.body, file);
    assert.equal(elements["current-source"]!.textContent, "你好");
    assert.equal(elements["current-translation"]!.textContent, "hello");
    assert.equal(elements["session-status"]!.textContent, "File transcript ready.");
  } finally {
    restore();
  }
});

test("offline file stream renders segments before the final transcript", async () => {
  const elements = installDocument();
  const file = namedBlob("clip.wav", "audio/wav");
  const stream = controlledNdjsonResponse();
  const restore = stubFetch(async () => stream.response);

  try {
    await bootApp();

    elements["language"]!.value = "Chinese";
    elements["translation-target-language"]!.value = "English";
    elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
    elements["audio-source"]!.dispatch("change", {});
    elements["offline-file"]!.files = [file];
    elements["offline-file"]!.dispatch("change", {});
    elements["transport-button"]!.click();
    await nextTick();

    stream.send(transcriptUpdateEvent());
    await nextTick();

    assert.equal(elements["current-source"]!.textContent, "你好");
    assert.equal(elements["session-status"]!.textContent, "Transcribing file...");

    stream.send(translationStableEvent("hello"));
    await nextTick();

    assert.equal(elements["current-translation"]!.textContent, "hello");

    stream.send(finalTranscriptEvent({ translation: "hello" }));
    await nextTick();

    assert.equal(elements["current-source"]!.textContent, "你好");
    assert.equal(elements["current-translation"]!.textContent, "hello");
    assert.equal(elements["session-status"]!.textContent, "Transcribing file...");

    stream.close();
    await nextTick();
    await nextTick();

    assert.equal(elements["session-status"]!.textContent, "File transcript ready.");
  } finally {
    restore();
  }
});

test("offline file stream keeps translation status after the final transcript", async () => {
  const elements = installDocument();
  const file = namedBlob("clip.wav", "audio/wav");
  const stream = controlledNdjsonResponse();
  const restore = stubFetch(async () => stream.response);

  try {
    await bootApp();

    elements["language"]!.value = "Chinese";
    elements["translation-target-language"]!.value = "English";
    elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
    elements["audio-source"]!.dispatch("change", {});
    elements["offline-file"]!.files = [file];
    elements["offline-file"]!.dispatch("change", {});
    elements["transport-button"]!.click();
    await nextTick();

    stream.send(transcriptUpdateEvent());
    stream.send(translationStatusEvent("timeout", "translation failed"));
    stream.send(
      finalTranscriptEvent({
        translation: null,
        translationMessage: "translation failed",
      }),
    );
    stream.close();
    await nextTick();
    await nextTick();

    assert.equal(elements["current-source"]!.textContent, "你好");
    assert.equal(elements["current-translation"]!.textContent, "translation failed");
    assert.equal(elements["session-status"]!.textContent, "File transcript ready.");
  } finally {
    restore();
  }
});

test("audio source mode changes show the next action in the status bar", async () => {
  const elements = installDocument();

  await bootApp({
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
        name: "Mic",
        kind: "microphone",
        isAvailable: true,
        detail: "available",
      },
    ],
  });

  elements["audio-source"]!.value = "mic_default";
  elements["audio-source"]!.dispatch("change", {});
  assert.equal(elements["session-status"]!.textContent, "Start mic captions. →");

  elements["audio-source"]!.value = "system_default";
  elements["audio-source"]!.dispatch("change", {});
  assert.equal(elements["session-status"]!.textContent, "Start system captions. →");

  elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
  elements["audio-source"]!.dispatch("change", {});
  assert.equal(elements["session-status"]!.textContent, "Choose audio or video file. →");
  assert.equal(elements["offline-file"]!.clicks, 1);

  elements["offline-file"]!.files = [namedBlob("clip.wav", "audio/wav")];
  elements["offline-file"]!.dispatch("change", {});
  assert.equal(elements["session-status"]!.textContent, "Start transcription. →");

  elements["audio-source"]!.value = "system_default";
  elements["audio-source"]!.dispatch("change", {});
  elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
  elements["audio-source"]!.dispatch("change", {});
  assert.equal(elements["session-status"]!.textContent, "Choose audio or video file. →");
  assert.equal(elements["offline-file"]!.clicks, 2);
});

test("offline file source asks for a fresh file after one transcription", async () => {
  const elements = installDocument();
  const file = namedBlob("clip.wav", "audio/wav");
  let requests = 0;
  const restore = stubFetch(async () => {
    requests += 1;
    return transcriptStreamResponse();
  });

  try {
    await bootApp();

    elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
    elements["audio-source"]!.dispatch("change", {});
    elements["offline-file"]!.files = [file];
    elements["offline-file"]!.dispatch("change", {});
    elements["transport-button"]!.click();
    await nextTick();

    elements["transport-button"]!.click();
    await nextTick();

    assert.equal(requests, 1);
    assert.equal(elements["offline-file"]!.clicks, 2);
    assert.equal(elements["session-status"]!.textContent, "Choose audio or video file. →");
  } finally {
    restore();
  }
});

test("offline service errors are shown as errors in the app status", async () => {
  const elements = installDocument();
  const restore = stubFetch(async () =>
    jsonResponse({ error: { code: "busy", message: "Another transcription session is active." } }, false, 409),
  );

  try {
    await bootApp();

    elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
    elements["audio-source"]!.dispatch("change", {});
    elements["offline-file"]!.files = [namedBlob("clip.wav", "audio/wav")];
    elements["offline-file"]!.dispatch("change", {});
    elements["transport-button"]!.click();
    await nextTick();

    assert.equal(elements["session-status"]!.textContent, "Previous session closing");
    assert.equal(elements["session-status"]!.dataset.tone, "error");
  } finally {
    restore();
  }
});

test("audio source listing failures still allow offline file transcription", async () => {
  const elements = installDocument();

  await bootApp({ listSourcesError: new Error("audio source probe failed") });

  assert.equal(elements["transport-button"]!.disabled, false);
  assert.equal(elements["stop-button"]!.disabled, true);
  assert.equal(elements["audio-source"]!.value, OFFLINE_FILE_SOURCE_ID);
  assert.equal(elements["session-status"]!.textContent, "audio source probe failed");

  elements["offline-file"]!.files = [namedBlob("clip.wav", "audio/wav")];
  elements["offline-file"]!.dispatch("change", {});

  assert.equal(elements["session-status"]!.textContent, "Start transcription. →");
});

test("invalid selected audio source is rejected before opening a websocket", async () => {
  const elements = installDocument();

  await bootApp();

  elements["audio-source"]!.value = "missing";
  elements["transport-button"]!.click();

  assert.equal(FakeWebSocket.instances.length, 0);
  assert.equal(elements["session-status"]!.textContent, "Selected audio source is invalid.");
});

test("normal running sessions do not show redundant status text", async () => {
  const elements = installDocument();

  await bootApp();

  elements["translation-target-language"]!.value = "Japanese";
  elements["transport-button"]!.click();
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

  elements["stop-button"]!.click();
  await nextTick();

  assert.equal(elements["transport-button"]!.disabled, true);
  assert.equal(elements["stop-button"]!.title, "Cancel final transcript");
  assert.equal(elements["stop-button"]!.attributes.get("aria-label"), "Cancel final transcript");

  socket.message({ type: "transcript_final", segments: [] });
  await nextTick();
});

test("pause control stops native capture and resumes on the same websocket", async () => {
  const elements = installDocument();
  const { audio } = await bootApp();

  elements["transport-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  await nextTick();

  audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  elements["transport-button"]!.click();
  await nextTick();

  assert.equal(elements["app-shell"]!.attributes.get("data-state"), "paused");
  assert.equal(elements["transport-button"]!.title, "Resume");
  assert.equal(elements["stop-button"]!.title, "Stop");
  assert.equal(elements["session-status"]!.textContent, "Paused");
  assert.equal(audio.stopCalls, 1);
  assert.equal(audio.frameHandler, null);
  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(socket.closeCalls, 0);
  assert.equal(socket.sent.length, 2);

  elements["transport-button"]!.click();
  await nextTick();

  assert.equal(elements["app-shell"]!.attributes.get("data-state"), "running");
  assert.equal(elements["transport-button"]!.title, "Pause");
  assert.deepEqual(audio.startCalls, ["system_default", "system_default"]);

  emitAudioFrame(audio, "efgh");

  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(socket.closeCalls, 0);
  assert.equal(socket.sent.length, 3);
  assert.deepEqual(socket.sent.slice(1), [new Uint8Array([4]), new Uint8Array([4])]);
});

test("active server session errors are shown as a retryable user status", async () => {
  const elements = installDocument();

  await bootApp();

  elements["transport-button"]!.click();
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
  elements["transport-button"]!.click();
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

  elements["transport-button"]!.click();
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

test("file source cannot replace live capture while captions are running", async () => {
  const elements = installDocument();
  const { audio } = await bootApp();

  elements["transport-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  await nextTick();

  elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
  elements["audio-source"]!.dispatch("change", {});
  await nextTick();

  assert.equal(elements["audio-source"]!.value, "system_default");
  assert.deepEqual(audio.startCalls, ["system_default"]);
  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(elements["session-status"]!.textContent, "File transcription unavailable while captions are running.");
});

test("invalid audio source changes are rejected while running", async () => {
  const elements = installDocument();

  await bootApp();

  elements["transport-button"]!.click();
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

  elements["transport-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  socket.message({ type: "ready", sample_rate: 16000 });
  await nextTick();

  socket.message({ type: "error", error: "Unsupported target_language: Swedish." });
  await nextTick();

  assert.equal(elements["session-status"]!.textContent, "Unsupported target_language: Swedish.");
  assert.equal(elements["stop-button"]!.title, "Stop");
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
  elements["audio-source"]!.value = OFFLINE_FILE_SOURCE_ID;
  elements["audio-source"]!.dispatch("change", {});

  assert.deepEqual(preferences.load(), {
    serverUrl: "ws://127.0.0.1:9100/ws/asr",
    asrLanguage: "Chinese",
    targetLanguage: "Japanese",
    audioSourceId: "system_default",
    captionOpacity: null,
  });
  assert.equal(elements["offline-file"]!.clicks, 1);
});

test("ignores stored options that no longer exist", async () => {
  const elements = installDocument();
  const preferences = new PreferencesStore(new MemoryKeyValueStore());
  preferences.save({ asrLanguage: "Klingon", audioSourceId: "ghost-device" });

  await bootApp({ preferences });

  assert.equal(elements["language"]!.value, "");
  assert.equal(elements["audio-source"]!.value, "system_default");
});

test("ignores stored file source when restoring the audio source", async () => {
  const elements = installDocument();
  const preferences = new PreferencesStore(new MemoryKeyValueStore());
  preferences.save({ audioSourceId: OFFLINE_FILE_SOURCE_ID });

  await bootApp({ preferences });

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

function historyItemsIn(historyList: FakeElement): FakeElement[] {
  return historyList.children.filter((child) => child.className.split(/\s+/).includes("history-item"));
}

function nextMacrotask(): Promise<void> {
  return new Promise((resolve) => globalThis.setTimeout(resolve, 0));
}

function emitAudioFrame(audio: ReturnType<typeof createFakeAudioAdapter>, dataBase64: string): void {
  const frameHandler = audio.frameHandler;
  assert.ok(frameHandler);
  frameHandler({ sampleRate: 16000, format: "pcm_s16le", dataBase64 });
}

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

function transcriptStreamResponse({ translation = null }: { translation?: string | null } = {}): Response {
  return ndjsonResponse([
    transcriptUpdateEvent(),
    ...(translation ? [translationStableEvent(translation)] : []),
    finalTranscriptEvent({ translation }),
  ]);
}

function transcriptUpdateEvent(): Record<string, unknown> {
  return {
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
  };
}

function longTranscriptUpdateEvent(lineCount: number): Record<string, unknown> {
  return {
    type: "transcript_update",
    revision: 1,
    stable_base: 0,
    stable_count: lineCount,
    stable_appends: Array.from({ length: lineCount }, (_item, index) => ({
      id: `seg_${String(index + 1).padStart(6, "0")}`,
      index: index + 1,
      start_ms: index * 1000,
      end_ms: index * 1000 + 500,
      text: `line ${index + 1}`,
      language: "Chinese",
      timing_status: "aligned",
    })),
    partial: null,
  };
}

function appendTranscriptLineEvent(index: number): Record<string, unknown> {
  return {
    type: "transcript_update",
    revision: 2,
    stable_base: index - 1,
    stable_count: index,
    stable_appends: [
      {
        id: `seg_${String(index).padStart(6, "0")}`,
        index,
        start_ms: (index - 1) * 1000,
        end_ms: (index - 1) * 1000 + 500,
        text: `line ${index}`,
        language: "Chinese",
        timing_status: "aligned",
      },
    ],
    partial: null,
  };
}

function translationStableEvent(text: string): Record<string, unknown> {
  return {
    type: "translation_stable",
    source_revision: 1,
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    source_segment_ids: ["seg_000001"],
    source_segment_indices: [1],
    text,
    target_language: "English",
  };
}

function translationStatusEvent(code: string, message: string): Record<string, unknown> {
  return {
    type: "translation_status",
    scope: "stable",
    code,
    message,
    source_revision: 1,
    source_segment_id: "seg_000001",
    source_segment_index: 1,
    source_segment_ids: ["seg_000001"],
    source_segment_indices: [1],
    target_language: "English",
  };
}

function finalTranscriptEvent({
  translation = null,
  translationMessage = null,
}: {
  translation?: string | null;
  translationMessage?: string | null;
} = {}): Record<string, unknown> {
  return {
    type: "transcript_final",
    revision: 1,
    final_revision: 1,
    stable_count: 1,
    segments: [
      {
        id: "seg_000001",
        index: 1,
        start_ms: 0,
        end_ms: 1200,
        text: "你好",
        language: "Chinese",
      },
    ],
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
          ...(translation ? { translation } : {}),
          ...(translationMessage ? { translationMessage } : {}),
        },
      ],
    },
  };
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

function controlledNdjsonResponse(): {
  close(): void;
  response: Response;
  send(payload: unknown): void;
} {
  const encoder = new TextEncoder();
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      controller = streamController;
    },
  });
  return {
    close() {
      controller?.close();
    },
    response: {
      body,
      ok: true,
      status: 200,
    } as Response,
    send(payload: unknown) {
      controller?.enqueue(encoder.encode(`${JSON.stringify(payload)}\n`));
    },
  };
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
