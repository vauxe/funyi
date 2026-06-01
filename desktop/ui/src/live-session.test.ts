import test from "node:test";
import assert from "node:assert/strict";

import type { AudioSourceKind } from "./audio-source-kind.js";
import { LiveSession } from "./live-session.js";
import type { LanguageConfigUpdate, RealtimeEvent, RealtimeStartPayload } from "./realtime-events.js";
import type { LiveSessionClient, LiveSessionClientCallbacks } from "./session-client.js";
import { createFakeAudioAdapter } from "./test-audio-adapter.fixture.js";
import { nextTick } from "./test-async.fixture.js";

class FakeAsrClient implements LiveSessionClient {
  closed = false;
  closeWait: Promise<void> | null = null;
  commands: string[] = [];
  finishResult = true;
  languageConfigs: LanguageConfigUpdate[] = [];
  onClose: LiveSessionClientCallbacks["onClose"];
  onError: LiveSessionClientCallbacks["onError"];
  onEvent: LiveSessionClientCallbacks["onEvent"];
  onStatus: LiveSessionClientCallbacks["onStatus"];
  sendPcmResult = true;
  sentPcm: Uint8Array[] = [];
  startPayload: RealtimeStartPayload | null = null;
  url: string;

  constructor({ url, onClose, onError, onEvent, onStatus }: LiveSessionClientCallbacks) {
    this.onClose = onClose;
    this.onError = onError;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.url = url;
  }

  async connect(startPayload: RealtimeStartPayload): Promise<void> {
    this.startPayload = startPayload;
    this.onStatus?.("WS OK", this);
  }

  close(): Promise<void> | void {
    this.closed = true;
    return this.closeWait || undefined;
  }

  finish(): boolean {
    this.commands.push("finish");
    return this.finishResult;
  }

  setLanguageConfig(config: LanguageConfigUpdate): void {
    this.languageConfigs.push(config);
  }

  sendPcm(bytes: Uint8Array): boolean {
    if (this.sendPcmResult) {
      this.sentPcm.push(bytes);
    }
    return this.sendPcmResult;
  }

  emit(event: RealtimeEvent): void | Promise<void> {
    return this.onEvent(event, this);
  }
}

interface HarnessOptions {
  onReady?: (event: RealtimeEvent) => void;
  onTranscriptEvent?: (event: RealtimeEvent) => void | Promise<void>;
}

function createHarness({ onReady, onTranscriptEvent }: HarnessOptions = {}) {
  const clients: FakeAsrClient[] = [];
  const statuses = new Map<string, unknown>();
  const audio = createFakeAudioAdapter();
  const clock = {
    scheduled: null as { callback: () => void | Promise<void>; delay: number; id: symbol } | null,
    clearTimeout(id: unknown): void {
      if (this.scheduled?.id === id) {
        this.scheduled = null;
      }
    },
    setTimeout(callback: () => void | Promise<void>, delay: number): symbol {
      const id = Symbol("timeout");
      this.scheduled = { callback, delay, id };
      return id;
    },
  };
  const session = new LiveSession({
    audio,
    clock,
    createClient: (options) => {
      const client = new FakeAsrClient(options);
      clients.push(client);
      return client;
    },
    finishTimeoutMs: 50,
    onReady,
    onStatus: (key, value) => statuses.set(key, value),
    onTranscriptEvent,
  });
  return { audio, clients, clock, session, statuses };
}

async function startRunningSession(
  harness: ReturnType<typeof createHarness>,
  audioSourceId = "system_default",
  audioSourceKind: AudioSourceKind = "system",
): Promise<void> {
  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId,
    audioSourceKind,
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  await harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });
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

function assertNoRuntimeCommands(client: FakeAsrClient): void {
  assert.deepEqual(client.commands, []);
  assert.deepEqual(client.languageConfigs, []);
}

test("starts capture after ready and forwards only valid pcm frames", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  assert.equal(harness.session.getState(), "running");
  assert.deepEqual(harness.audio.startCalls, ["system_default"]);
  assert.deepEqual(harness.clients[0]!.startPayload, { type: "start", sample_rate: 16000 });
  assert.equal(harness.statuses.get("captureStatus"), "Sys");

  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  harness.audio.frameHandler?.({ sampleRate: 48000, format: "pcm_s16le", dataBase64: "abcd" });

  assert.deepEqual([...harness.clients[0]!.sentPcm[0]!], [4]);
  assert.deepEqual(harness.statuses.get("audioStats"), { levelDb: null, droppedFrames: 0 });
});

test("forwards language config only while running", async () => {
  const harness = createHarness();

  harness.session.setLanguageConfig({ language: "English" });
  await startRunningSession(harness);

  harness.session.setLanguageConfig({ target_language: "Japanese" });
  assert.deepEqual(harness.clients[0]!.languageConfigs, [{ target_language: "Japanese" }]);

  await harness.session.stop({ sendFinish: false });
  harness.session.setLanguageConfig({ language: null });
  assert.deepEqual(harness.clients[0]!.languageConfigs, [{ target_language: "Japanese" }]);
});

test("keeps running session open after non-fatal service command error", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.clients[0]!.emit({ type: "error", error: "Unsupported target_language: Swedish." });

  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients[0]!.closed, false);
  assert.equal(harness.statuses.get("connectionStatus"), "Unsupported target_language: Swedish.");
});

test("fatal service errors abort the active session", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.clients[0]!.emit({ type: "error", error: "Realtime ASR session failed.", fatal: true });

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.statuses.get("connectionStatus"), "Realtime ASR session failed.");
});

test("warns when system capture keeps delivering silent pcm", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  for (let index = 0; index < 30; index += 1) {
    harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  }

  assert.match(String(harness.statuses.get("captureStatus") ?? ""), /Sys silent/);
  assert.equal(harness.statuses.get("audioHealth"), "systemSilent");
});

test("reports dropped frames when websocket backpressure refuses pcm", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  harness.clients[0]!.sendPcmResult = false;

  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });

  assert.equal(harness.clients[0]!.sentPcm.length, 0);
  assert.deepEqual(harness.statuses.get("audioStats"), { levelDb: null, droppedFrames: 2 });
});

test("switches audio source inside the running websocket session", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });

  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients.length, 1);
  assert.equal(harness.clients[0]!.closed, false);
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["system_default", "mic_default"]);
  assert.equal(harness.audio.stopCalls, 1);
  assert.equal(harness.statuses.get("captureStatus"), "Mic");

  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  assert.deepEqual([...harness.clients[0]!.sentPcm.at(-1)!], [4]);
});

test("switching while initial capture is starting replaces the native source", async () => {
  const harness = createHarness();
  const systemStart = deferred<void>();
  harness.audio.startCapture = async (sourceId: string) => {
    harness.audio.startCalls.push(sourceId);
    if (sourceId === "system_default") {
      await systemStart.promise;
    }
  };

  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  const ready = harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });
  await nextTick();

  const switching = harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  await nextTick();

  systemStart.resolve();
  await Promise.all([ready, switching]);

  assert.equal(harness.session.getState(), "running");
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["system_default", "mic_default"]);
  assert.equal(harness.audio.stopCalls, 1);
  assert.equal(harness.statuses.get("captureStatus"), "Mic");
});

test("stale initial source startup failure after switching does not close the ASR session", async () => {
  const harness = createHarness();
  const systemStart = deferred<void>();
  harness.audio.startCapture = async (sourceId: string) => {
    harness.audio.startCalls.push(sourceId);
    if (sourceId === "system_default") {
      await systemStart.promise;
      throw new Error("system source unavailable");
    }
  };

  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  const ready = harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });
  await nextTick();

  const switching = harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  await nextTick();

  systemStart.resolve();
  await Promise.all([ready, switching]);

  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients[0]!.closed, false);
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["system_default", "mic_default"]);
  assert.equal(harness.audio.stopCalls, 0);
  assert.equal(harness.statuses.get("captureStatus"), "Mic");
});

test("switching before initial capture listeners are ready still replaces the native source", async () => {
  const harness = createHarness();
  const frameListenerReady = deferred<void>();
  harness.audio.listenFrames = async (handler) => {
    await frameListenerReady.promise;
    harness.audio.frameHandler = handler;
    return () => {
      harness.audio.frameHandler = null;
      harness.audio.unlistenFrames += 1;
    };
  };

  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  const ready = harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });
  await nextTick();

  const switching = harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  await nextTick();

  frameListenerReady.resolve();
  await Promise.all([ready, switching]);

  assert.equal(harness.session.getState(), "running");
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["mic_default"]);
  assert.equal(harness.audio.stopCalls, 0);
  assert.equal(harness.statuses.get("captureStatus"), "Mic");
});

test("runtime audio source start failure does not close the ASR session", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  harness.audio.startCapture = async (sourceId: string) => {
    harness.audio.startCalls.push(sourceId);
    if (sourceId === "mic_default") {
      throw new Error("microphone unavailable");
    }
  };

  await harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });

  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients.length, 1);
  assert.equal(harness.clients[0]!.closed, false);
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["system_default", "mic_default"]);
  assert.equal(harness.audio.stopCalls, 1);
  assert.equal(harness.statuses.get("captureStatus"), "microphone unavailable");
});

test("runtime capture errors after an audio source switch do not close the ASR session", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  harness.audio.captureErrorHandler?.({ message: "device lost" });
  await nextTick();

  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients[0]!.closed, false);
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.equal(harness.statuses.get("captureStatus"), "device lost");
});

test("ignores a runtime audio source switch to the already active source", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.session.switchAudioSource({
    audioSourceId: "system_default",
    audioSourceKind: "system",
  });

  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["system_default"]);
  assert.equal(harness.audio.stopCalls, 0);
});

test("applies the last audio source selected during an in-flight switch", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  const firstStop = deferred<void>();
  let holdFirstStop = true;
  harness.audio.stopCapture = async () => {
    harness.audio.stopCalls += 1;
    if (holdFirstStop) {
      holdFirstStop = false;
      await firstStop.promise;
    }
  };

  const switchToMic = harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  await nextTick();
  const switchToSystem = harness.session.switchAudioSource({
    audioSourceId: "system_default",
    audioSourceKind: "system",
  });

  firstStop.resolve();
  await Promise.all([switchToMic, switchToSystem]);

  assertNoRuntimeCommands(harness.clients[0]!);
  assert.deepEqual(harness.audio.startCalls, ["system_default", "system_default"]);
  assert.equal(harness.statuses.get("captureStatus"), "Sys");
});

test("stop during an in-flight source switch prevents stale source start", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  const firstStop = deferred<void>();
  let holdFirstStop = true;
  harness.audio.stopCapture = async () => {
    harness.audio.stopCalls += 1;
    if (holdFirstStop) {
      holdFirstStop = false;
      await firstStop.promise;
    }
  };

  const switching = harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  await nextTick();
  const stopping = harness.session.stop({ sendFinish: false });
  await nextTick();

  firstStop.resolve();
  await Promise.all([switching, stopping]);

  assert.equal(harness.session.getState(), "idle");
  assert.deepEqual(harness.audio.startCalls, ["system_default"]);
  assert.equal(harness.audio.stopCalls, 1);
  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
});

test("stop during new source startup leaves capture stopped", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  const micStart = deferred<void>();
  harness.audio.startCapture = async (sourceId: string) => {
    harness.audio.startCalls.push(sourceId);
    if (sourceId === "mic_default") {
      await micStart.promise;
    }
  };

  const switching = harness.session.switchAudioSource({
    audioSourceId: "mic_default",
    audioSourceKind: "microphone",
  });
  await nextTick();
  assert.deepEqual(harness.audio.startCalls, ["system_default", "mic_default"]);

  const stopping = harness.session.stop({ sendFinish: false });
  await nextTick();
  micStart.resolve();
  await Promise.all([switching, stopping]);

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.audio.stopCalls, 2);
  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
});

test("ready callback errors abort before capture starts", async () => {
  const harness = createHarness({
    onReady: () => {
      throw new Error("ready render failed");
    },
  });
  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });

  await harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.deepEqual(harness.audio.startCalls, []);
  assert.equal(harness.statuses.get("connectionStatus"), "ready render failed");
});

test("uses microphone-specific capture status and silent warning", async () => {
  const harness = createHarness();
  await startRunningSession(harness, "opaque-mic-device", "microphone");

  assert.equal(harness.statuses.get("captureStatus"), "Mic");
  for (let index = 0; index < 30; index += 1) {
    harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  }

  assert.match(String(harness.statuses.get("captureStatus") ?? ""), /Mic silent/);
});

test("finish sends final command, times out cleanly, and restores idle state", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.session.stop();

  assert.equal(harness.session.getState(), "finishing");
  assert.deepEqual(harness.clients[0]!.commands, ["finish"]);
  assert.equal(harness.clock.scheduled?.delay, 50);
  assert.equal(harness.audio.stopCalls, 1);

  await harness.clock.scheduled?.callback();

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.statuses.get("connectionStatus"), "Timed out waiting for transcript_final.");
});

test("stop while finishing cancels the final wait", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.session.stop();
  await harness.session.stop();

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.clock.scheduled, null);
  assert.equal(harness.statuses.get("connectionStatus"), "Final transcript cancelled.");
});

test("transcript final completes the active session through the event path", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.clients[0]!.emit({ type: "transcript_final", segments: [] });

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
  assert.equal(harness.statuses.get("captureStatus"), "Done");
});

test("immediate stop waits for socket close before returning to idle", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  let releaseClose: () => void = () => {};
  harness.clients[0]!.closeWait = new Promise((resolve) => {
    releaseClose = resolve;
  });

  let stopped = false;
  const pendingStop = harness.session.stop({ sendFinish: false }).then(() => {
    stopped = true;
  });
  await nextTick();

  assert.equal(stopped, false);
  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients[0]!.closed, true);

  releaseClose();
  await pendingStop;

  assert.equal(stopped, true);
  assert.equal(harness.session.getState(), "idle");
});

test("capture errors report status without closing the ASR session", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  harness.audio.captureErrorHandler?.({ message: "device lost" });
  await nextTick();

  assert.equal(harness.session.getState(), "running");
  assert.equal(harness.clients[0]!.closed, false);
  assertNoRuntimeCommands(harness.clients[0]!);
  assert.equal(harness.statuses.get("captureStatus"), "device lost");
});

test("aborts and reports the reason when the socket closes mid-session", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  await harness.clients[0]!.onClose({ code: 1006 } as CloseEvent, harness.clients[0]!);

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
  assert.equal(harness.statuses.get("connectionStatus"), "WebSocket closed: 1006");
});

test("replay errors abort final handling without marking the session finished", async () => {
  const harness = createHarness({
    onTranscriptEvent: () => {
      throw new Error("stable cursor mismatch");
    },
  });
  await startRunningSession(harness);

  await harness.clients[0]!.emit({ type: "transcript_final", segments: [] });

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.statuses.get("connectionStatus"), "stable cursor mismatch");
  assert.notEqual(harness.statuses.get("captureStatus"), "Done");
});

test("queues language config while connecting and flushes it on ready", async () => {
  const harness = createHarness();
  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });

  assert.equal(harness.session.getState(), "connecting");
  harness.session.setLanguageConfig({ target_language: "Japanese" });
  assert.deepEqual(harness.clients[0]!.languageConfigs, []);

  await harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });

  assert.deepEqual(harness.clients[0]!.languageConfigs, [{ target_language: "Japanese" }]);
});

test("aborts immediately when the finish frame cannot be sent", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  harness.clients[0]!.finishResult = false;

  await harness.session.stop();

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.clock.scheduled, null);
  assert.deepEqual(harness.clients[0]!.commands, ["finish"]);
  assert.equal(harness.statuses.get("connectionStatus"), "Stopped before the final transcript could be requested.");
});

test("coalesces concurrent teardown so native capture stops once", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  let releaseClose: () => void = () => {};
  harness.clients[0]!.closeWait = new Promise((resolve) => {
    releaseClose = resolve;
  });

  const first = harness.session.stop({ sendFinish: false });
  const second = harness.session.stop({ sendFinish: false });
  await nextTick();

  assert.equal(harness.session.getState(), "running");

  releaseClose();
  await Promise.all([first, second]);

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.audio.stopCalls, 1);
  assert.equal(harness.clients[0]!.closed, true);
});

test("a stop during transcript_final completion coalesces into one teardown", async () => {
  let releaseTranscript: () => void = () => {};
  const harness = createHarness({
    onTranscriptEvent: () =>
      new Promise<void>((resolve) => {
        releaseTranscript = resolve;
      }),
  });
  await startRunningSession(harness);
  let releaseClose: () => void = () => {};
  harness.clients[0]!.closeWait = new Promise((resolve) => {
    releaseClose = resolve;
  });

  const completed = harness.clients[0]!.emit({ type: "transcript_final", segments: [] });
  releaseTranscript();
  await nextTick();
  // complete() is now awaiting the socket close; a stop must coalesce, not stop capture twice.
  const aborted = harness.session.stop({ sendFinish: false });
  await nextTick();

  assert.equal(harness.session.getState(), "running");
  releaseClose();
  await Promise.all([completed, aborted]);

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.audio.stopCalls, 1);
  assert.equal(harness.statuses.get("captureStatus"), "Done");
});

test("a teardown during the transcript handler skips completion", async () => {
  let releaseTranscript: () => void = () => {};
  const harness = createHarness({
    onTranscriptEvent: () =>
      new Promise<void>((resolve) => {
        releaseTranscript = resolve;
      }),
  });
  await startRunningSession(harness);

  const completed = harness.clients[0]!.emit({ type: "transcript_final", segments: [] });
  await nextTick();
  await harness.session.stop({ sendFinish: false });
  releaseTranscript();
  await completed;

  assert.equal(harness.session.getState(), "idle");
  assert.notEqual(harness.statuses.get("captureStatus"), "Done");
});

test("a language config queued for an aborted connect does not leak into the next session", async () => {
  const harness = createHarness();
  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  harness.session.setLanguageConfig({ target_language: "Japanese" });
  await harness.session.stop({ sendFinish: false });

  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  await harness.clients[1]!.emit({ type: "ready", sample_rate: 16000 });

  assert.deepEqual(harness.clients[1]!.languageConfigs, []);
});

test("merges multiple language configs queued while connecting", async () => {
  const harness = createHarness();
  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId: "system_default",
    audioSourceKind: "system",
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });

  harness.session.setLanguageConfig({ language: "English" });
  harness.session.setLanguageConfig({ target_language: "Japanese" });
  harness.session.setLanguageConfig({ language: "Chinese" });
  await harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });

  assert.deepEqual(harness.clients[0]!.languageConfigs, [{ language: "Chinese", target_language: "Japanese" }]);
});
