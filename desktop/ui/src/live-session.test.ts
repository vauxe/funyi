import test from "node:test";
import assert from "node:assert/strict";

import {
  LiveSession,
  type LiveSessionClient,
  type LiveSessionClientCallbacks,
  type RealtimeEvent,
  type AudioAdapter,
  type AudioFrame,
  type Unlisten,
} from "./live-session.js";

class FakeAsrClient implements LiveSessionClient {
  closed = false;
  closeWait: Promise<void> | null = null;
  commands: string[] = [];
  onClose: LiveSessionClientCallbacks["onClose"];
  onError: LiveSessionClientCallbacks["onError"];
  onEvent: LiveSessionClientCallbacks["onEvent"];
  onStatus: LiveSessionClientCallbacks["onStatus"];
  sendPcmResult = true;
  sentPcm: Uint8Array[] = [];
  startPayload: Record<string, unknown> | null = null;
  url: string;

  constructor({ url, onClose, onError, onEvent, onStatus }: LiveSessionClientCallbacks) {
    this.onClose = onClose;
    this.onError = onError;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.url = url;
  }

  async connect(startPayload: Record<string, unknown>): Promise<void> {
    this.startPayload = startPayload;
    this.onStatus?.("WS OK", this);
  }

  close(): Promise<void> | void {
    this.closed = true;
    return this.closeWait || undefined;
  }

  finish(): void {
    this.commands.push("finish");
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
  const statuses = new Map<string, string>();
  const audio = {
    captureErrorHandler: null as ((payload: { message?: string } | null | undefined) => void) | null,
    frameHandler: null as ((frame: AudioFrame) => void) | null,
    startCalls: [] as string[],
    stopCalls: 0,
    unlistenCaptureErrors: 0,
    unlistenFrames: 0,
    decodePcm: (base64: string) => new Uint8Array([base64.length]),
    listenCaptureErrors: async (handler: (payload: { message?: string } | null | undefined) => void): Promise<Unlisten> => {
      audio.captureErrorHandler = handler;
      return () => {
        audio.captureErrorHandler = null;
        audio.unlistenCaptureErrors += 1;
      };
    },
    listenFrames: async (handler: (frame: AudioFrame) => void): Promise<Unlisten> => {
      audio.frameHandler = handler;
      return () => {
        audio.frameHandler = null;
        audio.unlistenFrames += 1;
      };
    },
    startCapture: async (sourceId: string) => {
      audio.startCalls.push(sourceId);
    },
    stopCapture: async () => {
      audio.stopCalls += 1;
    },
  } satisfies AudioAdapter & {
    captureErrorHandler: ((payload: { message?: string } | null | undefined) => void) | null;
    frameHandler: ((frame: AudioFrame) => void) | null;
    startCalls: string[];
    stopCalls: number;
    unlistenCaptureErrors: number;
    unlistenFrames: number;
  };
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
): Promise<void> {
  harness.session.setAudioAvailable(true);
  await harness.session.start({
    audioSourceId,
    startPayload: { type: "start", sample_rate: 16000 },
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  await harness.clients[0]!.emit({ type: "ready", sample_rate: 16000 });
}

function nextTick(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
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
  assert.equal(harness.statuses.get("audioStats"), "Silent");
});

test("warns when macOS capture keeps delivering silent pcm", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  for (let index = 0; index < 30; index += 1) {
    harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  }

  assert.match(
    harness.statuses.get("captureStatus") || "",
    /Sys silent/,
  );
});

test("reports dropped frames when websocket backpressure refuses pcm", async () => {
  const harness = createHarness();
  await startRunningSession(harness);
  harness.clients[0]!.sendPcmResult = false;

  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });

  assert.equal(harness.clients[0]!.sentPcm.length, 0);
  assert.equal(harness.statuses.get("audioStats"), "Silent, dropped 2");
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
  await startRunningSession(harness, "macos_microphone:built-in");

  assert.equal(harness.statuses.get("captureStatus"), "Mic");
  for (let index = 0; index < 30; index += 1) {
    harness.audio.frameHandler?.({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "abcd" });
  }

  assert.match(
    harness.statuses.get("captureStatus") || "",
    /Mic silent/,
  );
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

test("capture errors abort the active session and clean listeners", async () => {
  const harness = createHarness();
  await startRunningSession(harness);

  harness.audio.captureErrorHandler?.({ message: "device lost" });
  await nextTick();

  assert.equal(harness.session.getState(), "idle");
  assert.equal(harness.clients[0]!.closed, true);
  assert.equal(harness.audio.frameHandler, null);
  assert.equal(harness.audio.captureErrorHandler, null);
  assert.equal(harness.statuses.get("connectionStatus"), "device lost");
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
