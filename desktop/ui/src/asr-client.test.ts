import test from "node:test";
import assert from "node:assert/strict";

import { AsrClient } from "./asr-client.js";

class FakeWebSocket {
  static CONNECTING = 0;
  static CLOSING = 2;
  static OPEN = 1;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  binaryType = "";
  bufferedAmount = 0;
  closeCalls = 0;
  onclose?: (event: CloseEvent) => void;
  onerror?: (event: Event) => void;
  onmessage?: (event: MessageEvent) => void;
  onopen?: () => void;
  readyState = FakeWebSocket.CONNECTING;
  sent: Array<string | Uint8Array> = [];
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(payload: string | Uint8Array): void {
    this.sent.push(payload);
  }

  close(): void {
    this.closeCalls += 1;
    this.readyState = FakeWebSocket.CLOSING;
  }

  emitClose(code = 1000): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code } as CloseEvent);
  }
}

test.beforeEach(() => {
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
});

test("connect sends start payload after socket opens", async () => {
  const statuses: Array<[string, AsrClient]> = [];
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    onStatus: (status, source) => statuses.push([status, source]),
  });

  const socket = await connectOpened(client, { type: "start", sample_rate: 16000 });

  assert.equal(socket.binaryType, "arraybuffer");
  assert.deepEqual(JSON.parse(String(socket.sent[0])), { type: "start", sample_rate: 16000 });
  assert.deepEqual(statuses, [["WS OK", client]]);
});

test("connect rejects if socket closes before open", async () => {
  let closedBy: AsrClient | null = null;
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    onClose: (_event, source) => {
      closedBy = source;
    },
  });

  const pending = client.connect({ type: "start" });
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emitClose(1006);

  await assert.rejects(pending, /WebSocket closed before start: 1006/);
  assert.equal(closedBy, client);
});

test("sendPcm drops frames when websocket buffer is over limit", async () => {
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    maxBufferedBytes: 4,
  });
  const socket = await connectOpened(client);

  socket.sent = [];
  socket.bufferedAmount = 5;
  assert.equal(client.sendPcm(new Uint8Array([1, 2])), false);
  assert.equal(socket.sent.length, 0);

  socket.bufferedAmount = 4;
  assert.equal(client.sendPcm(new Uint8Array([1, 2])), true);
  assert.equal(socket.sent.length, 1);
});

test("reports async message handler failures", async () => {
  const statuses: string[] = [];
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    onEvent: async () => {
      throw new Error("handler boom");
    },
    onStatus: (status) => statuses.push(status),
  });
  const socket = await connectOpened(client);

  socket.onmessage?.({ data: JSON.stringify({ type: "ready" }) } as MessageEvent);
  await nextTick();

  assert.equal(statuses.at(-1), "event handler failed: handler boom");
});

test("reports async close handler failures", async () => {
  const statuses: string[] = [];
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    onClose: async () => {
      throw new Error("close boom");
    },
    onStatus: (status) => statuses.push(status),
  });
  const socket = await connectOpened(client);

  socket.emitClose(1000);
  await nextTick();

  assert.equal(statuses.at(-1), "close handler failed: close boom");
});

test("close resolves only after websocket close is observed", async () => {
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  const socket = await connectOpened(client);

  let closed = false;
  const pendingClose = client.close().then(() => {
    closed = true;
  });
  await nextTick();

  assert.equal(socket.closeCalls, 1);
  assert.equal(closed, false);

  socket.emitClose();
  await pendingClose;

  assert.equal(closed, true);
});

test("setLanguageConfig sends runtime language command", async () => {
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
  });
  const socket = await connectOpened(client);

  client.setLanguageConfig({ language: "English", target_language: null });

  assert.deepEqual(JSON.parse(String(socket.sent.at(-1))), {
    type: "set_language",
    language: "English",
    target_language: null,
  });
});

function nextTick(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}

async function connectOpened(
  client: AsrClient,
  payload: Record<string, unknown> = { type: "start" },
): Promise<FakeWebSocket> {
  const pending = client.connect(payload);
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.readyState = FakeWebSocket.OPEN;
  socket.onopen?.();
  await pending;
  return socket;
}
