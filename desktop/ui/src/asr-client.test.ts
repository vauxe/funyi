import test from "node:test";
import assert from "node:assert/strict";

import { AsrClient } from "./asr-client.js";
import type { RealtimeStartPayload } from "./realtime-events.js";
import type { LiveSessionClient } from "./session-client.js";
import { nextTick } from "./test-async.fixture.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { FakeWebSocket } from "./test-websocket.fixture.js";

test.beforeEach(() => {
  FakeWebSocket.install();
});

test.afterEach(() => {
  clearBrowserGlobals("WebSocket");
});

test("connect sends start payload after socket opens", async () => {
  const statuses: Array<[string, LiveSessionClient]> = [];
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
  let closedBy: LiveSessionClient | null = null;
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    onClose: (_event, source) => {
      closedBy = source;
    },
  });

  const pending = client.connect({ type: "start", sample_rate: 16000 });
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

  socket.message({ type: "ready" });
  await nextTick();

  assert.equal(statuses.at(-1), "event handler failed: handler boom");
});

test("rejects non-object websocket messages before event handlers", async () => {
  const statuses: string[] = [];
  const events: unknown[] = [];
  const client = new AsrClient({
    url: "ws://127.0.0.1:8000/ws/asr",
    onEvent: (event) => {
      events.push(event);
    },
    onStatus: (status) => statuses.push(status),
  });
  const socket = await connectOpened(client);

  socket.message("[]");

  assert.equal(events.length, 0);
  assert.equal(statuses.at(-1), "invalid event: event payload must be an object");
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

async function connectOpened(
  client: AsrClient,
  payload: RealtimeStartPayload = { type: "start", sample_rate: 16000 },
): Promise<FakeWebSocket> {
  const pending = client.connect(payload);
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.readyState = FakeWebSocket.OPEN;
  socket.open();
  await pending;
  return socket;
}
