import test from "node:test";
import assert from "node:assert/strict";

class FakeElement {
  checked = false;
  children: FakeElement[] = [];
  className = "";
  disabled = false;
  listeners = new Map<string, Array<() => void>>();
  textContent = "";
  title = "";
  value = "";

  constructor(readonly tagName: string, readonly id = "") {}

  addEventListener(type: string, listener: () => void): void {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  append(...children: FakeElement[]): void {
    this.children.push(...children);
    if (this.tagName === "select" && !this.value && children[0]?.value) {
      this.value = children[0].value;
    }
  }

  click(): void {
    for (const listener of this.listeners.get("click") || []) {
      listener();
    }
  }

  replaceChildren(): void {
    this.children = [];
    if (this.tagName === "select") {
      this.value = "";
    }
  }
}

class FakeDocument {
  readonly elements: Record<string, FakeElement>;

  constructor(elements: Record<string, FakeElement>) {
    this.elements = elements;
  }

  createElement(tagName: string): FakeElement {
    return new FakeElement(tagName);
  }

  querySelector(selector: string): FakeElement | null {
    return selector.startsWith("#") ? this.elements[selector.slice(1)] || null : null;
  }
}

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];

  binaryType = "";
  bufferedAmount = 0;
  onclose?: (event: CloseEvent) => void;
  onerror?: (event: Event) => void;
  onmessage?: (event: MessageEvent) => void;
  onopen?: () => void;
  readyState = FakeWebSocket.CONNECTING;
  sent: Array<string | Uint8Array> = [];

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(payload: string | Uint8Array): void {
    this.sent.push(payload);
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  close(): void {
    this.onclose?.({ code: 1000 } as CloseEvent);
  }
}

test.beforeEach(() => {
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
});

test.afterEach(() => {
  Reflect.deleteProperty(globalThis, "document");
  Reflect.deleteProperty(globalThis, "window");
});

test("start button sends the selected UI options as the ASR start payload", async () => {
  const elements = installDocument();
  installTauriRuntime();

  await import("./app.js");
  await nextTick();

  elements["context"]!.value = "product names";
  elements["translation-enabled"]!.checked = false;
  elements["start-button"]!.click();

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
  assert.equal(payload.context, "product names");
  assert.equal(payload.translation, false);
});

function installDocument(): Record<string, FakeElement> {
  const elements = Object.fromEntries(
    [
      "server-url",
      "language",
      "context",
      "audio-source",
      "translation-enabled",
      "start-button",
      "stop-button",
      "flush-button",
      "export-srt-button",
      "connection-status",
      "ready-status",
      "capture-status",
      "audio-stats",
      "previous-source",
      "previous-translation",
      "current-source",
      "current-translation",
      "history-list",
    ].map((id) => [id, new FakeElement(elementTag(id), id)]),
  );

  elements["server-url"]!.value = "ws://127.0.0.1:8000/ws/asr";
  elements["language"]!.value = "Chinese";
  elements["translation-enabled"]!.checked = true;

  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: new FakeDocument(elements),
    writable: true,
  });
  return elements;
}

function installTauriRuntime(): void {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          async invoke<TResult>(command: string): Promise<TResult> {
            if (command === "list_audio_sources") {
              return [
                {
                  id: "system_default",
                  name: "System audio",
                  kind: "system",
                  isAvailable: true,
                  detail: "available",
                },
              ] as unknown as TResult;
            }
            return undefined as TResult;
          },
        },
        event: {
          async listen(): Promise<() => void> {
            return () => {};
          },
        },
      },
    },
    writable: true,
  });
}

function elementTag(id: string): string {
  if (id === "audio-source") {
    return "select";
  }
  if (id.endsWith("button")) {
    return "button";
  }
  return "input";
}

function nextTick(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}
