import test from "node:test";
import assert from "node:assert/strict";
import { ASR_LANGUAGE_OPTIONS, TRANSLATION_TARGET_LANGUAGE_OPTIONS } from "./languages.js";

type Listener = (event?: unknown) => void;
type Invocation = { command: string; args?: Record<string, unknown> };

class FakeElement {
  attributes = new Map<string, string>();
  checked = false;
  children: FakeElement[] = [];
  classList = {
    add: (name: string): void => {
      const classes = new Set(this.className.split(/\s+/).filter(Boolean));
      classes.add(name);
      this.className = [...classes].join(" ");
    },
    remove: (name: string): void => {
      const classes = new Set(this.className.split(/\s+/).filter(Boolean));
      classes.delete(name);
      this.className = [...classes].join(" ");
    },
    toggle: (name: string, enabled: boolean): void => {
      const classes = new Set(this.className.split(/\s+/).filter(Boolean));
      if (enabled) {
        classes.add(name);
      } else {
        classes.delete(name);
      }
      this.className = [...classes].join(" ");
    },
  };
  className = "";
  dataset: Record<string, string> = {};
  disabled = false;
  listeners = new Map<string, Listener[]>();
  textContent = "";
  title = "";
  value = "";

  constructor(readonly tagName: string, readonly id = "") {}

  addEventListener(type: string, listener: Listener): void {
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

  dispatch(type: string, event: unknown): void {
    for (const listener of this.listeners.get(type) || []) {
      listener(event);
    }
  }

  replaceChildren(...children: FakeElement[]): void {
    this.children = [];
    if (this.tagName === "select") {
      this.value = "";
    }
    this.append(...children);
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
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

  message(event: Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent);
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
  Reflect.deleteProperty(globalThis, "Element");
  Reflect.deleteProperty(globalThis, "window");
});

test("history button switches overlay mode and inline settings drive start payload", async () => {
  const elements = installDocument();
  const invocations = installTauriRuntime();

  await importApp("history");
  await nextTick();

  elements["history-button"]!.click();
  await nextTick();

  assert.equal(elements["app-shell"]!.attributes.get("data-overlay-mode"), "history");
  assert.equal(elements["history-button"]!.className, "is-expanded");
  assert.equal(elements["history-button"]!.attributes.get("aria-expanded"), "true");
  assert.deepEqual(invocations.at(-1), {
    command: "set_overlay_mode",
    args: { mode: "history" },
  });
  assert.equal(elements["connection-status"]!.textContent, "");
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
  installTauriRuntime();

  await importApp("default-translation-target");
  await nextTick();

  elements["session-button"]!.click();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);

  socket.open();
  const payload = JSON.parse(String(socket.sent[0]));
  assert.equal("target_language" in payload, false);
});

test("ready status includes the negotiated translation target", async () => {
  const elements = installDocument();
  installTauriRuntime();

  await importApp("ready-translation-target");
  await nextTick();

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

  assert.equal(elements["ready-status"]!.textContent, "16k · Translate Japanese");
});

test("macOS native drag does not run manual drag release commands", async () => {
  const elements = installDocument();
  const invocations = installTauriRuntime({ platform: "macos" });

  await importApp("macos-native-drag");
  await nextTick();

  elements["caption-strip"]!.dispatch("pointerdown", {
    button: 0,
    pointerId: 7,
    preventDefault: () => {},
    target: null,
  });
  await nextTick();

  assert.equal(elements["app-shell"]!.className, "is-dragging");
  dispatchWindow("pointerup", { pointerId: 7 });
  await nextTick();

  const dragCommands = invocations
    .map((invocation) => invocation.command)
    .filter((command) => command.includes("overlay_drag"));
  assert.deepEqual(dragCommands, ["start_overlay_drag", "finish_native_overlay_drag"]);
  assert.equal(elements["app-shell"]!.className, "");
});

function installDocument(): Record<string, FakeElement> {
  const elements = Object.fromEntries(
    [
      "server-url",
      "app-shell",
      "caption-strip",
      "language",
      "translation-target-language",
      "audio-source",
      "session-button",
      "history-button",
      "minimize-button",
      "close-button",
      "resize-north",
      "resize-east",
      "resize-south",
      "resize-west",
      "resize-north-east",
      "resize-north-west",
      "resize-south-east",
      "resize-south-west",
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
  elements["language"]!.value = "";
  elements["translation-target-language"]!.value = "English";

  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: new FakeDocument(elements),
    writable: true,
  });
  Object.defineProperty(globalThis, "Element", {
    configurable: true,
    value: FakeElement,
    writable: true,
  });
  return elements;
}

function installTauriRuntime(
  options: { platform?: string } = {},
): Invocation[] {
  const invocations: Invocation[] = [];
  const windowListeners = new Map<string, Listener[]>();
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      addEventListener(type: string, listener: Listener): void {
        const listeners = windowListeners.get(type) || [];
        listeners.push(listener);
        windowListeners.set(type, listeners);
      },
      clearTimeout,
      dispatch(type: string, event: unknown): void {
        for (const listener of windowListeners.get(type) || []) {
          listener(event);
        }
      },
      removeEventListener(type: string, listener: Listener): void {
        const listeners = windowListeners.get(type) || [];
        windowListeners.set(
          type,
          listeners.filter((item) => item !== listener),
        );
      },
      setTimeout,
      __TAURI__: {
        core: {
          async invoke<TResult>(command: string, args?: Record<string, unknown>): Promise<TResult> {
            invocations.push({ command, args });
            if (command === "desktop_platform") {
              return options.platform as TResult;
            }
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
  return invocations;
}

function elementTag(id: string): string {
  if (id === "audio-source" || id === "language" || id === "translation-target-language") {
    return "select";
  }
  if (id === "app-shell" || id === "caption-strip") {
    return "main";
  }
  if (id.endsWith("button")) {
    return "button";
  }
  return "input";
}

function selectValues(element: FakeElement): string[] {
  return element.children.map((child) => child.value);
}

function nextTick(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}

async function importApp(testName: string): Promise<void> {
  await import(`./app.js?${testName}-${Date.now()}`);
}

function dispatchWindow(type: string, event: unknown): void {
  (globalThis.window as unknown as { dispatch: (type: string, event: unknown) => void })
    .dispatch(type, event);
}
