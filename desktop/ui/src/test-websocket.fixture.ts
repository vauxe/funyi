type CloseBehavior = "manual" | "emit";

export class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  private static closeBehavior: CloseBehavior = "manual";

  binaryType = "";
  bufferedAmount = 0;
  closeCalls = 0;
  onclose?: (event: CloseEvent) => void;
  onerror?: (event: Event) => void;
  onmessage?: (event: MessageEvent) => void;
  onopen?: () => void;
  readyState = FakeWebSocket.CONNECTING;
  sent: Array<string | Uint8Array> = [];

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  static install({ closeBehavior = "manual" }: { closeBehavior?: CloseBehavior } = {}): void {
    FakeWebSocket.instances = [];
    FakeWebSocket.closeBehavior = closeBehavior;
    globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  }

  send(payload: string | Uint8Array): void {
    this.sent.push(payload);
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  message(event: Record<string, unknown> | string): void {
    const data = typeof event === "string" ? event : JSON.stringify(event);
    this.onmessage?.({ data } as MessageEvent);
  }

  close(): void {
    this.closeCalls += 1;
    this.readyState = FakeWebSocket.CLOSING;
    if (FakeWebSocket.closeBehavior === "emit") {
      this.emitClose();
    }
  }

  emitClose(code = 1000): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code } as CloseEvent);
  }
}
