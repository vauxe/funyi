export type AsrEvent = Record<string, unknown> & {
  type?: string;
};

export type AsrStartPayload = Record<string, unknown>;

type AsrEventCallback = (event: AsrEvent, source: AsrClient) => void | Promise<void>;
type AsrSocketCallback<TEvent> = (event: TEvent, source: AsrClient) => void | Promise<void>;
type StatusCallback = (status: string, source: AsrClient) => void;

export interface AsrClientOptions {
  url: string;
  onEvent?: AsrEventCallback;
  onStatus?: StatusCallback;
  onError?: AsrSocketCallback<Event>;
  onClose?: AsrSocketCallback<CloseEvent>;
  maxBufferedBytes?: number;
  closeTimeoutMs?: number;
}

export class AsrClient {
  private readonly closeTimeoutMs: number;
  private readonly maxBufferedBytes: number;
  private readonly onClose?: AsrSocketCallback<CloseEvent>;
  private readonly onError?: AsrSocketCallback<Event>;
  private readonly onEvent?: AsrEventCallback;
  private readonly onStatus?: StatusCallback;
  private readonly url: string;
  private closeWait: Promise<void> | null = null;
  private finishCloseWait: (() => void) | null = null;
  private ws: WebSocket | null = null;

  constructor({
    url,
    onEvent,
    onStatus,
    onError,
    onClose,
    maxBufferedBytes = 512 * 1024,
    closeTimeoutMs = 1000,
  }: AsrClientOptions) {
    this.url = url;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.onError = onError;
    this.onClose = onClose;
    this.maxBufferedBytes = maxBufferedBytes;
    this.closeTimeoutMs = closeTimeoutMs;
  }

  connect(startPayload: AsrStartPayload): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      ws.onopen = () => {
        this.emitStatus("WS OK");
        ws.send(JSON.stringify(startPayload));
        settled = true;
        resolve();
      };
      ws.onerror = (event) => {
        this.emitStatus("WS error");
        this.runCallback("error handler", () => this.onError?.(event, this));
        if (!settled) {
          settled = true;
          reject(new Error("WebSocket connection failed"));
        }
      };
      ws.onclose = (event) => {
        this.emitStatus("WS closed");
        if (this.ws === ws) {
          this.ws = null;
        }
        this.finishCloseWait?.();
        if (!settled) {
          settled = true;
          reject(new Error(`WebSocket closed before start: ${event.code}`));
        }
        this.runCallback("close handler", () => this.onClose?.(event, this));
      };
      ws.onmessage = (message) => {
        if (typeof message.data !== "string") {
          return;
        }
        try {
          const event = JSON.parse(message.data);
          this.runCallback("event handler", () => this.onEvent?.(event, this));
        } catch (error) {
          this.emitStatus(`invalid event: ${errorMessage(error)}`);
        }
      };
    });
  }

  sendPcm(bytes: Uint8Array): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    if (this.ws.bufferedAmount > this.maxBufferedBytes) {
      return false;
    }
    this.ws.send(bytes);
    return true;
  }

  finish(): void {
    this.sendCommand("finish");
  }

  close(): Promise<void> {
    const ws = this.ws;
    if (!ws || ws.readyState === WebSocket.CLOSED) {
      this.ws = null;
      return Promise.resolve();
    }
    if (this.closeWait) {
      return this.closeWait;
    }
    const closeWait = new Promise<void>((resolve) => {
      let settled = false;
      const timeout = globalThis.setTimeout(finish, this.closeTimeoutMs);
      this.finishCloseWait = finish;

      if (ws.readyState <= WebSocket.OPEN) {
        try {
          ws.close();
        } catch {
          finish();
        }
      }

      function finish(): void {
        if (settled) {
          return;
        }
        settled = true;
        globalThis.clearTimeout(timeout);
        resolve();
      }
    }).finally(() => {
      if (this.ws === ws) {
        this.ws = null;
      }
      this.closeWait = null;
      this.finishCloseWait = null;
    });
    this.closeWait = closeWait;
    return closeWait;
  }

  private sendCommand(type: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(JSON.stringify({ type }));
  }

  private emitStatus(status: string): void {
    this.onStatus?.(status, this);
  }

  private runCallback(label: string, callback: () => void | Promise<void>): void {
    try {
      void Promise.resolve(callback()).catch((error: unknown) => {
        this.emitStatus(`${label} failed: ${errorMessage(error)}`);
      });
    } catch (error) {
      this.emitStatus(`${label} failed: ${errorMessage(error)}`);
    }
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
