export type AsrEvent = Record<string, unknown> & {
  type?: string;
};

export type AsrStartPayload = Record<string, unknown>;

type AsrEventCallback = (event: AsrEvent, source: AsrClient) => void | Promise<void>;
type AsrSocketCallback<TEvent> = (event: TEvent, source: AsrClient) => void;
type StatusCallback = (status: string, source: AsrClient) => void;

export interface AsrClientOptions {
  url: string;
  onEvent?: AsrEventCallback;
  onStatus?: StatusCallback;
  onError?: AsrSocketCallback<Event>;
  onClose?: AsrSocketCallback<CloseEvent>;
  maxBufferedBytes?: number;
}

export class AsrClient {
  private readonly maxBufferedBytes: number;
  private readonly onClose?: AsrSocketCallback<CloseEvent>;
  private readonly onError?: AsrSocketCallback<Event>;
  private readonly onEvent?: AsrEventCallback;
  private readonly onStatus?: StatusCallback;
  private readonly url: string;
  private ws: WebSocket | null = null;

  constructor({ url, onEvent, onStatus, onError, onClose, maxBufferedBytes = 512 * 1024 }: AsrClientOptions) {
    this.url = url;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.onError = onError;
    this.onClose = onClose;
    this.maxBufferedBytes = maxBufferedBytes;
  }

  connect(startPayload: AsrStartPayload): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      ws.onopen = () => {
        this.emitStatus("connected");
        ws.send(JSON.stringify(startPayload));
        settled = true;
        resolve();
      };
      ws.onerror = (event) => {
        this.emitStatus("websocket error");
        this.onError?.(event, this);
        if (!settled) {
          settled = true;
          reject(new Error("WebSocket connection failed"));
        }
      };
      ws.onclose = (event) => {
        this.emitStatus("closed");
        if (!settled) {
          settled = true;
          reject(new Error(`WebSocket closed before start: ${event.code}`));
        }
        this.onClose?.(event, this);
      };
      ws.onmessage = (message) => {
        if (typeof message.data !== "string") {
          return;
        }
        try {
          const event = JSON.parse(message.data);
          this.onEvent?.(event, this);
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

  flush(): void {
    this.sendCommand("flush");
  }

  finish(): void {
    this.sendCommand("finish");
  }

  close(): void {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      this.ws.close();
    }
    this.ws = null;
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
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
