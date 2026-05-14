const DEFAULT_FINISH_TIMEOUT_MS = 120_000;
const DEFAULT_CLOCK: Clock = {
  clearTimeout: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
  setTimeout: (callback, delay) => globalThis.setTimeout(callback, delay),
};

export type SessionState = "idle" | "connecting" | "running" | "finishing";

export interface RealtimeEvent extends Record<string, unknown> {
  type?: string;
}

export interface AudioFrame {
  sampleRate: number;
  format: string;
  dataBase64: string;
}

export type Unlisten = () => void;

export interface LiveSessionClient {
  close(): void;
  connect(startPayload: Record<string, unknown>): Promise<void>;
  finish(): void;
  flush(): void;
  sendPcm(bytes: Uint8Array): boolean;
}

export interface LiveSessionClientCallbacks {
  url: string;
  onClose: (event: CloseEvent, source: LiveSessionClient) => void | Promise<void>;
  onError: (event: Event, source: LiveSessionClient) => void;
  onEvent: (event: RealtimeEvent, source: LiveSessionClient) => void | Promise<void>;
  onStatus: (status: string, source: LiveSessionClient) => void;
}

export interface AudioAdapter {
  decodePcm(base64: string): Uint8Array;
  listenCaptureErrors(handler: (payload: { message?: string } | null | undefined) => void): Promise<Unlisten>;
  listenFrames(handler: (frame: AudioFrame) => void): Promise<Unlisten>;
  startCapture(sourceId: string): Promise<void>;
  stopCapture(): Promise<void>;
}

interface Clock {
  clearTimeout(handle: unknown): void;
  setTimeout(callback: () => void, delay: number): unknown;
}

interface LiveSessionOptions {
  createClient(options: LiveSessionClientCallbacks): LiveSessionClient;
  audio: AudioAdapter;
  onReady?: (event: RealtimeEvent) => void;
  onStateChange?: (state: SessionState, detail: { canStart: boolean }) => void;
  onStatus?: (key: string, value: string) => void;
  onTranscriptEvent?: (event: RealtimeEvent) => void | Promise<void>;
  clock?: Clock;
  finishTimeoutMs?: number;
}

interface StartOptions {
  url: string;
  startPayload: Record<string, unknown>;
  audioSourceId: string;
}

export class LiveSession {
  private readonly audio: AudioAdapter;
  private readonly clock: Clock;
  private readonly createClient: (options: LiveSessionClientCallbacks) => LiveSessionClient;
  private readonly finishTimeoutMs: number;
  private readonly onReady?: (event: RealtimeEvent) => void;
  private readonly onStateChange?: (state: SessionState, detail: { canStart: boolean }) => void;
  private readonly onStatus?: (key: string, value: string) => void;
  private readonly onTranscriptEvent?: (event: RealtimeEvent) => void | Promise<void>;

  private audioAvailable = false;
  private audioSourceId = "";
  private client: LiveSessionClient | null = null;
  private finishTimeout: unknown = null;
  private framesDropped = 0;
  private framesSent = 0;
  private state: SessionState = "idle";
  private unlistenCaptureError: Unlisten | null = null;
  private unlistenFrame: Unlisten | null = null;

  constructor({
    createClient,
    audio,
    onReady,
    onStateChange,
    onStatus,
    onTranscriptEvent,
    clock = DEFAULT_CLOCK,
    finishTimeoutMs = DEFAULT_FINISH_TIMEOUT_MS,
  }: LiveSessionOptions) {
    this.createClient = createClient;
    this.audio = audio;
    this.onReady = onReady;
    this.onStateChange = onStateChange;
    this.onStatus = onStatus;
    this.onTranscriptEvent = onTranscriptEvent;
    this.clock = clock;
    this.finishTimeoutMs = finishTimeoutMs;
  }

  setAudioAvailable(available: boolean): void {
    this.audioAvailable = Boolean(available);
    this.notifyStateChange();
  }

  canStart(): boolean {
    return this.state === "idle" && this.audioAvailable;
  }

  getState(): SessionState {
    return this.state;
  }

  resetStats(): void {
    this.clearFinishTimeout();
    this.framesSent = 0;
    this.framesDropped = 0;
    this.updateAudioStats();
  }

  async start({ url, startPayload, audioSourceId }: StartOptions): Promise<boolean> {
    if (this.state !== "idle") {
      return false;
    }
    if (!this.audioAvailable) {
      this.setStatus("captureStatus", "No native audio source available.");
      return false;
    }

    this.audioSourceId = audioSourceId;
    this.setState("connecting");
    this.setStatus("connectionStatus", "Connecting");

    const client = this.createClient({
      url,
      onEvent: (event, source) => this.handleAsrEvent(event, source),
      onStatus: (status, source) => {
        if (source === this.client) {
          this.setStatus("connectionStatus", status);
        }
      },
      onError: (_event, source) => {
        if (source === this.client) {
          void this.abort("WebSocket connection failed.");
        }
      },
      onClose: (event, source) => this.handleAsrClose(event, source),
    });
    this.client = client;

    try {
      await client.connect(startPayload);
      return true;
    } catch (error) {
      if (client === this.client) {
        await this.abort(errorMessage(error));
      }
      return false;
    }
  }

  flush(): void {
    this.client?.flush();
  }

  async stop({ sendFinish = true }: { sendFinish?: boolean } = {}): Promise<void> {
    if (this.state === "finishing") {
      await this.abort("Final transcript cancelled.");
      return;
    }
    if (sendFinish) {
      await this.finish();
      return;
    }
    await this.abort();
  }

  async finish(): Promise<void> {
    if (this.state === "idle" || this.state === "finishing") {
      return;
    }
    if (this.state === "connecting") {
      await this.abort("Stopped before service was ready.");
      return;
    }

    this.setState("finishing");
    await this.stopCaptureOnly();
    this.setStatus("captureStatus", "Waiting for final transcript");
    this.client?.finish();
    this.scheduleFinishTimeout();
  }

  async abort(message = "", { closeSocket = true }: { closeSocket?: boolean } = {}): Promise<void> {
    if (this.state === "idle" && this.client === null) {
      return;
    }

    const client = this.client;
    this.client = null;
    this.setState("idle");
    this.clearFinishTimeout();
    await this.stopCaptureOnly();
    if (closeSocket) {
      client?.close();
    }
    if (message) {
      this.setStatus("connectionStatus", message);
    }
  }

  async complete(): Promise<void> {
    const client = this.client;
    this.client = null;
    this.setState("idle");
    this.clearFinishTimeout();
    await this.stopCaptureOnly();
    this.setStatus("captureStatus", "Finished");
    client?.close();
  }

  private async handleAsrEvent(event: RealtimeEvent, client: LiveSessionClient): Promise<void> {
    if (client !== this.client) {
      return;
    }

    if (event.type === "ready") {
      this.setState("running");
      this.onReady?.(event);
      try {
        await this.startCaptureAfterReady();
      } catch (error) {
        const message = errorMessage(error);
        this.setStatus("captureStatus", message);
        await this.abort(message);
      }
      return;
    }

    if (event.type === "error") {
      await this.abort(String(event.error || "Service error"));
      return;
    }

    try {
      await this.onTranscriptEvent?.(event);
    } catch (error) {
      const message = errorMessage(error);
      this.setStatus("connectionStatus", message);
      await this.abort(message);
      return;
    }

    if (event.type === "transcript_final") {
      await this.complete();
    }
  }

  private async handleAsrClose(event: CloseEvent, client: LiveSessionClient): Promise<void> {
    if (client !== this.client) {
      return;
    }

    const reason = this.state === "finishing"
      ? `WebSocket closed before transcript_final: ${event.code}`
      : `WebSocket closed: ${event.code}`;
    await this.abort(reason, { closeSocket: false });
  }

  private async startCaptureAfterReady(): Promise<void> {
    if (this.unlistenFrame !== null) {
      return;
    }
    if (!this.audioSourceId) {
      throw new Error("No audio source selected.");
    }

    this.unlistenFrame = await this.audio.listenFrames((frame) => this.handleAudioFrame(frame));
    this.unlistenCaptureError = await this.audio.listenCaptureErrors((payload) => {
      const message = payload?.message || "Audio capture failed.";
      this.setStatus("captureStatus", message);
      void this.abort(message);
    });
    await this.audio.startCapture(this.audioSourceId);
    this.setStatus("captureStatus", "Capturing system audio");
  }

  private async stopCaptureOnly(): Promise<void> {
    try {
      await this.audio.stopCapture();
    } catch (error) {
      this.setStatus("captureStatus", errorMessage(error));
    }
    this.unlistenFrame?.();
    this.unlistenCaptureError?.();
    this.unlistenFrame = null;
    this.unlistenCaptureError = null;
  }

  private handleAudioFrame(frame: AudioFrame): void {
    if (frame?.sampleRate !== 16000 || frame?.format !== "pcm_s16le") {
      this.framesDropped += 1;
      this.updateAudioStats();
      return;
    }

    const sent = this.client?.sendPcm(this.audio.decodePcm(frame.dataBase64));
    if (sent) {
      this.framesSent += 1;
    } else {
      this.framesDropped += 1;
    }
    this.updateAudioStats();
  }

  private scheduleFinishTimeout(): void {
    this.clearFinishTimeout();
    this.finishTimeout = this.clock.setTimeout(
      () => this.abort("Timed out waiting for transcript_final."),
      this.finishTimeoutMs,
    );
  }

  private clearFinishTimeout(): void {
    if (this.finishTimeout !== null) {
      this.clock.clearTimeout(this.finishTimeout);
      this.finishTimeout = null;
    }
  }

  private setState(state: SessionState): void {
    this.state = state;
    this.notifyStateChange();
  }

  private notifyStateChange(): void {
    this.onStateChange?.(this.state, { canStart: this.canStart() });
  }

  private setStatus(key: string, value: string): void {
    this.onStatus?.(key, value);
  }

  private updateAudioStats(): void {
    this.setStatus("audioStats", `frames sent ${this.framesSent}, dropped ${this.framesDropped}`);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
