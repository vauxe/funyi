import type { AudioAdapter } from "./audio-adapter.js";
import type { AudioSourceKind } from "./audio-source-kind.js";
import { errorMessage } from "./error-message.js";
import { DEFAULT_CLOCK, DEFAULT_FINISH_TIMEOUT_MS, FinishTimeout, type Clock } from "./finish-timeout.js";
import { LiveAudioCapture } from "./live-audio-capture.js";
import type { LanguageConfigUpdate, RealtimeEvent } from "./realtime-events.js";
import type { LiveSessionClient, LiveSessionClientCallbacks } from "./session-client.js";
import type { SessionState } from "./session-state.js";
import type { LiveSessionStartOptions } from "./session-start-options.js";
import {
  FINAL_TRANSCRIPT_CANCELLED_MESSAGE,
  NO_AUDIO_SOURCE_MESSAGE,
  type StatusKey,
  type StatusValue,
} from "./session-status.js";

interface LiveSessionOptions {
  createClient(options: LiveSessionClientCallbacks): LiveSessionClient;
  audio: AudioAdapter;
  onReady?: (event: RealtimeEvent) => void;
  onStateChange?: (state: SessionState, detail: { canStart: boolean }) => void;
  onStatus?: <K extends StatusKey>(key: K, value: StatusValue<K>) => void;
  onTranscriptEvent?: (event: RealtimeEvent) => void | Promise<void>;
  clock?: Clock;
  finishTimeoutMs?: number;
}

interface SelectedAudioSource {
  id: string;
  kind: AudioSourceKind;
}

export class LiveSession {
  private readonly audioCapture: LiveAudioCapture;
  private readonly createClient: (options: LiveSessionClientCallbacks) => LiveSessionClient;
  private readonly finishTimeout: FinishTimeout;
  private readonly onReady?: (event: RealtimeEvent) => void;
  private readonly onStateChange?: (state: SessionState, detail: { canStart: boolean }) => void;
  private readonly onStatus?: <K extends StatusKey>(key: K, value: StatusValue<K>) => void;
  private readonly onTranscriptEvent?: (event: RealtimeEvent) => void | Promise<void>;

  private audioAvailable = false;
  private client: LiveSessionClient | null = null;
  private selectedAudioSource: SelectedAudioSource | null = null;
  private state: SessionState = "idle";

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
    this.onReady = onReady;
    this.onStateChange = onStateChange;
    this.onStatus = onStatus;
    this.onTranscriptEvent = onTranscriptEvent;
    this.finishTimeout = new FinishTimeout(clock, finishTimeoutMs);
    this.audioCapture = new LiveAudioCapture({
      audio,
      onAbort: (message) => void this.abort(message),
      onStatus: (key, value) => this.setStatus(key, value),
    });
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

  setLanguageConfig(config: LanguageConfigUpdate): void {
    if (this.state !== "running" || this.client === null) {
      return;
    }
    this.client.setLanguageConfig(config);
  }

  resetStats(): void {
    this.finishTimeout.clear();
    this.audioCapture.resetStats();
  }

  async start({ url, startPayload, audioSourceId, audioSourceKind }: LiveSessionStartOptions): Promise<boolean> {
    if (this.state !== "idle") {
      return false;
    }
    if (!this.audioAvailable) {
      this.setStatus("captureStatus", NO_AUDIO_SOURCE_MESSAGE);
      return false;
    }

    this.selectedAudioSource = { id: audioSourceId, kind: audioSourceKind };
    this.setState("connecting");
    this.setStatus("connectionStatus", "WS...");

    const client = this.createSessionClient(url);
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

  async stop({ sendFinish = true }: { sendFinish?: boolean } = {}): Promise<void> {
    if (this.state === "finishing") {
      await this.abort(FINAL_TRANSCRIPT_CANCELLED_MESSAGE);
      return;
    }
    if (sendFinish) {
      await this.finish();
      return;
    }
    await this.abort();
  }

  private async abort(message = "", { closeSocket = true }: { closeSocket?: boolean } = {}): Promise<void> {
    if (this.state === "idle" && this.client === null) {
      return;
    }

    const client = this.releaseActiveSession();
    await this.stopCaptureOnly();
    if (closeSocket) {
      await client?.close();
    }
    this.setState("idle");
    if (message) {
      this.setStatus("connectionStatus", message);
    }
  }

  private async handleRealtimeEvent(event: RealtimeEvent, client: LiveSessionClient): Promise<void> {
    if (client !== this.client) {
      return;
    }

    if (event.type === "ready") {
      try {
        this.setState("running");
        this.onReady?.(event);
        await this.startCaptureAfterReady();
      } catch (error) {
        const message = errorMessage(error);
        this.setStatus("captureStatus", message);
        await this.abort(message);
      }
      return;
    }

    if (event.type === "error") {
      const message = String(event.error || "Service error");
      if (event.fatal === true || this.state !== "running") {
        await this.abort(message);
        return;
      }
      this.setStatus("connectionStatus", message);
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

  private async finish(): Promise<void> {
    if (this.state === "idle" || this.state === "finishing") {
      return;
    }
    if (this.state === "connecting") {
      await this.abort("Stopped before service was ready.");
      return;
    }

    this.setState("finishing");
    await this.stopCaptureOnly();
    this.setStatus("captureStatus", "Final");
    this.client?.finish();
    this.finishTimeout.schedule(() => this.abort("Timed out waiting for transcript_final."));
  }

  private async complete(): Promise<void> {
    const client = this.releaseActiveSession();
    await this.stopCaptureOnly();
    this.setStatus("captureStatus", "Done");
    await client?.close();
    this.setState("idle");
  }

  private async handleAsrClose(event: CloseEvent, client: LiveSessionClient): Promise<void> {
    if (client !== this.client) {
      return;
    }

    const reason =
      this.state === "finishing"
        ? `WebSocket closed before transcript_final: ${event.code}`
        : `WebSocket closed: ${event.code}`;
    await this.abort(reason, { closeSocket: false });
  }

  private createSessionClient(url: string): LiveSessionClient {
    return this.createClient({
      url,
      onEvent: (event, source) => this.handleRealtimeEvent(event, source),
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
  }

  private releaseActiveSession(): LiveSessionClient | null {
    const client = this.client;
    this.client = null;
    this.selectedAudioSource = null;
    this.finishTimeout.clear();
    return client;
  }

  private async startCaptureAfterReady(): Promise<void> {
    const source = this.selectedAudioSource;
    if (!source) {
      throw new Error("No audio source selected.");
    }

    await this.audioCapture.start({
      sourceId: source.id,
      sourceKind: source.kind,
      sendPcm: (bytes) => this.client?.sendPcm(bytes) ?? false,
    });
  }

  private async stopCaptureOnly(): Promise<void> {
    await this.audioCapture.stop();
  }

  private setState(state: SessionState): void {
    this.state = state;
    this.notifyStateChange();
  }

  private notifyStateChange(): void {
    this.onStateChange?.(this.state, { canStart: this.canStart() });
  }

  private setStatus<K extends StatusKey>(key: K, value: StatusValue<K>): void {
    this.onStatus?.(key, value);
  }
}
