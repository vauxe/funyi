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

interface LiveSessionAudioSource {
  audioSourceId: string;
  audioSourceKind: AudioSourceKind;
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
  private captureSendEpoch = 0;
  private audioSourceSwitch: Promise<void> | null = null;
  private client: LiveSessionClient | null = null;
  private desiredAudioSource: LiveSessionAudioSource | null = null;
  private pendingLanguageConfig: LanguageConfigUpdate | null = null;
  private requestedAudioSource: LiveSessionAudioSource | null = null;
  private selectedAudioSource: LiveSessionAudioSource | null = null;
  private state: SessionState = "idle";
  private teardown: Promise<void> | null = null;

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
    this.audioAvailable = available;
    this.notifyStateChange();
  }

  canStart(): boolean {
    return this.state === "idle" && this.audioAvailable;
  }

  getState(): SessionState {
    return this.state;
  }

  setLanguageConfig(config: LanguageConfigUpdate): void {
    if (this.state === "running" && this.client !== null) {
      this.client.setLanguageConfig(config);
      return;
    }
    // Buffer changes made while the socket is still opening and apply them once
    // the session is ready, so the UI and the server never diverge.
    if (this.state === "connecting") {
      this.pendingLanguageConfig = { ...this.pendingLanguageConfig, ...config };
    }
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

    const source = { audioSourceId, audioSourceKind };
    this.desiredAudioSource = source;
    this.selectedAudioSource = null;
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

  async pause(): Promise<void> {
    const client = this.client;
    if (this.state !== "running" || client === null) {
      return;
    }
    this.setState("paused");
    this.requestedAudioSource = null;
    await this.stopCaptureOnly();
  }

  async resume(): Promise<void> {
    const client = this.client;
    if (this.state !== "paused" || client === null) {
      return;
    }
    const source = this.desiredAudioSource ?? this.selectedAudioSource;
    if (source === null) {
      await this.abort("No audio source selected.");
      return;
    }

    this.setState("running");
    this.flushPendingLanguageConfig();
    const resumed = await this.startCaptureSource(source, client, { abortOnFailure: false });
    if (
      !resumed &&
      this.client === client &&
      this.getState() === "running" &&
      sameAudioSource(this.desiredAudioSource, source)
    ) {
      this.setState("paused");
    }
  }

  async switchAudioSource({ audioSourceId, audioSourceKind }: LiveSessionAudioSource): Promise<void> {
    const client = this.client;
    if (client === null) {
      return;
    }

    const source = { audioSourceId, audioSourceKind };
    this.desiredAudioSource = source;
    if (this.state === "paused") {
      this.requestedAudioSource = null;
      return;
    }
    if (this.state !== "running") {
      return;
    }
    if (
      this.audioSourceSwitch === null &&
      this.audioCapture.isActive() &&
      sameAudioSource(this.selectedAudioSource, source)
    ) {
      return;
    }

    this.requestedAudioSource = source;
    if (this.audioSourceSwitch === null) {
      this.audioSourceSwitch = this.runAudioSourceSwitches(client).finally(() => {
        if (this.client === client) {
          this.audioSourceSwitch = null;
        }
      });
    }
    await this.audioSourceSwitch;
  }

  private async abort(message = "", { closeSocket = true }: { closeSocket?: boolean } = {}): Promise<void> {
    if (this.teardown === null && this.state === "idle" && this.client === null) {
      return;
    }

    await this.runExclusiveTeardown(async (client) => {
      await this.stopCaptureOnly();
      if (closeSocket) {
        await client?.close();
      }
      this.setState("idle");
      if (message) {
        this.setStatus("connectionStatus", message);
      }
    });
  }

  private async runAudioSourceSwitches(client: LiveSessionClient): Promise<void> {
    while (this.isRunningSession(client) && this.requestedAudioSource !== null) {
      const source = this.requestedAudioSource;
      this.requestedAudioSource = null;
      if (this.audioCapture.isActive() && sameAudioSource(this.selectedAudioSource, source)) {
        continue;
      }
      if (this.audioCapture.isActive()) {
        await this.stopCaptureOnly();
        if (!this.isRunningSession(client)) {
          return;
        }
      }
      if (this.requestedAudioSource !== null) {
        continue;
      }
      if (!sameAudioSource(this.desiredAudioSource, source)) {
        continue;
      }
      await this.startCaptureSource(source, client, { abortOnFailure: false });
    }
  }

  private async startCaptureSource(
    source: LiveSessionAudioSource,
    client: LiveSessionClient,
    { abortOnFailure = true }: { abortOnFailure?: boolean } = {},
  ): Promise<boolean> {
    if (!this.isRunningSession(client)) {
      return false;
    }
    const sendEpoch = this.nextCaptureSendEpoch();
    try {
      await this.audioCapture.start({
        abortOnCaptureError: false,
        sourceId: source.audioSourceId,
        sourceKind: source.audioSourceKind,
        sendPcm: (bytes) =>
          this.isRunningSession(client) && this.captureSendEpoch === sendEpoch ? client.sendPcm(bytes) : undefined,
      });
      if (!this.isRunningSession(client)) {
        return false;
      }
      if (!sameAudioSource(this.desiredAudioSource, source)) {
        return false;
      }
      this.selectedAudioSource = source;
      return true;
    } catch (error) {
      const message = errorMessage(error);
      this.setStatus("captureStatus", message);
      if (abortOnFailure && this.isRunningSession(client)) {
        await this.abort(message);
      }
      return false;
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
        this.flushPendingLanguageConfig();
        const source = this.desiredAudioSource ?? this.selectedAudioSource;
        if (source === null) {
          throw new Error("No audio source selected.");
        }
        await this.startCaptureSource(source, client);
      } catch (error) {
        const message = errorMessage(error);
        this.setStatus("captureStatus", message);
        await this.abort(message);
      }
      return;
    }

    if (event.type === "error") {
      const message = String(event.error || "Service error");
      if (event.fatal === true || (this.state !== "running" && this.state !== "paused")) {
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

    // The transcript handler above is async; bail out if the session was torn
    // down while it ran so we do not overwrite a failure with "Done".
    if (client !== this.client) {
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
    const requested = this.client?.finish() ?? false;
    if (!requested) {
      await this.abort("Stopped before the final transcript could be requested.");
      return;
    }
    this.finishTimeout.schedule(() => this.abort("Timed out waiting for transcript_final."));
  }

  private async complete(): Promise<void> {
    await this.runExclusiveTeardown(async (client) => {
      await this.stopCaptureOnly();
      this.setStatus("captureStatus", "Done");
      await client?.close();
      this.setState("idle");
    });
  }

  // Coalesces concurrent teardown paths (abort/complete) so the active session
  // is released and native capture is stopped exactly once.
  private async runExclusiveTeardown(run: (client: LiveSessionClient | null) => Promise<void>): Promise<void> {
    if (this.teardown !== null) {
      return this.teardown;
    }
    const client = this.releaseActiveSession();
    const teardown = run(client).finally(() => {
      this.teardown = null;
    });
    this.teardown = teardown;
    return teardown;
  }

  private flushPendingLanguageConfig(): void {
    const pending = this.pendingLanguageConfig;
    this.pendingLanguageConfig = null;
    if (pending && this.client) {
      this.client.setLanguageConfig(pending);
    }
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
    this.audioSourceSwitch = null;
    this.invalidateCaptureSender();
    this.desiredAudioSource = null;
    this.selectedAudioSource = null;
    this.pendingLanguageConfig = null;
    this.requestedAudioSource = null;
    this.finishTimeout.clear();
    return client;
  }

  private async stopCaptureOnly(): Promise<void> {
    this.invalidateCaptureSender();
    this.selectedAudioSource = null;
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

  private isRunningSession(client: LiveSessionClient): boolean {
    return this.client === client && this.state === "running";
  }

  private nextCaptureSendEpoch(): number {
    this.captureSendEpoch += 1;
    return this.captureSendEpoch;
  }

  private invalidateCaptureSender(): void {
    this.captureSendEpoch += 1;
  }
}

function sameAudioSource(left: LiveSessionAudioSource | null, right: LiveSessionAudioSource): boolean {
  return left !== null && left.audioSourceId === right.audioSourceId && left.audioSourceKind === right.audioSourceKind;
}
