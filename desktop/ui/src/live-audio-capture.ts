import { isExpectedAudioFrame } from "./audio-format.js";
import type { AudioAdapter } from "./audio-adapter.js";
import { AUDIO_CAPTURE_FAILED_MESSAGE, type Unlisten } from "./audio-capture-events.js";
import { EMPTY_AUDIO_STATS, isAudible, pcmLevelDb } from "./audio-level.js";
import {
  audioSourceShortLabel,
  silentAudioHealthStatus,
  silentAudioSourceStatus,
  type AudioSourceKind,
} from "./audio-source-kind.js";
import { errorMessage } from "./error-message.js";
import type { StatusKey, StatusValue } from "./session-status.js";

const SILENT_FRAME_WARNING_THRESHOLD = 30;

interface LiveAudioCaptureOptions {
  audio: AudioAdapter;
  onAbort(message: string): void;
  onStatus<K extends StatusKey>(key: K, value: StatusValue<K>): void;
}

interface StartCaptureOptions {
  sourceId: string;
  sourceKind: AudioSourceKind;
  sendPcm(bytes: Uint8Array): boolean;
}

export class LiveAudioCapture {
  private droppedAudioFrames = 0;
  private lastAudioLevelDb: number | null = null;
  private sendPcm: ((bytes: Uint8Array) => boolean) | null = null;
  private silentFrameWarningActive = false;
  private silentFrames = 0;
  private sourceKind: AudioSourceKind = "system";
  private unlistenCaptureError: Unlisten | null = null;
  private unlistenFrame: Unlisten | null = null;

  constructor(private readonly options: LiveAudioCaptureOptions) {}

  resetStats(): void {
    this.droppedAudioFrames = 0;
    this.lastAudioLevelDb = null;
    this.silentFrameWarningActive = false;
    this.silentFrames = 0;
    this.setStatus("audioHealth", "");
    this.setStatus("audioStats", EMPTY_AUDIO_STATS);
  }

  async start({ sourceId, sourceKind, sendPcm }: StartCaptureOptions): Promise<void> {
    if (this.unlistenFrame !== null) {
      return;
    }

    this.sourceKind = sourceKind;
    this.sendPcm = sendPcm;
    try {
      this.unlistenFrame = await this.options.audio.listenFrames((frame) => this.handleAudioFrame(frame));
      this.unlistenCaptureError = await this.options.audio.listenCaptureErrors((payload) => {
        this.abortCapture(payload.message);
      });
      await this.options.audio.startCapture(sourceId);
    } catch (error) {
      this.clearListeners();
      this.sendPcm = null;
      throw error;
    }
    this.setStatus("audioHealth", "");
    this.setStatus("captureStatus", this.captureSourceLabel());
  }

  async stop(): Promise<void> {
    try {
      await this.options.audio.stopCapture();
    } catch (error) {
      this.setStatus("captureStatus", errorMessage(error));
    }
    this.clearListeners();
    this.sendPcm = null;
  }

  private handleAudioFrame(frame: unknown): void {
    if (!isExpectedAudioFrame(frame)) {
      return;
    }

    let bytes: Uint8Array;
    try {
      bytes = this.options.audio.decodePcm(frame.dataBase64);
    } catch (error) {
      this.abortCapture(errorMessage(error));
      return;
    }

    this.lastAudioLevelDb = pcmLevelDb(bytes);
    this.updateSilentCaptureStatus(this.lastAudioLevelDb);
    if (this.sendPcm && !this.sendPcm(bytes)) {
      this.droppedAudioFrames += 1;
    }
    this.updateAudioStats();
  }

  private setStatus<K extends StatusKey>(key: K, value: StatusValue<K>): void {
    this.options.onStatus(key, value);
  }

  private abortCapture(message: string): void {
    const status = message || AUDIO_CAPTURE_FAILED_MESSAGE;
    this.setStatus("captureStatus", status);
    this.options.onAbort(status);
  }

  private clearListeners(): void {
    this.unlistenFrame?.();
    this.unlistenCaptureError?.();
    this.unlistenFrame = null;
    this.unlistenCaptureError = null;
  }

  private updateAudioStats(): void {
    this.setStatus("audioStats", { levelDb: this.lastAudioLevelDb, droppedFrames: this.droppedAudioFrames });
  }

  private updateSilentCaptureStatus(levelDb: number | null): void {
    if (isAudible(levelDb)) {
      this.silentFrames = 0;
      if (this.silentFrameWarningActive) {
        this.silentFrameWarningActive = false;
        this.setStatus("audioHealth", "");
        this.setStatus("captureStatus", this.captureSourceLabel());
      }
      return;
    }

    this.silentFrames += 1;
    if (!this.silentFrameWarningActive && this.silentFrames >= SILENT_FRAME_WARNING_THRESHOLD) {
      this.silentFrameWarningActive = true;
      this.setStatus("audioHealth", silentAudioHealthStatus(this.sourceKind));
      this.setStatus("captureStatus", this.silentCaptureMessage());
    }
  }

  private captureSourceLabel(): string {
    return audioSourceShortLabel(this.sourceKind);
  }

  private silentCaptureMessage(): string {
    return silentAudioSourceStatus(this.sourceKind);
  }
}
