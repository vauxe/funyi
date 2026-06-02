import {
  isActiveSessionState,
  isAudioSourceConfigurationLocked,
  isLanguageConfigurationLocked,
  type SessionState,
} from "./session-state.js";
import type { StatusSummary } from "./status-summary.js";

interface SessionControlsElements {
  appShell: HTMLElement;
  audioSource: HTMLSelectElement;
  language: HTMLSelectElement;
  serverUrl: HTMLInputElement;
  sessionStatus: HTMLElement;
  stopButton: HTMLButtonElement;
  transportButton: HTMLButtonElement;
  translationTargetLanguage: HTMLSelectElement;
  volumeIndicator: HTMLElement;
}

export class SessionControlsView {
  constructor(private readonly elements: SessionControlsElements) {}

  renderState(state: SessionState, { canStart }: { canStart: boolean }): void {
    const active = isActiveSessionState(state);
    const languageConfigurationLocked = isLanguageConfigurationLocked(state);
    const audioSourceLocked = isAudioSourceConfigurationLocked(state);
    this.elements.appShell.setAttribute("data-state", state);
    this.elements.transportButton.disabled =
      state === "connecting" || state === "finishing" || (state === "idle" && !canStart);
    this.elements.transportButton.classList.toggle("is-pause", state === "running");
    const transportButtonLabel = transportButtonLabelForState(state);
    this.elements.transportButton.title = transportButtonLabel;
    this.elements.transportButton.setAttribute("aria-label", transportButtonLabel);
    this.elements.stopButton.disabled = state === "idle";
    this.elements.stopButton.classList.toggle("is-cancel", state === "connecting" || state === "finishing");
    const stopButtonLabel = stopButtonLabelForState(state);
    this.elements.stopButton.title = stopButtonLabel;
    this.elements.stopButton.setAttribute("aria-label", stopButtonLabel);
    this.elements.serverUrl.disabled = active;
    this.elements.language.disabled = languageConfigurationLocked;
    this.elements.translationTargetLanguage.disabled = languageConfigurationLocked;
    this.elements.audioSource.disabled = audioSourceLocked;
  }

  renderStatus({ text, tone, level, volume = 0 }: StatusSummary): void {
    const active = text !== "";
    const audioLevel = level ?? "silent";
    this.elements.appShell.dataset.statusActive = String(active);
    this.elements.sessionStatus.textContent = text;
    this.elements.sessionStatus.dataset.active = String(active);
    this.elements.sessionStatus.dataset.tone = tone;
    if (level) {
      this.elements.sessionStatus.dataset.level = level;
    } else {
      delete this.elements.sessionStatus.dataset.level;
    }
    this.elements.sessionStatus.title = text;
    this.elements.sessionStatus.setAttribute("aria-label", text);
    this.renderVolumeIndicator(audioLevel, volume);
  }

  private renderVolumeIndicator(level: NonNullable<StatusSummary["level"]>, volume: number): void {
    const normalizedVolume = Math.min(1, Math.max(0, volume));
    this.elements.volumeIndicator.dataset.level = level;
    this.elements.volumeIndicator.style.setProperty("--volume-bar-low", volumeBarScale(normalizedVolume, 0.18, 0.42));
    this.elements.volumeIndicator.style.setProperty("--volume-bar-mid", volumeBarScale(normalizedVolume, 0.12, 0.72));
    this.elements.volumeIndicator.style.setProperty("--volume-bar-high", volumeBarScale(normalizedVolume, 0.08, 0.92));
  }
}

function volumeBarScale(volume: number, base: number, range: number): string {
  return (base + volume * range).toFixed(2);
}

function transportButtonLabelForState(state: SessionState): string {
  if (state === "connecting") {
    return "Starting";
  }
  if (state === "running") {
    return "Pause";
  }
  if (state === "paused") {
    return "Resume";
  }
  if (state === "finishing") {
    return "Finalizing";
  }
  return "Start";
}

function stopButtonLabelForState(state: SessionState): string {
  if (state === "finishing") {
    return "Cancel final transcript";
  }
  if (state === "connecting") {
    return "Cancel start";
  }
  return "Stop";
}
