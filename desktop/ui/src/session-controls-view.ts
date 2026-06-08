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
    const activeValue = String(active);
    setDatasetIfChanged(this.elements.appShell, "statusActive", activeValue);
    setDatasetIfChanged(this.elements.sessionStatus, "active", activeValue);
    setTextIfChanged(this.elements.sessionStatus, text);
    setTitleIfChanged(this.elements.sessionStatus, text);
    setAttributeIfChanged(this.elements.sessionStatus, "aria-label", text);
    setDatasetIfChanged(this.elements.sessionStatus, "tone", tone);
    setOptionalDatasetIfChanged(this.elements.sessionStatus, "level", level ?? null);
    this.renderVolumeIndicator(audioLevel, volume);
  }

  private renderVolumeIndicator(level: NonNullable<StatusSummary["level"]>, volume: number): void {
    const normalizedVolume = Math.min(1, Math.max(0, volume));
    setDatasetIfChanged(this.elements.volumeIndicator, "level", level);
    setStylePropertyIfChanged(
      this.elements.volumeIndicator,
      "--volume-bar-low",
      volumeBarScale(normalizedVolume, 0.18, 0.42),
    );
    setStylePropertyIfChanged(
      this.elements.volumeIndicator,
      "--volume-bar-mid",
      volumeBarScale(normalizedVolume, 0.12, 0.72),
    );
    setStylePropertyIfChanged(
      this.elements.volumeIndicator,
      "--volume-bar-high",
      volumeBarScale(normalizedVolume, 0.08, 0.92),
    );
  }
}

function setTextIfChanged(element: HTMLElement, value: string): void {
  if (element.textContent !== value) {
    element.textContent = value;
  }
}

function setTitleIfChanged(element: HTMLElement, value: string): void {
  if (element.title !== value) {
    element.title = value;
  }
}

function setAttributeIfChanged(element: HTMLElement, name: string, value: string): void {
  if (element.getAttribute(name) !== value) {
    element.setAttribute(name, value);
  }
}

function setDatasetIfChanged(element: HTMLElement, key: string, value: string): void {
  if (element.dataset[key] !== value) {
    element.dataset[key] = value;
  }
}

function setOptionalDatasetIfChanged(element: HTMLElement, key: string, value: string | null): void {
  if (value === null) {
    if (element.dataset[key] !== undefined) {
      delete element.dataset[key];
    }
    return;
  }
  setDatasetIfChanged(element, key, value);
}

function setStylePropertyIfChanged(element: HTMLElement, name: string, value: string): void {
  if (element.style.getPropertyValue(name) !== value) {
    element.style.setProperty(name, value);
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
