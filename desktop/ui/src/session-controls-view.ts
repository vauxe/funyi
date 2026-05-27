import { isActiveSessionState, isSessionConfigurationLocked, type SessionState } from "./session-state.js";
import type { StatusSummary } from "./status-summary.js";

interface SessionControlsElements {
  appShell: HTMLElement;
  audioSource: HTMLSelectElement;
  language: HTMLSelectElement;
  serverUrl: HTMLInputElement;
  sessionButton: HTMLButtonElement;
  sessionStatus: HTMLElement;
  translationTargetLanguage: HTMLSelectElement;
}

export class SessionControlsView {
  constructor(private readonly elements: SessionControlsElements) {}

  renderState(state: SessionState, { canStart }: { canStart: boolean }): void {
    const active = isActiveSessionState(state);
    const configurationLocked = isSessionConfigurationLocked(state);
    this.elements.appShell.setAttribute("data-state", state);
    this.elements.sessionButton.disabled = state === "idle" && !canStart;
    this.elements.sessionButton.classList.toggle("is-stop", active);
    this.elements.sessionButton.classList.toggle("is-finishing", state === "finishing");
    const sessionButtonLabel = buttonLabelForState(state);
    this.elements.sessionButton.title = sessionButtonLabel;
    this.elements.sessionButton.setAttribute("aria-label", sessionButtonLabel);
    this.elements.serverUrl.disabled = active;
    this.elements.language.disabled = configurationLocked;
    this.elements.translationTargetLanguage.disabled = configurationLocked;
    this.elements.audioSource.disabled = active;
  }

  renderStatus({ text, tone, level }: StatusSummary): void {
    const active = text !== "";
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
  }
}

function buttonLabelForState(state: SessionState): string {
  if (state === "finishing") {
    return "Cancel final transcript";
  }
  if (isActiveSessionState(state)) {
    return "Stop";
  }
  return "Start";
}
