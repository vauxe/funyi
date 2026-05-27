import { errorMessage } from "./error-message.js";
import type { SessionState } from "./session-state.js";
import type { StatusKey, StatusValue, StatusValues } from "./session-status.js";
import { summarizeStatus, type StatusSummary } from "./status-summary.js";

interface StatusControllerOptions {
  render(summary: StatusSummary): void;
}

export class StatusController {
  private overlayOwnsConnectionStatus = false;
  private sessionState: SessionState = "idle";
  private readonly values: StatusValues = {
    audioHealth: "",
    audioStats: "",
    captureStatus: "",
    connectionStatus: "",
  };

  constructor(private readonly options: StatusControllerOptions) {}

  setSessionState(state: SessionState): void {
    this.sessionState = state;
    this.render();
  }

  setStatus<K extends StatusKey>(key: K, value: StatusValue<K>): void {
    if (key === "connectionStatus") {
      this.overlayOwnsConnectionStatus = false;
    }
    this.values[key] = value || "";
    this.render();
  }

  setOverlayError(error: unknown): void {
    this.overlayOwnsConnectionStatus = true;
    this.values.connectionStatus = errorMessage(error);
    this.render();
  }

  clearOverlayError(): void {
    if (!this.overlayOwnsConnectionStatus) {
      return;
    }
    this.overlayOwnsConnectionStatus = false;
    this.values.connectionStatus = "";
    this.render();
  }

  private render(): void {
    this.options.render(summarizeStatus(this.values, this.sessionState));
  }
}
