import test from "node:test";
import assert from "node:assert/strict";

import type { SessionState } from "./session-state.js";
import { StatusController } from "./status-controller.js";
import type { StatusSummary } from "./status-summary.js";

test("renders summaries from session state and status values", () => {
  let sessionState: SessionState = "connecting";
  const summaries: StatusSummary[] = [];
  const controller = new StatusController({
    render: (summary) => summaries.push(summary),
  });

  controller.setSessionState(sessionState);
  sessionState = "running";
  controller.setSessionState(sessionState);
  controller.setStatus("audioStats", "-20dB, dropped 1");

  assert.deepEqual(summaries, [
    { text: "Connecting...", tone: "active" },
    { text: "", tone: "idle", level: "silent", volume: 0 },
    { text: "Audio lagging", tone: "warn", level: "live", volume: 1 },
  ]);
});

test("keeps stale overlay clears from hiding newer session status", () => {
  const summaries: StatusSummary[] = [];
  const controller = new StatusController({
    render: (summary) => summaries.push(summary),
  });

  controller.setOverlayError(new Error("Minimize failed."));
  controller.setStatus("connectionStatus", "WebSocket closed: 1006");
  controller.clearOverlayError();

  assert.equal(summaries.at(-1)?.text, "Connection closed.");
});

test("clears overlay-owned errors", () => {
  const summaries: StatusSummary[] = [];
  const controller = new StatusController({
    render: (summary) => summaries.push(summary),
  });

  controller.setOverlayError("Close failed.");
  controller.clearOverlayError();

  assert.deepEqual(summaries.at(-1), { text: "", tone: "idle" });
});
