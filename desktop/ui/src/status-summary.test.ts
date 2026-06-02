import test from "node:test";
import assert from "node:assert/strict";

import type { StatusValues } from "./session-status.js";
import { summarizeStatus } from "./status-summary.js";

const EMPTY_STATUS: StatusValues = {
  audioHealth: "",
  audioStats: { levelDb: null, droppedFrames: 0 },
  captureStatus: "",
  connectionStatus: "",
};

test("summarizes connection lifecycle without redundant running text", () => {
  assert.deepEqual(summarizeStatus(EMPTY_STATUS, "connecting"), {
    text: "Connecting...",
    tone: "active",
  });
  assert.deepEqual(summarizeStatus(EMPTY_STATUS, "finishing"), {
    text: "Finishing...",
    tone: "active",
  });
  assert.deepEqual(summarizeStatus(EMPTY_STATUS, "running"), {
    text: "",
    tone: "idle",
    level: "silent",
    volume: 0,
  });
  assert.deepEqual(summarizeStatus(EMPTY_STATUS, "paused"), {
    text: "Paused",
    tone: "active",
  });
});

test("maps technical errors to compact user-facing status text", () => {
  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        connectionStatus: "Another realtime session is active.",
      },
      "idle",
    ),
    { text: "Previous session closing", tone: "error" },
  );

  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        connectionStatus: "WebSocket closed before start",
      },
      "idle",
    ),
    { text: "Connection closed.", tone: "error" },
  );

  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        captureStatus: "Microphone permission denied.",
      },
      "idle",
    ),
    { text: "Microphone permission denied.", tone: "error" },
  );

  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        captureStatus: "Selected audio source is invalid.",
      },
      "idle",
    ),
    { text: "Selected audio source is invalid.", tone: "error" },
  );
});

test("reports audio health warnings while preserving level state", () => {
  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        audioHealth: "microphoneSilent",
        audioStats: { levelDb: -48, droppedFrames: 0 },
      },
      "running",
    ),
    { text: "No mic audio", tone: "warn", level: "low", volume: 0.53 },
  );

  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        audioHealth: "systemSilent",
      },
      "running",
    ),
    { text: "No system audio", tone: "warn", level: "silent", volume: 0 },
  );

  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        audioStats: { levelDb: -20, droppedFrames: 2 },
      },
      "running",
    ),
    { text: "Audio lagging", tone: "warn", level: "live", volume: 1 },
  );

  assert.deepEqual(
    summarizeStatus(
      {
        ...EMPTY_STATUS,
        audioHealth: "",
        captureStatus: "Waiting for silent device check",
      },
      "running",
    ),
    { text: "", tone: "idle", level: "silent", volume: 0 },
  );
});
