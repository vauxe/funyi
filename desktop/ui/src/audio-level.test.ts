import test from "node:test";
import assert from "node:assert/strict";

import { audioStatsState, isAudible, pcmLevelDb } from "./audio-level.js";

test("computes pcm level from little-endian signed 16-bit samples", () => {
  const level = pcmLevelDb(new Uint8Array([0xff, 0x7f, 0x01, 0x80]));

  assert.notEqual(level, null);
  assert.ok(level !== null && level > -0.01);
});

test("treats empty and silent pcm as inaudible", () => {
  assert.equal(pcmLevelDb(new Uint8Array()), null);
  assert.equal(pcmLevelDb(new Uint8Array([0, 0, 0, 0])), null);
  assert.equal(isAudible(null), false);
  assert.equal(isAudible(-90), false);
  assert.equal(isAudible(-40), true);
});

test("derives display state from structured audio stats", () => {
  assert.deepEqual(audioStatsState({ levelDb: null, droppedFrames: 0 }), {
    level: "silent",
    volume: 0,
    hasDroppedFrames: false,
  });
  assert.deepEqual(audioStatsState({ levelDb: -48.4, droppedFrames: 0 }), {
    level: "low",
    volume: 0.53,
    hasDroppedFrames: false,
  });
  assert.deepEqual(audioStatsState({ levelDb: -20.2, droppedFrames: 3 }), {
    level: "live",
    volume: 1,
    hasDroppedFrames: true,
  });
  assert.deepEqual(audioStatsState({ levelDb: -20.2, droppedFrames: 0 }), {
    level: "live",
    volume: 1,
    hasDroppedFrames: false,
  });
});
