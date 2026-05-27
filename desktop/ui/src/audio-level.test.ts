import test from "node:test";
import assert from "node:assert/strict";

import { formatAudioStats, isAudible, parseAudioStatsState, pcmLevelDb } from "./audio-level.js";

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

test("formats audio stats with dropped frame count", () => {
  assert.equal(formatAudioStats(null, 0), "Silent");
  assert.equal(formatAudioStats(-48.4, 0), "-48dB");
  assert.equal(formatAudioStats(-20.2, 3), "-20dB, dropped 3");
});

test("reads formatted audio stats into display state", () => {
  assert.deepEqual(parseAudioStatsState(""), { level: "silent", volume: 0, hasDroppedFrames: false });
  assert.deepEqual(parseAudioStatsState("Silent"), { level: "silent", volume: 0, hasDroppedFrames: false });
  assert.deepEqual(parseAudioStatsState("-48dB"), { level: "low", volume: 0.53, hasDroppedFrames: false });
  assert.deepEqual(parseAudioStatsState("-20dB, dropped 3"), { level: "live", volume: 1, hasDroppedFrames: true });
  assert.deepEqual(parseAudioStatsState("-20dB, dropped 0"), { level: "live", volume: 1, hasDroppedFrames: false });
});
