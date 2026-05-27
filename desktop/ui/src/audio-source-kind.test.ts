import test from "node:test";
import assert from "node:assert/strict";

import {
  audioSourceDefaultName,
  audioSourceKindFromAudioHealthStatus,
  audioSourceShortLabel,
  parseAudioSourceKind,
  silentAudioHealthStatus,
  silentAudioSourceStatus,
} from "./audio-source-kind.js";

test("formats audio source kind labels consistently", () => {
  assert.equal(audioSourceShortLabel("system"), "Sys");
  assert.equal(audioSourceShortLabel("microphone"), "Mic");
  assert.equal(audioSourceDefaultName("system"), "Audio");
  assert.equal(audioSourceDefaultName("microphone"), "Microphone");
  assert.equal(silentAudioSourceStatus("system"), "Sys silent");
  assert.equal(silentAudioSourceStatus("microphone"), "Mic silent");
  assert.equal(silentAudioHealthStatus("system"), "systemSilent");
  assert.equal(silentAudioHealthStatus("microphone"), "microphoneSilent");
  assert.equal(audioSourceKindFromAudioHealthStatus("systemSilent"), "system");
  assert.equal(audioSourceKindFromAudioHealthStatus("microphoneSilent"), "microphone");
  assert.equal(audioSourceKindFromAudioHealthStatus(""), null);
});

test("parses audio source kind payloads", () => {
  assert.equal(parseAudioSourceKind("system", "kind"), "system");
  assert.equal(parseAudioSourceKind("microphone", "kind"), "microphone");
  assert.throws(() => parseAudioSourceKind("speaker", "kind"), /kind must be system or microphone/);
});
