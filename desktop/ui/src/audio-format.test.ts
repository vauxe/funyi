import test from "node:test";
import assert from "node:assert/strict";

import { AUDIO_FORMAT, AUDIO_SAMPLE_RATE, decodeBase64Pcm, isExpectedAudioFrame } from "./audio-format.js";

test("defines the realtime pcm format contract", () => {
  assert.equal(AUDIO_SAMPLE_RATE, 16000);
  assert.equal(AUDIO_FORMAT, "pcm_s16le");
  assert.equal(isExpectedAudioFrame({ sampleRate: 16000, format: "pcm_s16le", dataBase64: "AA==" }), true);
  assert.equal(isExpectedAudioFrame({ sampleRate: 48000, format: "pcm_s16le" }), false);
  assert.equal(isExpectedAudioFrame({ sampleRate: 16000, format: "f32le" }), false);
  assert.equal(isExpectedAudioFrame({ sampleRate: 16000, format: "pcm_s16le" }), false);
  assert.equal(isExpectedAudioFrame(null), false);
  assert.equal(isExpectedAudioFrame([]), false);
});

test("decodes base64 pcm bytes", () => {
  assert.deepEqual([...decodeBase64Pcm("AQID")], [1, 2, 3]);
});
