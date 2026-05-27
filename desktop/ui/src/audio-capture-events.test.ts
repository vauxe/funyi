import test from "node:test";
import assert from "node:assert/strict";

import { parseAudioCaptureError } from "./audio-capture-events.js";

test("parses native audio capture error payloads", () => {
  assert.deepEqual(parseAudioCaptureError({ message: "device lost" }), { message: "device lost" });
  assert.deepEqual(parseAudioCaptureError({ message: "" }), { message: "Audio capture failed." });
  assert.deepEqual(parseAudioCaptureError(null), { message: "Audio capture failed." });
  assert.deepEqual(parseAudioCaptureError([]), { message: "Audio capture failed." });
});
