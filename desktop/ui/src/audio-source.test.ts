import test from "node:test";
import assert from "node:assert/strict";

import { parseAudioSources } from "./audio-source.js";

test("parses native audio source payloads", () => {
  assert.deepEqual(
    parseAudioSources([
      {
        id: "system_default",
        name: "Audio",
        kind: "system",
        isAvailable: true,
        detail: "available",
      },
    ]),
    [
      {
        id: "system_default",
        name: "Audio",
        kind: "system",
        isAvailable: true,
        detail: "available",
      },
    ],
  );
});

test("rejects invalid native audio source payloads", () => {
  assert.throws(() => parseAudioSources({ id: "system_default" }), /audio sources payload must be an array/);
  assert.throws(() => parseAudioSources([null]), /audio source 0 must be an object/);
  assert.throws(
    () => parseAudioSources([{ id: "system_default", name: "Audio", kind: "speaker", isAvailable: true, detail: "" }]),
    /audio source 0\.kind must be system or microphone/,
  );
  assert.throws(
    () => parseAudioSources([{ id: "system_default", name: "Audio", kind: "system", isAvailable: "yes", detail: "" }]),
    /audio source 0\.isAvailable must be a boolean/,
  );
});
