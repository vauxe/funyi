import test from "node:test";
import assert from "node:assert/strict";

import {
  isActiveSessionState,
  isAudioSourceConfigurationLocked,
  isLanguageConfigurationLocked,
} from "./session-state.js";

test("describes shared session state semantics", () => {
  assert.equal(isActiveSessionState("idle"), false);
  assert.equal(isActiveSessionState("connecting"), true);
  assert.equal(isActiveSessionState("running"), true);
  assert.equal(isActiveSessionState("paused"), true);
  assert.equal(isActiveSessionState("finishing"), true);

  assert.equal(isLanguageConfigurationLocked("idle"), false);
  assert.equal(isLanguageConfigurationLocked("connecting"), true);
  assert.equal(isLanguageConfigurationLocked("running"), false);
  assert.equal(isLanguageConfigurationLocked("paused"), true);
  assert.equal(isLanguageConfigurationLocked("finishing"), true);

  assert.equal(isAudioSourceConfigurationLocked("idle"), false);
  assert.equal(isAudioSourceConfigurationLocked("connecting"), true);
  assert.equal(isAudioSourceConfigurationLocked("running"), false);
  assert.equal(isAudioSourceConfigurationLocked("paused"), false);
  assert.equal(isAudioSourceConfigurationLocked("finishing"), true);
});
