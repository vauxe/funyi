import test from "node:test";
import assert from "node:assert/strict";

import { isActiveSessionState, isSessionConfigurationLocked } from "./session-state.js";

test("describes shared session state semantics", () => {
  assert.equal(isActiveSessionState("idle"), false);
  assert.equal(isActiveSessionState("connecting"), true);
  assert.equal(isActiveSessionState("running"), true);
  assert.equal(isActiveSessionState("finishing"), true);

  assert.equal(isSessionConfigurationLocked("idle"), false);
  assert.equal(isSessionConfigurationLocked("connecting"), true);
  assert.equal(isSessionConfigurationLocked("running"), false);
  assert.equal(isSessionConfigurationLocked("finishing"), true);
});
