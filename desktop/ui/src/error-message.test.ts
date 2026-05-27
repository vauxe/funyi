import test from "node:test";
import assert from "node:assert/strict";

import { errorMessage } from "./error-message.js";

test("formats unknown errors consistently", () => {
  assert.equal(errorMessage(new Error("failed")), "failed");
  assert.equal(errorMessage("failed"), "failed");
  assert.equal(errorMessage(null), "null");
});
