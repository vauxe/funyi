import test from "node:test";
import assert from "node:assert/strict";

import {
  isRecord,
  optionalRecord,
  recordArray,
  requiredBoolean,
  requiredRecord,
  requiredString,
} from "./runtime-guards.js";

test("recognizes object records without accepting arrays", () => {
  assert.equal(isRecord({}), true);
  assert.equal(isRecord({ type: "ready" }), true);
  assert.equal(isRecord([]), false);
  assert.equal(isRecord(null), false);
  assert.equal(isRecord("ready"), false);
});

test("validates common payload field shapes", () => {
  assert.deepEqual(requiredRecord({ id: "source" }, "source"), { id: "source" });
  assert.equal(optionalRecord(undefined, "partial"), null);
  assert.equal(requiredString("Audio", "source.name"), "Audio");
  assert.equal(requiredBoolean(false, "source.isAvailable"), false);

  assert.deepEqual(recordArray([{ id: "one" }], "segments"), [{ id: "one" }]);
  assert.deepEqual(recordArray(undefined, "segments"), []);
});

test("reports field-specific validation errors", () => {
  assert.throws(() => requiredRecord([], "source"), /source must be an object/);
  assert.throws(() => recordArray({}, "segments"), /segments must be an array/);
  assert.throws(() => recordArray([null], "segments"), /segments item must be an object/);
  assert.throws(() => requiredString(3, "source.name"), /source\.name must be a string/);
  assert.throws(() => requiredBoolean("yes", "source.isAvailable"), /source\.isAvailable must be a boolean/);
});
