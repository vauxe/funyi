import test from "node:test";
import assert from "node:assert/strict";

import { overlayDragFinishedParseError, parseOverlayDragFinished, parseOverlayDragId } from "./overlay-events.js";

test("parses overlay drag finished payload with camelCase drag id", () => {
  assert.deepEqual(parseOverlayDragFinished({ dragId: 42 }), { dragId: 42 });
  assert.deepEqual(parseOverlayDragFinished({ dragId: 42, error: "rebound failed" }), {
    dragId: 42,
    error: "rebound failed",
  });
});

test("normalizes empty overlay drag finished error text after parsing the drag id", () => {
  assert.deepEqual(parseOverlayDragFinished({ dragId: 42, error: "" }), {
    dragId: 42,
    error: "overlay drag release failed",
  });
});

test("rejects overlay drag finished payloads outside the frontend contract", () => {
  assert.throws(() => parseOverlayDragFinished({ drag_id: 42 }), /dragId must be a non-negative integer/);
  assert.throws(() => parseOverlayDragFinished({ dragId: 42, error: 5 }), /error must be a string/);
  assert.throws(() => parseOverlayDragFinished({ dragId: 42, error: null }), /error must be a string/);
  assert.throws(() => parseOverlayDragFinished({ dragId: 42, error: {} }), /error must be a string/);
  assert.throws(() => parseOverlayDragFinished(null), /payload must be an object/);
  assert.throws(() => parseOverlayDragFinished([{ dragId: 42 }]), /payload must be an object/);
});

test("rejects invalid overlay drag ids", () => {
  assert.throws(() => parseOverlayDragId(-1), /must be a non-negative integer/);
  assert.throws(() => parseOverlayDragId(1.5), /must be a non-negative integer/);
  assert.throws(() => parseOverlayDragId("1"), /must be a non-negative integer/);
});

test("converts parser failures into a finish event that can clear active native drag", () => {
  assert.deepEqual(overlayDragFinishedParseError(new Error("bad payload")), {
    dragId: null,
    error: "bad payload",
  });
  assert.deepEqual(overlayDragFinishedParseError(""), {
    dragId: null,
    error: "overlay drag release failed",
  });
});
