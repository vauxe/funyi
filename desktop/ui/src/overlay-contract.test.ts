import test from "node:test";
import assert from "node:assert/strict";

import {
  OVERLAY_MODES,
  RESIZE_DIRECTION_ATTRIBUTE,
  RESIZE_DIRECTIONS,
  RESIZE_HANDLE_SELECTOR,
  isResizeDirection,
} from "./overlay-contract.js";

test("defines overlay modes as runtime contract values", () => {
  assert.deepEqual(OVERLAY_MODES, ["compact", "history"]);
});

test("validates resize directions from markup", () => {
  assert.deepEqual(RESIZE_DIRECTIONS.filter(isResizeDirection), [...RESIZE_DIRECTIONS]);
  assert.equal(isResizeDirection("Sideways"), false);
  assert.equal(isResizeDirection(undefined), false);
});

test("defines the resize handle markup selector in one place", () => {
  assert.equal(RESIZE_DIRECTION_ATTRIBUTE, "data-resize-direction");
  assert.equal(RESIZE_HANDLE_SELECTOR, "[data-resize-direction]");
});
