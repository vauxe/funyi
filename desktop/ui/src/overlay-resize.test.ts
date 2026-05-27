import test from "node:test";
import assert from "node:assert/strict";

import { MIN_COMPACT_HEIGHT, nextCompactHeight } from "./overlay-resize.js";

test("computes compact height from vertical resize direction", () => {
  assert.equal(nextCompactHeight({ direction: "North", startY: 100, startHeight: 180 }, 80, 180), 200);
  assert.equal(nextCompactHeight({ direction: "SouthEast", startY: 100, startHeight: 180 }, 125, 180), 205);
});

test("keeps horizontal compact resize on the current height", () => {
  assert.equal(nextCompactHeight({ direction: "East", startY: 100, startHeight: 180 }, 125, 190), 190);
});

test("clamps compact height and preserves fallback for invalid math", () => {
  assert.equal(nextCompactHeight({ direction: "South", startY: 100, startHeight: 40 }, 80, 180), MIN_COMPACT_HEIGHT);
  assert.equal(nextCompactHeight({ direction: "South", startY: Number.NaN, startHeight: 180 }, 80, 192), 192);
});
