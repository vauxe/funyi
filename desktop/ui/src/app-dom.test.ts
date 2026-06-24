import test from "node:test";
import assert from "node:assert/strict";

import { getAppElements } from "./app-dom.js";
import { RESIZE_DIRECTIONS } from "./overlay-contract.js";
import { installFakeAppDocument } from "./test-app-document.fixture.js";
import { clearDomGlobals } from "./test-browser-globals.fixture.js";

test.afterEach(() => {
  clearDomGlobals();
});

test("collects app elements and resize handles from stable selectors", () => {
  const elements = installFakeAppDocument({ resizeDirections: RESIZE_DIRECTIONS });

  const dom = getAppElements();

  assert.equal(dom.appShell, elements["app-shell"]);
  assert.deepEqual(
    dom.resizeHandles.map((handle) => handle.direction),
    [...RESIZE_DIRECTIONS],
  );
  assert.equal(elements["app-shell"]!.tagName, "main");
  assert.equal(elements["caption-strip"]!.tagName, "section");
  assert.equal(elements["history-list"]!.tagName, "section");
  assert.equal(elements["history-list"]!.attributes.get("data-overlay-drag-ignore"), "");
  assert.equal(dom.volumeIndicator, elements["volume-indicator"]);
  assert.equal(elements["volume-indicator"]!.tagName, "span");
  assert.equal(elements["resize-north"]!.tagName, "div");
});

test("fails early when required markup is missing", () => {
  installFakeAppDocument({ elementIds: [], resizeDirections: [] });

  assert.throws(() => getAppElements(), /Missing required element: #app-shell/);
});

test("fails early when a resize handle has an invalid direction", () => {
  installFakeAppDocument({ resizeDirections: ["Sideways"] });

  assert.throws(() => getAppElements(), /Invalid resize direction: Sideways/);
});

test("fails early when a resize handle direction is empty", () => {
  installFakeAppDocument({ resizeDirections: [""] });

  assert.throws(() => getAppElements(), /Invalid resize direction: \(empty\)/);
});
