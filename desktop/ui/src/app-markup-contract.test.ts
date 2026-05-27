import test from "node:test";
import assert from "node:assert/strict";

import { APP_ELEMENT_SELECTORS } from "./app-dom.js";
import { RESIZE_DIRECTION_ATTRIBUTE, RESIZE_DIRECTIONS } from "./overlay-contract.js";
import { htmlAttributeValues } from "./test-contract-parsers.fixture.js";
import { readDesktopFile } from "./test-project-files.fixture.js";

const APP_HTML = readDesktopFile("ui", "index.html");

test("app markup contains every element required by the DOM contract", () => {
  const ids = new Set(htmlAttributeValues(APP_HTML, "id"));

  for (const selector of Object.values(APP_ELEMENT_SELECTORS)) {
    assert.ok(selector.startsWith("#"), `app selector must be an id selector: ${selector}`);
    assert.ok(ids.has(selector.slice(1)), `missing markup for ${selector}`);
  }
});

test("app markup exposes the full resize handle contract", () => {
  assert.deepEqual(htmlAttributeValues(APP_HTML, RESIZE_DIRECTION_ATTRIBUTE).sort(), [...RESIZE_DIRECTIONS].sort());
});
