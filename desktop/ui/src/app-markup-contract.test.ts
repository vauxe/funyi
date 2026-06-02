import test from "node:test";
import assert from "node:assert/strict";

import { APP_ELEMENT_SELECTORS } from "./app-dom.js";
import { RESIZE_DIRECTION_ATTRIBUTE, RESIZE_DIRECTIONS } from "./overlay-contract.js";
import { htmlAttributeValues, htmlElementById } from "./test-contract-parsers.fixture.js";
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

test("caption opacity control exposes the full 0 to 100 percent range", () => {
  const opacity = htmlElementById(APP_HTML, "caption-opacity");

  assert.equal(opacity.type, "range");
  assert.equal(opacity.min, "0");
  assert.equal(opacity.max, "100");
  assert.equal(opacity.step, "1");
});

test("caption controls put the session action before secondary actions", () => {
  assert.deepEqual(controlIdsInOrder("caption-controls"), [
    "volume-indicator",
    "transport-button",
    "stop-button",
    "settings-button",
    "minimize-button",
    "close-button",
  ]);
});

function controlIdsInOrder(className: string): string[] {
  const groupMatch = new RegExp(`<div\\b(?=[^>]*\\bclass="${className}")[\\s\\S]*?</div>`, "u").exec(APP_HTML);
  assert.ok(groupMatch?.[0], `missing control group .${className}`);
  return [...groupMatch[0].matchAll(/\bid="([^"]+)"/gu)]
    .map((match) => match[1])
    .filter((id): id is string => Boolean(id));
}
