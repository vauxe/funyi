import test from "node:test";
import assert from "node:assert/strict";

import { RESIZE_DIRECTION_ATTRIBUTE } from "./overlay-contract.js";
import {
  htmlElementById,
  htmlElements,
  type HtmlAttributes,
} from "./test-contract-parsers.fixture.js";
import { readDesktopFile } from "./test-project-files.fixture.js";

const APP_HTML = readDesktopFile("ui", "index.html");

test("icon-only buttons have stable accessible names", () => {
  const buttons = htmlElements(APP_HTML, "button").filter((button) => hasClass(button, "icon-button"));

  assert.ok(buttons.length > 0, "expected icon-only controls");
  for (const button of buttons) {
    assert.equal(button.type, "button", `#${button.id} should not submit forms`);
    assert.ok(button["aria-label"], `#${button.id} is missing aria-label`);
    assert.equal(button.title, button["aria-label"], `#${button.id} title should match aria-label`);
  }
});

test("non-visual control groups and resize handles stay screen-reader coherent", () => {
  assert.deepEqual(
    namedGroup(".caption-controls"),
    { role: "group", "aria-label": "Session controls" },
  );
  assert.deepEqual(
    namedGroup(".language-settings"),
    { role: "group", "aria-label": "Language settings" },
  );
  assert.equal(htmlElementById(APP_HTML, "session-status").id, "session-status");

  const resizeHandles = htmlElements(APP_HTML, "div").filter((element) => RESIZE_DIRECTION_ATTRIBUTE in element);
  assert.ok(resizeHandles.length > 0, "expected resize handles");
  for (const handle of resizeHandles) {
    assert.equal(handle["aria-hidden"], "true", `#${handle.id} should be hidden from assistive tech`);
  }
});

test("status updates are announced politely", () => {
  const statusLine = htmlElements(APP_HTML, "section").find((section) => hasClass(section, "status-line"));

  assert.equal(statusLine?.["aria-live"], "polite");
});

function namedGroup(className: string): Pick<HtmlAttributes, "aria-label" | "role"> {
  const classToken = className.replace(/^\./u, "");
  const element = htmlElements(APP_HTML).find((candidate) => hasClass(candidate, classToken));
  assert.ok(element, `missing ${className}`);
  const role = element.role;
  const label = element["aria-label"];
  assert.ok(role, `${className} is missing role`);
  assert.ok(label, `${className} is missing aria-label`);
  return {
    role,
    "aria-label": label,
  };
}

function hasClass(element: HtmlAttributes, className: string): boolean {
  return (element.class || "").split(/\s+/u).includes(className);
}
