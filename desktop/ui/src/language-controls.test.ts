import test from "node:test";
import assert from "node:assert/strict";

import { LanguageControls } from "./language-controls.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { asDomElement, FakeDocument, FakeElement, installFakeDocument } from "./test-dom.fixture.js";

test.beforeEach(() => {
  installFakeDocument(new FakeDocument());
});

test.afterEach(() => {
  clearBrowserGlobals("document");
});

test("renders language selectors with stable defaults", () => {
  const language = new FakeElement("select");
  const translationTarget = new FakeElement("select");
  const controls = new LanguageControls(
    asDomElement<HTMLSelectElement>(language),
    asDomElement<HTMLSelectElement>(translationTarget),
  );

  controls.render();

  assert.equal(language.children[0]?.textContent, "Auto");
  assert.equal(translationTarget.children[0]?.textContent, "Off");
  assert.equal(translationTarget.children.some((option) => option.value === "Traditional Chinese"), true);
  assert.equal(translationTarget.children.some((option) => option.value === "Swedish"), false);
  assert.equal(controls.asrLanguage, null);
  assert.equal(controls.targetLanguage, "");
  assert.equal(controls.translationEnabled, false);
});

test("exposes trimmed runtime language config", () => {
  const language = new FakeElement("select");
  const translationTarget = new FakeElement("select");
  const controls = new LanguageControls(
    asDomElement<HTMLSelectElement>(language),
    asDomElement<HTMLSelectElement>(translationTarget),
  );
  controls.render();

  language.value = " Chinese ";
  translationTarget.value = " Japanese ";

  assert.equal(controls.asrLanguage, "Chinese");
  assert.equal(controls.targetLanguage, "Japanese");
  assert.equal(controls.translationEnabled, true);
});
