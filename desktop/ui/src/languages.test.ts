import test from "node:test";
import assert from "node:assert/strict";

import { languageTag } from "./languages.js";

test("languageTag maps display names to BCP-47 tags", () => {
  assert.equal(languageTag("English"), "en");
  assert.equal(languageTag("Chinese"), "zh");
  assert.equal(languageTag("Traditional Chinese"), "zh-Hant");
  assert.equal(languageTag("Cantonese"), "yue");
});

test("languageTag passes through already-valid BCP-47 tags", () => {
  assert.equal(languageTag("en-US"), "en-US");
  assert.equal(languageTag("zh-Hant"), "zh-Hant");
});

test("languageTag returns empty for missing, unknown, or unsafe values", () => {
  assert.equal(languageTag(undefined), "");
  assert.equal(languageTag(""), "");
  assert.equal(languageTag("Klingon"), "");
  assert.equal(languageTag('en" onmouseover=x'), "");
});
