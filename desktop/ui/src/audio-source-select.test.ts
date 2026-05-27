import test from "node:test";
import assert from "node:assert/strict";

import { AudioSourceSelect } from "./audio-source-select.js";
import type { AudioSource } from "./audio-source.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { asDomElement, FakeDocument, FakeElement, installFakeDocument } from "./test-dom.fixture.js";

test.beforeEach(() => {
  installFakeDocument(new FakeDocument());
});

test.afterEach(() => {
  clearBrowserGlobals("document");
});

test("renders source labels and tracks selected source kind", () => {
  const select = new FakeElement("select");
  const sourceSelect = new AudioSourceSelect(asDomElement<HTMLSelectElement>(select));

  sourceSelect.render([
    source({
      id: "system",
      name: "Audio",
      kind: "system",
    }),
    source({
      id: "mic",
      name: "Studio Mic",
      kind: "microphone",
    }),
  ]);

  assert.equal(sourceSelect.hasAvailableSource, true);
  assert.deepEqual(select.children.map((child) => child.textContent), [
    "Sys · Audio",
    "Mic · Studio Mic",
  ]);
  assert.equal(sourceSelect.selectedKind, "system");

  select.value = "mic";
  assert.equal(sourceSelect.selectedKind, "microphone");
});

test("preserves unavailable detail and disables unavailable sources", () => {
  const select = new FakeElement("select");
  const sourceSelect = new AudioSourceSelect(asDomElement<HTMLSelectElement>(select));

  sourceSelect.render([
    source({
      id: "missing",
      isAvailable: false,
      name: "Audio",
      detail: "permission missing",
    }),
  ]);

  assert.equal(sourceSelect.hasAvailableSource, false);
  assert.equal(sourceSelect.unavailableDetail, "permission missing");
  assert.equal(select.children[0]?.disabled, true);
  assert.equal(select.children[0]?.textContent, "Sys · Audio unavailable");
  assert.equal(select.value, "");
  assert.equal(sourceSelect.selectedKind, null);
  select.value = "missing";
  assert.equal(sourceSelect.selectedKind, null);
});

test("selects the first available source regardless of enumeration order", () => {
  const select = new FakeElement("select");
  const sourceSelect = new AudioSourceSelect(asDomElement<HTMLSelectElement>(select));

  sourceSelect.render([
    source({ id: "missing", isAvailable: false, name: "Audio" }),
    source({ id: "mic", kind: "microphone", name: "Studio Mic" }),
  ]);

  assert.equal(select.value, "mic");
  assert.equal(sourceSelect.selectedKind, "microphone");
});

test("uses kind-specific source labels when native names are blank", () => {
  const select = new FakeElement("select");
  const sourceSelect = new AudioSourceSelect(asDomElement<HTMLSelectElement>(select));

  sourceSelect.render([
    source({ id: "system", name: "  " }),
    source({ id: "mic", kind: "microphone", name: "" }),
  ]);

  assert.deepEqual(select.children.map((child) => child.textContent), [
    "Sys · Audio",
    "Mic · Microphone",
  ]);
});

function source(overrides: Partial<AudioSource>): AudioSource {
  return {
    detail: "available",
    id: "source",
    isAvailable: true,
    kind: "system",
    name: "Audio",
    ...overrides,
  };
}
