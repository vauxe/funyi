import test from "node:test";
import assert from "node:assert/strict";

import type { PreparedBackground } from "./background-image.js";
import { MemoryKeyValueStore, PreferencesStore } from "./preferences.js";
import { SettingsController } from "./settings-controller.js";
import { nextTick } from "./test-async.fixture.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { asDomElement, FakeElement, installFakeElementConstructors } from "./test-dom.fixture.js";

test.beforeEach(() => {
  installFakeElementConstructors();
});

test.afterEach(() => {
  clearBrowserGlobals("Element", "HTMLElement");
});

interface Harness {
  controller: SettingsController;
  elements: Record<string, FakeElement>;
  store: PreferencesStore;
  copied: string[];
  revoked: string[];
  prepared: PreparedBackground;
  setTranscript(value: string): void;
  failBackground(error: Error): void;
}

function createHarness(): Harness {
  const elements: Record<string, FakeElement> = {
    root: new FakeElement("main", "app-shell"),
    settingsButton: new FakeElement("button", "settings-button"),
    settingsPanel: new FakeElement("section", "settings-panel"),
    captionOpacity: new FakeElement("input"),
    backgroundButton: new FakeElement("button"),
    backgroundFile: new FakeElement("input"),
    backgroundClearButton: new FakeElement("button"),
    exportButton: new FakeElement("button"),
    settingsStatus: new FakeElement("p"),
  };
  const store = new PreferencesStore(new MemoryKeyValueStore());
  const copied: string[] = [];
  const revoked: string[] = [];
  const prepared: PreparedBackground = { stored: { mime: "image/jpeg", data: "AAAA" }, objectUrl: "blob:new" };
  let transcript = "transcript text";
  let prepareError: Error | null = null;

  const controller = new SettingsController({
    elements: {
      root: asDomElement(elements.root!),
      settingsButton: asDomElement<HTMLButtonElement>(elements.settingsButton!),
      settingsPanel: asDomElement(elements.settingsPanel!),
      captionOpacity: asDomElement<HTMLInputElement>(elements.captionOpacity!),
      backgroundButton: asDomElement<HTMLButtonElement>(elements.backgroundButton!),
      backgroundFile: asDomElement<HTMLInputElement>(elements.backgroundFile!),
      backgroundClearButton: asDomElement<HTMLButtonElement>(elements.backgroundClearButton!),
      exportButton: asDomElement<HTMLButtonElement>(elements.exportButton!),
      settingsStatus: asDomElement(elements.settingsStatus!),
    },
    preferences: store,
    buildTranscript: () => transcript,
    copyText: async (text) => {
      copied.push(text);
    },
    prepareBackground: async () => {
      if (prepareError) {
        throw prepareError;
      }
      return prepared;
    },
    objectUrlFromStored: () => "blob:restored",
    revokeObjectUrl: (url) => revoked.push(url),
  });

  return {
    controller,
    elements,
    store,
    copied,
    revoked,
    prepared,
    setTranscript: (value) => {
      transcript = value;
    },
    failBackground: (error) => {
      prepareError = error;
    },
  };
}

test("init applies stored opacity and starts closed", () => {
  const harness = createHarness();
  harness.store.save({ captionOpacity: 0.4 });

  harness.controller.init();

  assert.equal(harness.elements.captionOpacity!.value, "40");
  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-opacity"), "0.40");
  assert.equal(harness.elements.settingsPanel!.attributes.get("data-open"), "false");
  assert.equal(harness.elements.settingsButton!.attributes.get("aria-expanded"), "false");
});

test("init restores a stored background image", () => {
  const harness = createHarness();
  harness.store.saveBackground({ mime: "image/jpeg", data: "AAAA" });

  harness.controller.init();

  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-image"), 'url("blob:restored")');
});

test("opacity input applies and persists", () => {
  const harness = createHarness();
  harness.controller.init();

  harness.elements.captionOpacity!.value = "50";
  harness.elements.captionOpacity!.dispatch("input", {});

  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-opacity"), "0.50");
  assert.equal(harness.store.load().captionOpacity, 0.5);
});

test("the settings button toggles the panel and Escape closes it", () => {
  const harness = createHarness();
  harness.controller.init();

  harness.elements.settingsButton!.dispatch("click", { stopPropagation: () => {} });
  assert.equal(harness.elements.settingsPanel!.attributes.get("data-open"), "true");
  assert.equal(harness.elements.settingsButton!.attributes.get("aria-expanded"), "true");

  harness.elements.root!.dispatch("keydown", { key: "Escape" });
  assert.equal(harness.elements.settingsPanel!.attributes.get("data-open"), "false");
});

test("a pointerdown outside the panel closes it", () => {
  const harness = createHarness();
  harness.controller.init();
  harness.elements.settingsButton!.dispatch("click", { stopPropagation: () => {} });
  assert.equal(harness.elements.settingsPanel!.attributes.get("data-open"), "true");

  harness.elements.root!.dispatch("pointerdown", { target: new FakeElement("div", "elsewhere") });

  assert.equal(harness.elements.settingsPanel!.attributes.get("data-open"), "false");
});

test("a pointerdown inside the panel keeps it open", () => {
  const harness = createHarness();
  harness.controller.init();
  harness.elements.settingsButton!.dispatch("click", { stopPropagation: () => {} });

  harness.elements.root!.dispatch("pointerdown", { target: harness.elements.settingsPanel });

  assert.equal(harness.elements.settingsPanel!.attributes.get("data-open"), "true");
});

test("export copies the transcript and reports success", async () => {
  const harness = createHarness();
  harness.controller.init();

  harness.elements.exportButton!.dispatch("click", {});
  await nextTick();

  assert.deepEqual(harness.copied, ["transcript text"]);
  assert.equal(harness.elements.settingsStatus!.textContent, "Copied to clipboard");
});

test("export reports when there is nothing to copy", async () => {
  const harness = createHarness();
  harness.setTranscript("");
  harness.controller.init();

  harness.elements.exportButton!.dispatch("click", {});
  await nextTick();

  assert.deepEqual(harness.copied, []);
  assert.equal(harness.elements.settingsStatus!.textContent, "Nothing to copy yet");
});

test("choosing a file persists, applies, and resets the input", async () => {
  const harness = createHarness();
  harness.controller.init();
  (harness.elements.backgroundFile! as unknown as { files: Blob[] }).files = [new Blob(["x"])];

  harness.elements.backgroundFile!.dispatch("change", {});
  await nextTick();

  assert.deepEqual(harness.store.loadBackground(), { mime: "image/jpeg", data: "AAAA" });
  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-image"), 'url("blob:new")');
  assert.equal(harness.elements.settingsStatus!.textContent, "Background updated");
  assert.equal(harness.elements.backgroundFile!.value, "");
});

test("clearing the background removes it and revokes the object URL", () => {
  const harness = createHarness();
  harness.store.saveBackground({ mime: "image/jpeg", data: "AAAA" });
  harness.controller.init();

  harness.elements.backgroundClearButton!.dispatch("click", {});

  assert.equal(harness.store.loadBackground(), null);
  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-image"), "none");
  assert.equal(harness.elements.settingsStatus!.textContent, "Background cleared");
  assert.deepEqual(harness.revoked, ["blob:restored"]);
});

test("a quota failure surfaces a friendly message and persists nothing", async () => {
  const harness = createHarness();
  harness.controller.init();
  harness.failBackground(new Error("QuotaExceededError: ..."));
  (harness.elements.backgroundFile! as unknown as { files: Blob[] }).files = [new Blob(["x"])];

  harness.elements.backgroundFile!.dispatch("change", {});
  await nextTick();

  assert.equal(harness.elements.settingsStatus!.textContent, "Image too large to save");
  assert.equal(harness.store.loadBackground(), null);
  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-image"), "none");
});

test("a generic preparation failure reports a generic message", async () => {
  const harness = createHarness();
  harness.controller.init();
  harness.failBackground(new Error("decode boom"));
  (harness.elements.backgroundFile! as unknown as { files: Blob[] }).files = [new Blob(["x"])];

  harness.elements.backgroundFile!.dispatch("change", {});
  await nextTick();

  assert.equal(harness.elements.settingsStatus!.textContent, "Couldn't set background");
});

test("opacity clamps at the readable floor and ceiling", () => {
  const harness = createHarness();
  harness.controller.init();

  harness.elements.captionOpacity!.value = "0";
  harness.elements.captionOpacity!.dispatch("input", {});
  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-opacity"), "0.20");
  assert.equal(harness.store.load().captionOpacity, 0.2);

  harness.elements.captionOpacity!.value = "100";
  harness.elements.captionOpacity!.dispatch("input", {});
  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-opacity"), "1.00");
  assert.equal(harness.store.load().captionOpacity, 1);
});

test("a non-numeric opacity value is ignored", () => {
  const harness = createHarness();
  harness.controller.init();

  harness.elements.captionOpacity!.value = "abc";
  harness.elements.captionOpacity!.dispatch("input", {});

  assert.equal(harness.elements.root!.styleValues.get("--caption-bg-opacity"), "0.72");
  assert.equal(harness.store.load().captionOpacity, null);
});

test("a change event with no file still resets the input for re-selection", async () => {
  const harness = createHarness();
  harness.controller.init();
  harness.elements.backgroundFile!.value = "C:/fake/path.png";

  harness.elements.backgroundFile!.dispatch("change", {});
  await nextTick();

  assert.equal(harness.elements.backgroundFile!.value, "");
  assert.deepEqual(harness.copied, []);
});
