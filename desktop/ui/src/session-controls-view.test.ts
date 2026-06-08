import test from "node:test";
import assert from "node:assert/strict";

import { SessionControlsView } from "./session-controls-view.js";
import { asDomElement, FakeElement } from "./test-dom.fixture.js";

test("renders session control state for active and finishing sessions", () => {
  const elements = createElements();
  const view = new SessionControlsView(sessionControlsElements(elements));

  view.renderState("idle", { canStart: true });

  assert.equal(elements.transportButton.disabled, false);
  assert.equal(elements.transportButton.className, "");
  assert.equal(elements.transportButton.title, "Start");
  assert.equal(elements.stopButton.disabled, true);
  assert.equal(elements.stopButton.className, "");
  assert.equal(elements.stopButton.title, "Stop");

  view.renderState("connecting", { canStart: true });

  assert.equal(elements.transportButton.disabled, true);
  assert.equal(elements.transportButton.title, "Starting");
  assert.equal(elements.stopButton.disabled, false);
  assert.equal(elements.stopButton.className, "is-cancel");
  assert.equal(elements.stopButton.title, "Cancel start");

  view.renderState("running", { canStart: true });

  assert.equal(elements.appShell.attributes.get("data-state"), "running");
  assert.equal(elements.transportButton.disabled, false);
  assert.equal(elements.transportButton.className, "is-pause");
  assert.equal(elements.transportButton.title, "Pause");
  assert.equal(elements.stopButton.disabled, false);
  assert.equal(elements.stopButton.className, "");
  assert.equal(elements.stopButton.title, "Stop");
  assert.equal(elements.serverUrl.disabled, true);
  assert.equal(elements.language.disabled, false);
  assert.equal(elements.translationTargetLanguage.disabled, false);
  assert.equal(elements.audioSource.disabled, false);

  view.renderState("paused", { canStart: true });

  assert.equal(elements.appShell.attributes.get("data-state"), "paused");
  assert.equal(elements.transportButton.disabled, false);
  assert.equal(elements.transportButton.className, "");
  assert.equal(elements.transportButton.title, "Resume");
  assert.equal(elements.stopButton.disabled, false);
  assert.equal(elements.stopButton.className, "");
  assert.equal(elements.stopButton.title, "Stop");
  assert.equal(elements.serverUrl.disabled, true);
  assert.equal(elements.language.disabled, true);
  assert.equal(elements.translationTargetLanguage.disabled, true);
  assert.equal(elements.audioSource.disabled, false);

  view.renderState("finishing", { canStart: true });

  assert.equal(elements.transportButton.disabled, true);
  assert.equal(elements.transportButton.className, "");
  assert.equal(elements.transportButton.title, "Finalizing");
  assert.equal(elements.stopButton.disabled, false);
  assert.equal(elements.stopButton.className, "is-cancel");
  assert.equal(elements.stopButton.title, "Cancel final transcript");
  assert.equal(elements.language.disabled, true);
  assert.equal(elements.translationTargetLanguage.disabled, true);
  assert.equal(elements.audioSource.disabled, true);
});

test("renders status summary datasets and clears stale level", () => {
  const elements = createElements();
  const view = new SessionControlsView(sessionControlsElements(elements));

  view.renderStatus({ text: "Audio lagging", tone: "warn", level: "low", volume: 0.53 });
  view.renderStatus({ text: "", tone: "idle" });

  assert.equal(elements.appShell.dataset.statusActive, "false");
  assert.equal(elements.sessionStatus.textContent, "");
  assert.equal(elements.sessionStatus.dataset.active, "false");
  assert.equal(elements.sessionStatus.dataset.tone, "idle");
  assert.equal("level" in elements.sessionStatus.dataset, false);
  assert.equal(elements.sessionStatus.attributes.get("aria-label"), "");
  assert.equal(elements.volumeIndicator.dataset.level, "silent");
  assert.equal(elements.volumeIndicator.styleValues.get("--volume-bar-low"), "0.18");
  assert.equal(elements.volumeIndicator.styleValues.get("--volume-bar-mid"), "0.12");
  assert.equal(elements.volumeIndicator.styleValues.get("--volume-bar-high"), "0.08");
});

test("renders volume indicator from audio level summaries", () => {
  const elements = createElements();
  const view = new SessionControlsView(sessionControlsElements(elements));

  view.renderStatus({ text: "", tone: "idle", level: "live", volume: 0.75 });

  assert.equal(elements.volumeIndicator.dataset.level, "live");
  assert.equal(elements.volumeIndicator.styleValues.get("--volume-bar-low"), "0.49");
  assert.equal(elements.volumeIndicator.styleValues.get("--volume-bar-mid"), "0.66");
  assert.equal(elements.volumeIndicator.styleValues.get("--volume-bar-high"), "0.77");
});

test("does not rewrite unchanged status and volume DOM state", () => {
  const elements = createElements();
  const view = new SessionControlsView(sessionControlsElements(elements));

  const summary = { text: "Audio lagging", tone: "warn" as const, level: "live" as const, volume: 0.75 };
  view.renderStatus(summary);
  const writes = trackDomWrites(elements.appShell, elements.sessionStatus, elements.volumeIndicator);
  view.renderStatus(summary);

  assert.deepEqual(writes, []);
});

function createElements(): Record<
  | "appShell"
  | "audioSource"
  | "language"
  | "serverUrl"
  | "sessionStatus"
  | "stopButton"
  | "transportButton"
  | "translationTargetLanguage"
  | "volumeIndicator",
  FakeElement
> {
  return {
    appShell: new FakeElement(),
    audioSource: new FakeElement(),
    language: new FakeElement(),
    serverUrl: new FakeElement(),
    sessionStatus: new FakeElement(),
    stopButton: new FakeElement(),
    transportButton: new FakeElement(),
    translationTargetLanguage: new FakeElement(),
    volumeIndicator: new FakeElement(),
  };
}

function sessionControlsElements(
  elements: ReturnType<typeof createElements>,
): ConstructorParameters<typeof SessionControlsView>[0] {
  return {
    appShell: asDomElement(elements.appShell),
    audioSource: asDomElement<HTMLSelectElement>(elements.audioSource),
    language: asDomElement<HTMLSelectElement>(elements.language),
    serverUrl: asDomElement<HTMLInputElement>(elements.serverUrl),
    sessionStatus: asDomElement(elements.sessionStatus),
    stopButton: asDomElement<HTMLButtonElement>(elements.stopButton),
    transportButton: asDomElement<HTMLButtonElement>(elements.transportButton),
    translationTargetLanguage: asDomElement<HTMLSelectElement>(elements.translationTargetLanguage),
    volumeIndicator: asDomElement(elements.volumeIndicator),
  };
}

function trackDomWrites(...elements: FakeElement[]): string[] {
  const writes: string[] = [];
  for (const element of elements) {
    trackPropertyWrites(element, "textContent", writes);
    trackPropertyWrites(element, "title", writes);
    trackDatasetWrites(element, writes);
    trackAttributeWrites(element, writes);
    trackStyleWrites(element, writes);
  }
  return writes;
}

function trackPropertyWrites(element: FakeElement, property: "textContent" | "title", writes: string[]): void {
  let value = element[property];
  Object.defineProperty(element, property, {
    configurable: true,
    get: () => value,
    set: (next: string) => {
      writes.push(`${property}=${next}`);
      value = next;
    },
  });
}

function trackDatasetWrites(element: FakeElement, writes: string[]): void {
  element.dataset = new Proxy(element.dataset, {
    deleteProperty(target, property) {
      writes.push(`delete data-${String(property)}`);
      return Reflect.deleteProperty(target, property);
    },
    set(target, property, value) {
      writes.push(`data-${String(property)}=${String(value)}`);
      return Reflect.set(target, property, String(value));
    },
  });
}

function trackAttributeWrites(element: FakeElement, writes: string[]): void {
  const setAttribute = element.setAttribute.bind(element);
  element.setAttribute = (name: string, value: string): void => {
    writes.push(`${name}=${value}`);
    setAttribute(name, value);
  };
}

function trackStyleWrites(element: FakeElement, writes: string[]): void {
  const setProperty = element.style.setProperty;
  element.style.setProperty = (name: string, value: string): void => {
    writes.push(`${name}=${value}`);
    setProperty(name, value);
  };
}
