import test from "node:test";
import assert from "node:assert/strict";

import { SessionControlsView } from "./session-controls-view.js";
import { asDomElement, FakeElement } from "./test-dom.fixture.js";

test("renders session control state for active and finishing sessions", () => {
  const elements = createElements();
  const view = new SessionControlsView(sessionControlsElements(elements));

  view.renderState("running", { canStart: true });

  assert.equal(elements.appShell.attributes.get("data-state"), "running");
  assert.equal(elements.sessionButton.className, "is-stop");
  assert.equal(elements.sessionButton.title, "Stop");
  assert.equal(elements.serverUrl.disabled, true);
  assert.equal(elements.language.disabled, false);
  assert.equal(elements.translationTargetLanguage.disabled, false);
  assert.equal(elements.audioSource.disabled, true);

  view.renderState("finishing", { canStart: true });

  assert.equal(elements.sessionButton.className, "is-stop is-finishing");
  assert.equal(elements.sessionButton.title, "Cancel final transcript");
  assert.equal(elements.language.disabled, true);
  assert.equal(elements.translationTargetLanguage.disabled, true);
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

function createElements(): Record<
  | "appShell"
  | "audioSource"
  | "language"
  | "serverUrl"
  | "sessionButton"
  | "sessionStatus"
  | "translationTargetLanguage"
  | "volumeIndicator",
  FakeElement
> {
  return {
    appShell: new FakeElement(),
    audioSource: new FakeElement(),
    language: new FakeElement(),
    serverUrl: new FakeElement(),
    sessionButton: new FakeElement(),
    sessionStatus: new FakeElement(),
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
    sessionButton: asDomElement<HTMLButtonElement>(elements.sessionButton),
    sessionStatus: asDomElement(elements.sessionStatus),
    translationTargetLanguage: asDomElement<HTMLSelectElement>(elements.translationTargetLanguage),
    volumeIndicator: asDomElement(elements.volumeIndicator),
  };
}
