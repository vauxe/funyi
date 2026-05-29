import test from "node:test";
import assert from "node:assert/strict";

import { OverlayController } from "./overlay-controller.js";
import type { OverlayMode } from "./overlay-contract.js";
import { nextTick } from "./test-async.fixture.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import {
  asDomElement,
  FakeDocument,
  FakeElement,
  installFakeDocument,
  installFakeElementConstructors,
} from "./test-dom.fixture.js";
import { createFakeOverlayHost, type FakeOverlayInvocation } from "./test-overlay-host.fixture.js";
import { pointerEvent } from "./test-pointer-event.fixture.js";
import { installFakeWindowRuntime } from "./test-window.fixture.js";

test.afterEach(() => {
  clearBrowserGlobals("document", "Element", "HTMLElement", "window");
});

test("window height switches overlay mode without a history button command", async () => {
  const harness = createHarness();
  harness.controller.bind();
  harness.windowRuntime.setInnerHeight(320);
  harness.windowRuntime.dispatch("resize", {});

  await nextTick();

  assert.deepEqual(harness.invocations, []);
  assert.equal(harness.controller.mode, "history");
  assert.equal(harness.elements.root.attributes.get("data-overlay-mode"), "history");
  assert.equal(harness.elements.root.dataset.overlayMode, "history");
  assert.deepEqual(harness.appliedModes, ["history"]);

  harness.windowRuntime.setInnerHeight(220);
  harness.windowRuntime.dispatch("resize", {});

  assert.equal(harness.controller.mode, "compact");
  assert.equal(harness.elements.root.attributes.get("data-overlay-mode"), "compact");
  assert.deepEqual(harness.appliedModes, ["history", "compact"]);
});

test("drag emits start update end commands and clears drag state", async () => {
  const harness = createHarness();
  harness.controller.bind();
  harness.document.activeElement = harness.elements.editable;

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.editable.blurred, true);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("drag ignores interactive targets inside the drag surface", async () => {
  const harness = createHarness();
  harness.controller.bind();
  const label = new FakeElement("label");
  const labelText = new FakeElement("span");
  label.append(labelText);

  harness.elements.dragSurface.dispatch(
    "pointerdown",
    pointerEvent({
      pointerId: 7,
      target: labelText,
    }),
  );
  await nextTick();

  assert.deepEqual(harness.invocations, []);
  assert.equal(harness.elements.root.className, "");
});

test("drag ignores editable history text inside the drag surface", async () => {
  const harness = createHarness();
  harness.controller.bind();
  const historyText = new FakeElement("div");
  historyText.setAttribute("contenteditable", "plaintext-only");
  harness.elements.dragSurface.append(historyText);

  harness.elements.dragSurface.dispatch(
    "pointerdown",
    pointerEvent({
      pointerId: 7,
      target: historyText,
    }),
  );
  await nextTick();

  assert.deepEqual(harness.invocations, []);
  assert.equal(harness.elements.root.className, "");
});

test("drag ignores pointerdowns inside the settings panel", async () => {
  const harness = createHarness();
  harness.controller.bind();
  const panel = new FakeElement("section", "settings-panel");
  const panelStatus = new FakeElement("p");
  panel.append(panelStatus);
  harness.elements.dragSurface.append(panel);

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7, target: panelStatus }));
  await nextTick();

  assert.deepEqual(harness.invocations, []);
  assert.equal(harness.elements.root.className, "");
});

test("compact resize delegates window size changes to native resize commands", async () => {
  const harness = createHarness();
  harness.controller.bind();

  harness.elements.resizeNorth.dispatch(
    "pointerdown",
    pointerEvent({
      currentTarget: harness.elements.resizeNorth,
      pointerId: 9,
      clientY: 100,
    }),
  );
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9, clientY: 80 }));
  harness.windowRuntime.runNextTimeout();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9, clientY: 80 }));
  await nextTick();

  assert.equal(harness.elements.root.styleValues.has("--compact-height"), false);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize"],
  );
});

function createHarness(): {
  appliedModes: OverlayMode[];
  controller: OverlayController;
  document: { activeElement: FakeElement | null };
  elements: {
    dragSurface: FakeElement;
    editable: FakeElement;
    resizeNorth: FakeElement;
    root: FakeElement;
  };
  invocations: FakeOverlayInvocation[];
  windowRuntime: ReturnType<typeof installFakeWindowRuntime>;
} {
  const documentRuntime = installFakeDocument(new FakeDocument());
  installFakeElementConstructors();
  const windowRuntime = installFakeWindowRuntime();
  const host = createFakeOverlayHost();
  const appliedModes: OverlayMode[] = [];
  const root = new FakeElement();
  const dragSurface = new FakeElement();
  const resizeNorth = new FakeElement();
  const controller = new OverlayController(
    host,
    {
      dragSurface: asDomElement(dragSurface),
      resizeHandles: [{ element: asDomElement(resizeNorth), direction: "North" }],
      root: asDomElement(root),
    },
    {
      onClearError: () => {},
      onError: (error) => {
        throw error;
      },
      onModeApplied: (mode) => appliedModes.push(mode),
    },
  );

  return {
    appliedModes,
    controller,
    document: documentRuntime,
    elements: {
      dragSurface,
      editable: new FakeElement("input"),
      resizeNorth,
      root,
    },
    invocations: host.invocations,
    windowRuntime,
  };
}
