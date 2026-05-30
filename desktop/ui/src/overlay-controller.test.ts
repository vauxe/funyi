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
import {
  createFakeOverlayHost,
  type FakeOverlayHost,
  type FakeOverlayInvocation,
} from "./test-overlay-host.fixture.js";
import { pointerEvent } from "./test-pointer-event.fixture.js";
import { installFakeWindowRuntime } from "./test-window.fixture.js";

test.afterEach(() => {
  clearBrowserGlobals("document", "Element", "HTMLElement", "window");
});

test("window height switches overlay mode without a history button command", async () => {
  const harness = createHarness();
  await bindHarness(harness);
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

test("native drag clears only after native finished event", async () => {
  const harness = createHarness();
  await bindHarness(harness);
  harness.document.activeElement = harness.elements.editable;

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 7);

  harness.host.emitOverlayDragFinished(0);
  assert.equal(harness.elements.editable.blurred, true);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("bind waits for native drag release listener before enabling overlay pointer actions", async () => {
  const host = createFakeOverlayHost();
  const listen = deferred<() => void>();
  host.listenOverlayDragFinished = () => listen.promise;
  const harness = createHarness({ host });

  harness.controller.bind();
  startDrag(harness, 7);
  startResizeNorth(harness, 9);
  await nextTick();

  assert.equal(harness.invocations.length, 0);

  listen.resolve(() => {});
  await nextTick();
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
});

test("bind keeps overlay pointer actions disabled when native drag release listener registration fails", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  host.listenOverlayDragFinished = async () => {
    throw new Error("listener failed");
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });

  await bindHarness(harness);
  startDrag(harness, 7);
  startResizeNorth(harness, 9);
  await nextTick();

  assert.equal(errors.length, 1);
  assert.deepEqual(harness.invocations, []);
  assert.equal(harness.elements.root.className, "");
});

test("native drag release event clears stale drag state", async () => {
  const harness = createHarness();
  await bindHarness(harness);

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  harness.host.emitOverlayDragFinished(0);

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("drag with no native finished event falls back to pointerup end flow", async () => {
  const host = createFakeOverlayHost();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("native drag stays active after pointerup until native finished event", async () => {
  const harness = createHarness();
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  await nextTick();
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 7);

  harness.host.emitOverlayDragFinished(0);
  await nextTick();
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayDrag"],
  );
});

test("native drag ignores move updates until native finished event", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    throw new Error("native drag should not update through fallback");
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 7 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(errors, []);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");

  host.emitOverlayDragFinished(0);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayResize"],
  );
});

test("native drag finished error reports the failure and clears drag state", async () => {
  const errors: unknown[] = [];
  const harness = createHarness({ onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.host.emitOverlayDragFinished(0, "rebound failed");
  await nextTick();

  assert.equal(errors.length, 1);
  assert.match(String(errors[0]), /rebound failed/);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("malformed native drag finished payload reports error and clears active native drag", async () => {
  const errors: unknown[] = [];
  const harness = createHarness({ onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.host.emitOverlayDragFinished(null, "overlay drag finished payload dragId must be a non-negative integer");
  await nextTick();

  assert.equal(errors.length, 1);
  assert.match(String(errors[0]), /dragId must be a non-negative integer/);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);

  startResizeNorth(harness, 9);
  await nextTick();
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayResize"],
  );
});

test("native drag release event before startOverlayDrag resolves clears once id is known", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  host.emitOverlayDragFinished(4);

  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 7);

  start.resolve(4);
  await nextTick();

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("malformed native drag finished payload before start resolves clears once native id is known", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId: 7 }));
  await nextTick();
  host.emitOverlayDragFinished(null, "overlay drag finished payload dragId must be a non-negative integer");

  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 7);

  start.resolve(4);
  await nextTick();

  assert.equal(errors.length, 1);
  assert.match(String(errors[0]), /dragId must be a non-negative integer/);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("active native drag cannot be replaced before native finished event", async () => {
  const harness = createHarness();
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  harness.host.emitOverlayDragFinished(0);

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("pending startOverlayDrag cannot be replaced by a newer drag", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  start.resolve(0);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  host.emitOverlayDragFinished(0);

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("pending startOverlayDrag falls back to pointerup end flow when native id is null", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");

  start.resolve(null);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("pending startOverlayDrag waits for native id after pointerup", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");

  start.resolve(0);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  host.emitOverlayDragFinished(0);

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("native drag ignores finished events for another drag id", async () => {
  const harness = createHarness();
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.host.emitOverlayDragFinished(99);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  harness.host.emitOverlayDragFinished(0);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayResize"],
  );
});

test("pending native drag ignores finished events for a different resolved id", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  host.emitOverlayDragFinished(99);
  await nextTick();

  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  start.resolve(0);
  await nextTick();
  host.emitOverlayDragFinished(99);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  host.emitOverlayDragFinished(0);
  await nextTick();

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("early native drag finished event is scoped to the pending drag", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const firstStart = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return firstStart.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  host.emitOverlayDragFinished(7);
  firstStart.reject(new Error("native drag start failed"));
  await nextTick();

  assert.equal(errors.length, 1);

  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return 7;
  };
  startDrag(harness, 2);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 2);

  host.emitOverlayDragFinished(7);
  await nextTick();

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("pending startOverlayDrag does not update before native start resolves", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );

  start.resolve(null);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag"],
  );

  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();
});

test("failed startOverlayDrag clears pending drag and allows a later drag", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const start = deferred<number | null>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return start.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  startDrag(harness, 1);
  await nextTick();
  start.reject(new Error("drag failed"));
  await nextTick();

  assert.equal(errors.length, 1);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);

  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return 1;
  };
  startDrag(harness, 1);
  await nextTick();
  host.emitOverlayDragFinished(1);

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("active fallback drag cannot be replaced before native drag state ends", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const update = deferred<void>();
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    return update.promise;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.runNextTimeout();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");

  update.reject(new Error("old update failed"));
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.equal(errors.length, 1);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  end.resolve();
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag", "startOverlayDrag"],
  );
});

test("stale drag move update failure cannot clear pending finishDrag", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const moveUpdate = deferred<void>();
  const end = deferred<void>();
  let updateCalls = 0;
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    updateCalls += 1;
    if (updateCalls === 1) {
      return moveUpdate.promise;
    }
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.runNextTimeout();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag"],
  );

  moveUpdate.reject(new Error("old update failed"));
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(errors, []);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 1);

  end.resolve();
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "updateOverlayDrag", "endOverlayDrag", "startOverlayResize"],
  );
});

test("finishDrag waits for an in-flight move update before ending native drag state", async () => {
  const host = createFakeOverlayHost();
  const moveUpdate = deferred<void>();
  const finalUpdate = deferred<void>();
  const end = deferred<void>();
  let updateCalls = 0;
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    updateCalls += 1;
    return updateCalls === 1 ? moveUpdate.promise : finalUpdate.promise;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.runNextTimeout();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag"],
  );

  moveUpdate.resolve();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "updateOverlayDrag"],
  );

  finalUpdate.resolve();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );

  end.resolve();
  await nextTick();
});

test("active drag update failure still ends native drag state", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    throw new Error("drag update failed");
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.equal(errors.length, 1);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("final drag update and end failures are reported together", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    throw new Error("final drag update failed");
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    throw new Error("final drag end failed");
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();

  assert.equal(errors.length, 1);
  assert.match(String(errors[0]), /final drag update failed; final drag end failed/);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("pending finishDrag blocks replacing it with a newer drag", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );

  end.resolve();
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag", "startOverlayDrag"],
  );
});

test("pending finishDrag with lost capture blocks starting resize", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  harness.elements.dragSurface.pointerCapture = null;
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );

  end.resolve();
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag", "startOverlayResize"],
  );
});

test("failed pending finishDrag clears the session and allows a later drag", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );

  end.reject(new Error("drag finish failed"));
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.equal(errors.length, 1);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag", "startOverlayDrag"],
  );
});

test("duplicate drag end events emit one native finish flow", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 1);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );

  end.resolve();
  await nextTick();

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.dragSurface.pointerCapture, null);
});

test("active resize cannot be replaced while startOverlayResize is pending", async () => {
  const host = createFakeOverlayHost();
  const firstStart = deferred<void>();
  host.startOverlayResize = async (direction) => {
    host.invocations.push({ method: "startOverlayResize", args: { direction } });
    return firstStart.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(harness.invocations, [{ method: "startOverlayResize", args: { direction: "North" } }]);
  assert.equal(harness.elements.root.className, "is-resizing");
  assert.equal(harness.elements.resizeNorth.pointerCapture, 9);

  firstStart.resolve();
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.resizeNorth.pointerCapture, null);
});

test("pending startOverlayResize waits to end until native start resolves", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<void>();
  host.startOverlayResize = async (direction) => {
    host.invocations.push({ method: "startOverlayResize", args: { direction } });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();

  assert.deepEqual(harness.invocations, [{ method: "startOverlayResize", args: { direction: "North" } }]);
  assert.equal(harness.elements.root.className, "is-resizing");

  start.resolve();
  await nextTick();

  assert.deepEqual(harness.invocations, [
    { method: "startOverlayResize", args: { direction: "North" } },
    { method: "endOverlayResize", args: undefined },
  ]);
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.resizeNorth.pointerCapture, null);
});

test("pending startOverlayResize does not update before native start resolves", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<void>();
  host.startOverlayResize = async (direction) => {
    host.invocations.push({ method: "startOverlayResize", args: { direction } });
    return start.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize"],
  );

  start.resolve();
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize"],
  );

  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();
});

test("pending startOverlayResize with lost capture blocks starting drag", async () => {
  const host = createFakeOverlayHost();
  const start = deferred<void>();
  const end = deferred<void>();
  host.startOverlayResize = async (direction) => {
    host.invocations.push({ method: "startOverlayResize", args: { direction } });
    return start.promise;
  };
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.elements.resizeNorth.pointerCapture = null;
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(harness.invocations, [{ method: "startOverlayResize", args: { direction: "North" } }]);
  assert.equal(harness.elements.root.className, "is-resizing");

  start.resolve();
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(harness.invocations, [
    { method: "startOverlayResize", args: { direction: "North" } },
    { method: "endOverlayResize" },
  ]);

  end.resolve();
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(harness.invocations, [
    { method: "startOverlayResize", args: { direction: "North" } },
    { method: "endOverlayResize" },
    { method: "startOverlayDrag", args: undefined },
  ]);
});

test("pending finishResize blocks replacing it with a newer resize", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize", "startOverlayResize"],
  );
});

test("failed pending finishResize clears the session and allows a later resize", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.reject(new Error("resize finish failed"));
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.equal(errors.length, 1);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize", "startOverlayResize"],
  );
});

test("pending finishResize with lost capture blocks starting drag", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  harness.elements.resizeNorth.pointerCapture = null;
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize", "startOverlayDrag"],
  );
});

test("duplicate resize end events emit one native finish flow", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();

  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.resizeNorth.pointerCapture, null);
});

test("stale fallback drag capture ends native drag state before starting resize", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.elements.dragSurface.pointerCapture = null;
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "endOverlayDrag"],
  );

  end.resolve();
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "endOverlayDrag", "startOverlayResize"],
  );
  assert.equal(harness.elements.root.className, "is-resizing");
  assert.equal(harness.elements.resizeNorth.pointerCapture, 9);
});

test("stale fallback drag capture waits for in-flight update before ending native drag state", async () => {
  const host = createFakeOverlayHost();
  const moveUpdate = deferred<void>();
  const end = deferred<void>();
  host.startOverlayDrag = async () => {
    host.invocations.push({ method: "startOverlayDrag" });
    return null;
  };
  host.updateOverlayDrag = async () => {
    host.invocations.push({ method: "updateOverlayDrag" });
    return moveUpdate.promise;
  };
  host.endOverlayDrag = async () => {
    host.invocations.push({ method: "endOverlayDrag" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 7 }));
  harness.windowRuntime.runNextTimeout();
  harness.elements.dragSurface.pointerCapture = null;
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag"],
  );

  moveUpdate.resolve();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag"],
  );

  end.resolve();
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "updateOverlayDrag", "endOverlayDrag", "startOverlayResize"],
  );
});

test("active native drag with lost capture blocks starting resize", async () => {
  const harness = createHarness();
  await bindHarness(harness);

  startDrag(harness, 7);
  await nextTick();
  harness.elements.dragSurface.pointerCapture = null;
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");

  harness.host.emitOverlayDragFinished(0);
  await nextTick();
  startResizeNorth(harness, 9);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayDrag", "startOverlayResize"],
  );
});

test("stale resize capture ends native resize state before starting another resize", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.elements.resizeNorth.pointerCapture = null;
  startResizeNorth(harness, 10);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
  startResizeNorth(harness, 10);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize", "startOverlayResize"],
  );
  assert.equal(harness.elements.root.className, "is-resizing");
  assert.equal(harness.elements.resizeNorth.pointerCapture, 10);
});

test("stale resize capture ends native resize state before starting drag", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.elements.resizeNorth.pointerCapture = null;
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize", "startOverlayDrag"],
  );
  assert.equal(harness.elements.root.className, "is-dragging");
  assert.equal(harness.elements.dragSurface.pointerCapture, 7);
});

test("stale resize capture waits for in-flight update before ending native resize state", async () => {
  const host = createFakeOverlayHost();
  const moveUpdate = deferred<void>();
  const end = deferred<void>();
  host.updateOverlayResize = async () => {
    host.invocations.push({ method: "updateOverlayResize" });
    return moveUpdate.promise;
  };
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  harness.elements.resizeNorth.pointerCapture = null;
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize"],
  );

  moveUpdate.resolve();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
  startDrag(harness, 7);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize", "startOverlayDrag"],
  );
});

test("resize move after pointerup does not update while finish is pending", async () => {
  const host = createFakeOverlayHost();
  const end = deferred<void>();
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
});

test("stale resize move update failure cannot clear pending finishResize", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  const moveUpdate = deferred<void>();
  const end = deferred<void>();
  let updateCalls = 0;
  host.updateOverlayResize = async () => {
    host.invocations.push({ method: "updateOverlayResize" });
    updateCalls += 1;
    if (updateCalls === 1) {
      return moveUpdate.promise;
    }
  };
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize"],
  );

  moveUpdate.reject(new Error("old resize update failed"));
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(errors, []);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize"],
  );
  assert.equal(harness.elements.root.className, "is-resizing");
  assert.equal(harness.elements.resizeNorth.pointerCapture, 9);

  end.resolve();
  await nextTick();
  startDrag(harness, 1);
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize", "startOverlayDrag"],
  );
});

test("finishResize waits for an in-flight move update before ending native resize state", async () => {
  const host = createFakeOverlayHost();
  const moveUpdate = deferred<void>();
  const end = deferred<void>();
  host.updateOverlayResize = async () => {
    host.invocations.push({ method: "updateOverlayResize" });
    return moveUpdate.promise;
  };
  host.endOverlayResize = async () => {
    host.invocations.push({ method: "endOverlayResize" });
    return end.promise;
  };
  const harness = createHarness({ host });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  harness.windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 9 }));
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize"],
  );

  moveUpdate.resolve();
  await nextTick();

  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize"],
  );

  end.resolve();
  await nextTick();
});

test("active resize update failure still ends native resize state", async () => {
  const host = createFakeOverlayHost();
  const errors: unknown[] = [];
  host.updateOverlayResize = async () => {
    host.invocations.push({ method: "updateOverlayResize" });
    throw new Error("resize update failed");
  };
  const harness = createHarness({ host, onError: (error) => errors.push(error) });
  await bindHarness(harness);

  startResizeNorth(harness, 9);
  await nextTick();
  harness.windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 9 }));
  harness.windowRuntime.runNextTimeout();
  await nextTick();

  assert.equal(errors.length, 1);
  assert.deepEqual(
    harness.invocations.map((item) => item.method),
    ["startOverlayResize", "updateOverlayResize", "endOverlayResize"],
  );
  assert.equal(harness.elements.root.className, "");
  assert.equal(harness.elements.resizeNorth.pointerCapture, null);
});

test("drag ignores interactive targets inside the drag surface", async () => {
  const harness = createHarness();
  await bindHarness(harness);
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
  await bindHarness(harness);
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
  await bindHarness(harness);
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
  await bindHarness(harness);

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
  assert.deepEqual(harness.invocations, [
    { method: "startOverlayResize", args: { direction: "North" } },
    { method: "updateOverlayResize", args: undefined },
    { method: "endOverlayResize", args: undefined },
  ]);
});

interface OverlayHarness {
  appliedModes: OverlayMode[];
  controller: OverlayController;
  document: { activeElement: FakeElement | null };
  elements: {
    dragSurface: FakeElement;
    editable: FakeElement;
    resizeNorth: FakeElement;
    root: FakeElement;
  };
  host: FakeOverlayHost;
  invocations: FakeOverlayInvocation[];
  windowRuntime: ReturnType<typeof installFakeWindowRuntime>;
}

function createHarness({
  host = createFakeOverlayHost(),
  onError,
}: {
  host?: FakeOverlayHost;
  onError?: (error: unknown) => void;
} = {}): OverlayHarness {
  const documentRuntime = installFakeDocument(new FakeDocument());
  installFakeElementConstructors();
  const windowRuntime = installFakeWindowRuntime();
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
      onError:
        onError ??
        ((error) => {
          throw error;
        }),
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
    host,
    invocations: host.invocations,
    windowRuntime,
  };
}

function startDrag(harness: OverlayHarness, pointerId: number): void {
  harness.elements.dragSurface.dispatch("pointerdown", pointerEvent({ pointerId }));
}

async function bindHarness(harness: OverlayHarness): Promise<void> {
  harness.controller.bind();
  await nextTick();
}

function startResizeNorth(harness: OverlayHarness, pointerId: number): void {
  harness.elements.resizeNorth.dispatch(
    "pointerdown",
    pointerEvent({
      currentTarget: harness.elements.resizeNorth,
      pointerId,
    }),
  );
}

function deferred<T>(): {
  promise: Promise<T>;
  reject(error: unknown): void;
  resolve(value: T): void;
} {
  let reject!: (error: unknown) => void;
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}
