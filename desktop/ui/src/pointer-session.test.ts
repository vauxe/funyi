import test from "node:test";
import assert from "node:assert/strict";

import { FrameScheduler, PointerSession } from "./pointer-session.js";
import { clearBrowserGlobals } from "./test-browser-globals.fixture.js";
import { asDomElement, FakeElement } from "./test-dom.fixture.js";
import { pointerEvent } from "./test-pointer-event.fixture.js";
import { installFakeWindowRuntime } from "./test-window.fixture.js";

test.afterEach(() => {
  clearBrowserGlobals("window");
});

test("pointer session captures one pointer and clears listeners on end", () => {
  const windowRuntime = installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const moves: number[] = [];
  const ends: number[] = [];
  const session = new PointerSession(asDomElement(root), "is-active", {
    onMove: (event) => moves.push(event.pointerId),
    onEnd: (event) => {
      ends.push(event.pointerId);
      session.clear();
    },
  });

  assert.equal(session.start(pointerEvent({ pointerId: 7 }), asDomElement(surface)), true);
  assert.equal(session.start(pointerEvent({ pointerId: 8 }), asDomElement(surface)), false);

  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 8 }));
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 7 }));
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 7 }));
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 7 }));

  assert.deepEqual(moves, [7]);
  assert.deepEqual(ends, [7]);
  assert.equal(root.className, "");
  assert.equal(surface.pointerCapture, null);
});

test("pointer session recovers when native dragging drops pointer capture", () => {
  installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const canceled: number[] = [];
  const session = new PointerSession(asDomElement(root), "is-active", {
    onStaleSession: (token) => canceled.push(token),
    onMove: () => {},
    onEnd: () => {},
  });

  assert.equal(session.start(pointerEvent({ pointerId: 7 }), asDomElement(surface)), true);
  surface.pointerCapture = null;

  assert.equal(session.start(pointerEvent({ pointerId: 8 }), asDomElement(surface)), true);
  assert.deepEqual(canceled, [0]);
  assert.equal(root.className, "is-active");
  assert.equal(surface.pointerCapture, 8);
});

test("pointer session does not replace a captured pointer with the same pointer id", () => {
  const windowRuntime = installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const canceled: number[] = [];
  const moves: number[] = [];
  const ends: number[] = [];
  const session = new PointerSession(asDomElement(root), "is-active", {
    onStaleSession: (token) => canceled.push(token),
    onMove: (event) => moves.push(event.pointerId),
    onEnd: (event) => {
      ends.push(event.pointerId);
      session.clear();
    },
  });

  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), true);
  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), false);
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));

  assert.deepEqual(canceled, []);
  assert.deepEqual(moves, [1]);
  assert.deepEqual(ends, [1]);
  assert.equal(root.className, "");
  assert.equal(surface.pointerCapture, null);
});

test("pointer session stops move handling after the first end event", () => {
  const windowRuntime = installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const moves: number[] = [];
  const ends: number[] = [];
  const session = new PointerSession(asDomElement(root), "is-active", {
    onMove: (event) => moves.push(event.pointerId),
    onEnd: (event) => {
      ends.push(event.pointerId);
    },
  });

  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), true);
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));

  assert.deepEqual(moves, [1]);
  assert.deepEqual(ends, [1]);
  assert.equal(root.className, "is-active");
  assert.equal(surface.pointerCapture, 1);
});

test("pointer session treats pointercancel as the end event", () => {
  const windowRuntime = installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const moves: number[] = [];
  const ends: number[] = [];
  const session = new PointerSession(asDomElement(root), "is-active", {
    onMove: (event) => moves.push(event.pointerId),
    onEnd: (event, token) => {
      ends.push(event.pointerId);
      session.clear(token);
    },
  });

  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), true);
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointercancel", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointermove", pointerEvent({ pointerId: 1 }));
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));

  assert.deepEqual(moves, [1]);
  assert.deepEqual(ends, [1]);
  assert.equal(root.className, "");
  assert.equal(surface.pointerCapture, null);
});

test("pointer session does not treat an ending pointer as stale capture", () => {
  const windowRuntime = installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const canceled: number[] = [];
  const session = new PointerSession(asDomElement(root), "is-active", {
    onStaleSession: (token) => canceled.push(token),
    onMove: () => {},
    onEnd: () => {},
  });

  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), true);
  windowRuntime.dispatch("pointerup", pointerEvent({ pointerId: 1 }));
  surface.pointerCapture = null;

  assert.equal(session.clearIfCaptureLost(), false);
  assert.deepEqual(canceled, []);
  assert.equal(session.activeToken, 0);
  assert.equal(root.className, "is-active");
});

test("pointer session ignores stale async clear for a newer session", () => {
  installFakeWindowRuntime();
  const root = new FakeElement();
  const surface = new FakeElement();
  const session = new PointerSession(asDomElement(root), "is-active", {
    onMove: () => {},
    onEnd: () => {},
  });

  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), true);
  const firstToken = session.activeToken;
  surface.pointerCapture = null;
  assert.equal(session.start(pointerEvent({ pointerId: 1 }), asDomElement(surface)), true);
  const secondToken = session.activeToken;

  session.clear(firstToken);

  assert.equal(root.className, "is-active");
  assert.equal(surface.pointerCapture, 1);

  session.clear(secondToken);

  assert.equal(root.className, "");
  assert.equal(surface.pointerCapture, null);
});

test("frame scheduler coalesces pending work and can cancel it", () => {
  const windowRuntime = installFakeWindowRuntime();
  let runs = 0;
  const scheduler = new FrameScheduler(() => {
    runs += 1;
  });

  scheduler.schedule();
  scheduler.schedule();
  assert.equal(windowRuntime.pendingTimeouts.size, 1);

  scheduler.cancel();
  assert.equal(windowRuntime.pendingTimeouts.size, 0);

  scheduler.schedule();
  windowRuntime.runNextTimeout();
  assert.equal(runs, 1);
});
