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
