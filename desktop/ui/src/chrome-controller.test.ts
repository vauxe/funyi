import test from "node:test";
import assert from "node:assert/strict";

import { ChromeController } from "./chrome-controller.js";
import type { Clock } from "./finish-timeout.js";
import { asDomElement, FakeElement } from "./test-dom.fixture.js";

class FakeClock implements Clock {
  private callbacks = new Map<number, () => void>();
  private nextId = 1;

  setTimeout(callback: () => void | Promise<void>): unknown {
    const id = this.nextId++;
    this.callbacks.set(id, callback as () => void);
    return id;
  }

  clearTimeout(handle: unknown): void {
    this.callbacks.delete(handle as number);
  }

  get pending(): number {
    return this.callbacks.size;
  }

  // Fire every timer currently scheduled (a re-armed timer is left pending).
  flush(): void {
    const callbacks = [...this.callbacks.values()];
    this.callbacks.clear();
    for (const callback of callbacks) {
      callback();
    }
  }
}

interface Harness {
  controller: ChromeController;
  root: FakeElement;
  clock: FakeClock;
  setStayVisible(value: boolean): void;
}

function createHarness(): Harness {
  const root = new FakeElement("main", "app-shell");
  const clock = new FakeClock();
  let stayVisible = false;
  const controller = new ChromeController({
    root: asDomElement(root),
    clock,
    shouldStayVisible: () => stayVisible,
  });
  controller.init();
  return {
    controller,
    root,
    clock,
    setStayVisible: (value) => {
      stayVisible = value;
    },
  };
}

function chrome(root: FakeElement): string | undefined {
  return root.dataset.chrome;
}

test("starts visible and arms the idle countdown from launch", () => {
  const { root, clock } = createHarness();
  assert.equal(chrome(root), "visible");
  assert.equal(clock.pending, 1);
});

test("auto-hides after inactivity even with no session running", () => {
  const { root, clock } = createHarness();
  clock.flush();
  assert.equal(chrome(root), "hidden");
});

test("activity reveals chrome and restarts the countdown", () => {
  const { root, clock } = createHarness();
  clock.flush();
  assert.equal(chrome(root), "hidden");

  root.dispatch("pointermove", {});
  assert.equal(chrome(root), "visible");
  assert.equal(clock.pending, 1, "re-armed after activity");

  clock.flush();
  assert.equal(chrome(root), "hidden");
});

test("keyboard activity reveals chrome while hidden", () => {
  const { root, clock } = createHarness();
  clock.flush();
  assert.equal(chrome(root), "hidden");

  root.dispatch("keydown", {});
  assert.equal(chrome(root), "visible");
});

test("a session-state transition reveals chrome, but a repeat does not", () => {
  const { root, clock, controller } = createHarness();
  clock.flush();
  assert.equal(chrome(root), "hidden");

  controller.setSessionState("connecting");
  assert.equal(chrome(root), "visible", "a real transition surfaces the controls");

  clock.flush();
  assert.equal(chrome(root), "hidden");

  // A duplicate notification (e.g. audio availability nudging onStateChange with
  // the same state) must not keep flashing the controls back on.
  controller.setSessionState("connecting");
  assert.equal(chrome(root), "hidden");
  assert.equal(clock.pending, 0, "no fresh countdown for a non-transition");
});

test("stays visible while pinned, then hides once released", () => {
  const { root, clock, setStayVisible } = createHarness();
  setStayVisible(true);

  clock.flush();
  assert.equal(chrome(root), "visible", "pinned open keeps chrome visible");
  assert.equal(clock.pending, 1, "re-armed to re-check later");

  setStayVisible(false);
  clock.flush();
  assert.equal(chrome(root), "hidden");
});
