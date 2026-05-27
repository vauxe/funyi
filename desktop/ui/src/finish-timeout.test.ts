import test from "node:test";
import assert from "node:assert/strict";

import { FinishTimeout, type Clock } from "./finish-timeout.js";

test("schedules and clears the finish timeout", () => {
  const clock = fakeClock();
  const timeout = new FinishTimeout(clock, 50);

  timeout.schedule(() => {
    clock.fired += 1;
  });

  assert.equal(clock.scheduled?.delay, 50);
  timeout.clear();
  clock.scheduled?.callback();

  assert.equal(clock.scheduled, null);
  assert.equal(clock.fired, 0);
});

test("rescheduling replaces the previous timeout", () => {
  const clock = fakeClock();
  const timeout = new FinishTimeout(clock, 50);

  timeout.schedule(() => {
    clock.fired += 1;
  });
  const first = clock.scheduled;
  timeout.schedule(() => {
    clock.fired += 10;
  });

  first?.callback();
  clock.scheduled?.callback();

  assert.equal(clock.fired, 10);
});

function fakeClock(): Clock & {
  fired: number;
  scheduled: { callback: () => void | Promise<void>; delay: number; id: symbol } | null;
} {
  return {
    fired: 0,
    scheduled: null,
    clearTimeout(id: unknown): void {
      if (this.scheduled?.id === id) {
        this.scheduled = null;
      }
    },
    setTimeout(callback: () => void | Promise<void>, delay: number): symbol {
      const id = Symbol("timeout");
      const scheduledCallback = (): void | Promise<void> => {
        if (this.scheduled?.id !== id) {
          return undefined;
        }
        return callback();
      };
      this.scheduled = { callback: scheduledCallback, delay, id };
      return id;
    },
  };
}
