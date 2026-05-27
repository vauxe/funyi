import type { FakeElement } from "./test-dom.fixture.js";

interface FakePointerEventOptions {
  button?: number;
  clientY?: number;
  currentTarget?: FakeElement | null;
  pointerId: number;
  target?: FakeElement | null;
}

export function pointerEvent({
  button = 0,
  clientY = 0,
  currentTarget = null,
  pointerId,
  target = null,
}: FakePointerEventOptions): PointerEvent {
  return {
    button,
    clientY,
    currentTarget,
    pointerId,
    preventDefault: () => {},
    stopPropagation: () => {},
    target,
  } as unknown as PointerEvent;
}
