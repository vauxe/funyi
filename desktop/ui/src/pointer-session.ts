interface PointerSessionCallbacks {
  onMove(event: PointerEvent): void;
  onEnd(event: PointerEvent): void;
}

interface ActivePointer {
  pointerId: number;
  surface: HTMLElement;
}

export class PointerSession {
  private activePointer: ActivePointer | null = null;

  private readonly handleMove = (event: PointerEvent): void => {
    if (this.matches(event)) {
      this.callbacks.onMove(event);
    }
  };

  private readonly handleEnd = (event: PointerEvent): void => {
    if (this.matches(event)) {
      this.callbacks.onEnd(event);
    }
  };

  constructor(
    private readonly root: HTMLElement,
    private readonly activeClassName: string,
    private readonly callbacks: PointerSessionCallbacks,
  ) {}

  get isActive(): boolean {
    return this.activePointer !== null;
  }

  start(event: PointerEvent, surface: HTMLElement): boolean {
    if (this.activePointer) {
      return false;
    }

    this.activePointer = { pointerId: event.pointerId, surface };
    surface.setPointerCapture(event.pointerId);
    this.root.classList.add(this.activeClassName);
    window.addEventListener("pointermove", this.handleMove);
    window.addEventListener("pointerup", this.handleEnd);
    window.addEventListener("pointercancel", this.handleEnd);
    return true;
  }

  clear(): void {
    const activePointer = this.activePointer;
    if (!activePointer) {
      return;
    }

    if (activePointer.surface.hasPointerCapture(activePointer.pointerId)) {
      activePointer.surface.releasePointerCapture(activePointer.pointerId);
    }
    this.activePointer = null;
    this.root.classList.remove(this.activeClassName);
    window.removeEventListener("pointermove", this.handleMove);
    window.removeEventListener("pointerup", this.handleEnd);
    window.removeEventListener("pointercancel", this.handleEnd);
  }

  private matches(event: PointerEvent): boolean {
    return event.pointerId === this.activePointer?.pointerId;
  }
}

export class FrameScheduler {
  private pendingFrame: { cancel(): void } | null = null;

  constructor(private readonly callback: () => void) {}

  schedule(): void {
    if (this.pendingFrame !== null) {
      return;
    }

    const run = (): void => {
      this.pendingFrame = null;
      this.callback();
    };

    if (typeof requestAnimationFrame === "function" && typeof cancelAnimationFrame === "function") {
      const frame = requestAnimationFrame(run);
      this.pendingFrame = { cancel: () => cancelAnimationFrame(frame) };
      return;
    }

    const timeout = window.setTimeout(run, 16);
    this.pendingFrame = { cancel: () => window.clearTimeout(timeout) };
  }

  cancel(): void {
    if (this.pendingFrame === null) {
      return;
    }
    this.pendingFrame.cancel();
    this.pendingFrame = null;
  }
}
