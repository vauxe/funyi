interface PointerSessionCallbacks {
  onStaleSession?(token: number): void;
  onMove(event: PointerEvent): void;
  onEnd(event: PointerEvent, token: number): void;
}

interface ActivePointer {
  ending: boolean;
  pointerId: number;
  surface: HTMLElement;
  token: number;
}

export class PointerSession {
  private activePointer: ActivePointer | null = null;
  private nextToken = 0;

  private readonly handleMove = (event: PointerEvent): void => {
    if (this.matches(event)) {
      this.callbacks.onMove(event);
    }
  };

  private readonly handleEnd = (event: PointerEvent): void => {
    const activePointer = this.activePointer;
    if (event.pointerId === activePointer?.pointerId && !activePointer.ending) {
      activePointer.ending = true;
      window.removeEventListener("pointermove", this.handleMove);
      this.callbacks.onEnd(event, activePointer.token);
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

  get activeToken(): number | null {
    return this.activePointer?.token ?? null;
  }

  start(event: PointerEvent, surface: HTMLElement): boolean {
    if (this.activePointer) {
      if (this.activePointer.ending) {
        return false;
      }
      if (this.activePointer.surface.hasPointerCapture(this.activePointer.pointerId)) {
        return false;
      }
      const staleToken = this.activePointer.token;
      this.callbacks.onStaleSession?.(staleToken);
      this.clear(staleToken);
    }

    this.activePointer = { ending: false, pointerId: event.pointerId, surface, token: this.nextToken };
    this.nextToken += 1;
    surface.setPointerCapture(event.pointerId);
    this.root.classList.add(this.activeClassName);
    window.addEventListener("pointermove", this.handleMove);
    window.addEventListener("pointerup", this.handleEnd);
    window.addEventListener("pointercancel", this.handleEnd);
    return true;
  }

  clearIfCaptureLost(): boolean {
    const activePointer = this.activePointer;
    if (!activePointer || activePointer.ending || activePointer.surface.hasPointerCapture(activePointer.pointerId)) {
      return false;
    }
    const staleToken = activePointer.token;
    this.callbacks.onStaleSession?.(staleToken);
    this.clear(staleToken);
    return true;
  }

  clear(token?: number | null): void {
    const activePointer = this.activePointer;
    if (!activePointer) {
      return;
    }
    if (token != null && activePointer.token !== token) {
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
