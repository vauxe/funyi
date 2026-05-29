import type { OverlayHost } from "./host-contract.js";
import type { OverlayMode, ResizeDirection, ResizeHandle } from "./overlay-contract.js";
import { FrameScheduler, PointerSession } from "./pointer-session.js";

const HISTORY_AUTO_HEIGHT = 300;

interface OverlayControllerElements {
  root: HTMLElement;
  dragSurface: HTMLElement;
  resizeHandles: ResizeHandle[];
}

interface OverlayControllerCallbacks {
  onClearError(): void;
  onError(error: unknown): void;
  onModeApplied(mode: OverlayMode): void;
}

export class OverlayController {
  private modeValue: OverlayMode = "compact";

  private readonly dragSession: PointerSession;
  private readonly resizeSession: PointerSession;
  private readonly dragUpdateScheduler: FrameScheduler;
  private readonly resizeUpdateScheduler: FrameScheduler;

  constructor(
    private readonly host: OverlayHost,
    private readonly elements: OverlayControllerElements,
    private readonly callbacks: OverlayControllerCallbacks,
  ) {
    this.dragUpdateScheduler = new FrameScheduler(() => {
      if (!this.dragSession.isActive) {
        return;
      }
      void this.host.updateOverlayDrag().catch((error: unknown) => {
        this.callbacks.onError(error);
        this.clearDrag();
      });
    });
    this.resizeUpdateScheduler = new FrameScheduler(() => {
      if (!this.resizeSession.isActive) {
        return;
      }
      void this.host.updateOverlayResize().catch((error: unknown) => {
        this.callbacks.onError(error);
        this.clearResize();
      });
    });
    this.dragSession = new PointerSession(elements.root, "is-dragging", {
      onMove: () => this.dragUpdateScheduler.schedule(),
      onEnd: () => void this.finishDrag(),
    });
    this.resizeSession = new PointerSession(elements.root, "is-resizing", {
      onMove: () => this.resizeUpdateScheduler.schedule(),
      onEnd: () => void this.finishResize(),
    });
  }

  get mode(): OverlayMode {
    return this.modeValue;
  }

  bind(): void {
    this.elements.dragSurface.addEventListener("pointerdown", (event) => void this.startDrag(event));
    for (const handle of this.elements.resizeHandles) {
      handle.element.addEventListener("pointerdown", (event) => void this.startResize(event, handle.direction));
    }
    window.addEventListener("resize", () => this.syncModeFromWindowHeight());
    this.syncModeFromWindowHeight();
  }

  async minimize(): Promise<void> {
    try {
      await this.host.minimizeOverlay();
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
    }
  }

  close(): Promise<void> {
    return this.host.closeOverlay();
  }

  private applyMode(mode: OverlayMode): void {
    if (mode === this.modeValue) {
      return;
    }
    this.modeValue = mode;
    this.elements.root.setAttribute("data-overlay-mode", mode);
    this.callbacks.onModeApplied(mode);
  }

  private syncModeFromWindowHeight(): void {
    this.applyMode(window.innerHeight >= HISTORY_AUTO_HEIGHT ? "history" : "compact");
  }

  private async startDrag(event: PointerEvent): Promise<void> {
    if (event.button !== 0 || isInteractiveTarget(event.target)) {
      return;
    }
    blurActiveEditableControl();
    event.preventDefault();
    if (!this.dragSession.start(event, this.elements.dragSurface)) {
      return;
    }

    try {
      await this.host.startOverlayDrag();
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
      this.clearDrag();
    }
  }

  private async finishDrag(): Promise<void> {
    try {
      this.dragUpdateScheduler.cancel();
      await this.host.updateOverlayDrag();
      await this.host.endOverlayDrag();
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
    } finally {
      this.clearDrag();
    }
  }

  private clearDrag(): void {
    this.dragUpdateScheduler.cancel();
    this.dragSession.clear();
  }

  private async startResize(event: PointerEvent, direction: ResizeDirection): Promise<void> {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    const surface = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (!surface || !this.resizeSession.start(event, surface)) {
      return;
    }

    try {
      await this.host.startOverlayResize(direction);
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
      this.clearResize();
    }
  }

  private async finishResize(): Promise<void> {
    try {
      this.resizeUpdateScheduler.cancel();
      await this.host.endOverlayResize();
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
    } finally {
      this.clearResize();
    }
  }

  private clearResize(): void {
    this.resizeUpdateScheduler.cancel();
    this.resizeSession.clear();
  }
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  // #settings-panel is a child of the drag surface; pointerdowns on its padding,
  // labels, or status text must not start a window drag.
  return (
    target instanceof Element &&
    Boolean(target.closest("button,input,select,textarea,a,label,[contenteditable],#settings-panel"))
  );
}

function blurActiveEditableControl(): void {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof Element)) {
    return;
  }
  const tagName = activeElement.tagName.toLowerCase();
  if (tagName === "input" || tagName === "select" || tagName === "textarea") {
    (activeElement as HTMLElement).blur();
  }
}
