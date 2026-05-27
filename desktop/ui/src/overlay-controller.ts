import type { OverlayHost } from "./host-contract.js";
import type { OverlayMode, ResizeDirection } from "./overlay-contract.js";
import { DEFAULT_COMPACT_HEIGHT, nextCompactHeight } from "./overlay-resize.js";
import { FrameScheduler, PointerSession } from "./pointer-session.js";

interface ResizeHandle {
  element: HTMLElement;
  direction: ResizeDirection;
}

interface OverlayControllerElements {
  root: HTMLElement;
  dragSurface: HTMLElement;
  historyButton: HTMLButtonElement;
  resizeHandles: ResizeHandle[];
}

interface OverlayControllerCallbacks {
  onClearError(): void;
  onError(error: unknown): void;
  onModeApplied(mode: OverlayMode): void;
}

interface ActiveResize {
  direction: ResizeDirection;
  mode: OverlayMode;
  startY: number;
  startHeight: number;
}

export class OverlayController {
  private activeResize: ActiveResize | null = null;
  private compactHeight = DEFAULT_COMPACT_HEIGHT;
  private modeChanging = false;
  private modeValue: OverlayMode = "compact";
  private transitionSequence = 0;

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
      if (!this.activeResize) {
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
      onMove: (event) => this.handleResizeMove(event),
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
    this.elements.historyButton.addEventListener("click", () => void this.toggleHistory());
  }

  minimize(): Promise<void> {
    return this.runOverlayCommand(() => this.host.minimizeOverlay());
  }

  close(): Promise<void> {
    return this.host.closeOverlay();
  }

  private async toggleHistory(): Promise<void> {
    await this.setMode(this.modeValue === "history" ? "compact" : "history");
  }

  private async setMode(mode: OverlayMode): Promise<void> {
    const previousMode = this.modeValue;
    if (mode === previousMode || this.modeChanging) {
      return;
    }

    this.modeChanging = true;
    const transitionSequence = this.beginTransition();
    let modeApplied = false;
    try {
      this.applyMode(mode);
      modeApplied = true;
      await this.host.setOverlayMode(mode);
      this.callbacks.onClearError();
    } catch (error) {
      if (modeApplied && this.modeValue !== previousMode) {
        this.applyMode(previousMode);
      }
      this.callbacks.onError(error);
    } finally {
      this.scheduleTransitioningClear(transitionSequence);
      this.modeChanging = false;
    }
  }

  private applyMode(mode: OverlayMode): void {
    this.modeValue = mode;
    this.elements.root.setAttribute("data-overlay-mode", mode);
    syncModeButton(this.elements.historyButton, mode === "history", "Hide history", "Show history");
    this.callbacks.onModeApplied(mode);
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
    this.activeResize = {
      direction,
      mode: this.modeValue,
      startY: event.clientY,
      startHeight: this.compactHeight,
    };

    try {
      await this.host.startOverlayResize(direction);
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
      this.clearResize();
    }
  }

  private handleResizeMove(event: PointerEvent): void {
    const resize = this.activeResize;
    if (!resize) {
      return;
    }
    if (resize.mode === "compact") {
      this.applyCompactResizeCssHeight(resize, event.clientY);
    }
    this.resizeUpdateScheduler.schedule();
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
    this.activeResize = null;
    this.resizeUpdateScheduler.cancel();
    this.resizeSession.clear();
  }

  private applyCompactResizeCssHeight(resize: ActiveResize, currentY: number): void {
    const height = nextCompactHeight(resize, currentY, this.compactHeight);
    if (height === this.compactHeight) {
      return;
    }
    this.compactHeight = height;
    this.elements.root.style.setProperty("--compact-height", `${this.compactHeight}px`);
  }

  private async runOverlayCommand(command: () => Promise<void>): Promise<void> {
    try {
      await command();
      this.callbacks.onClearError();
    } catch (error) {
      this.callbacks.onError(error);
    }
  }

  private beginTransition(): number {
    this.transitionSequence += 1;
    this.elements.root.dataset.overlayTransitioning = "true";
    return this.transitionSequence;
  }

  private scheduleTransitioningClear(sequence: number): void {
    const clear = (): void => {
      if (sequence !== this.transitionSequence) {
        return;
      }
      delete this.elements.root.dataset.overlayTransitioning;
    };
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => requestAnimationFrame(clear));
      return;
    }
    window.setTimeout(clear, 32);
  }
}

function syncModeButton(button: HTMLButtonElement, expanded: boolean, expandedLabel: string, collapsedLabel: string): void {
  const label = expanded ? expandedLabel : collapsedLabel;
  button.classList.toggle("is-expanded", expanded);
  button.title = label;
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", String(expanded));
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  return target instanceof Element
    && Boolean(target.closest("button,input,select,textarea,a,label"));
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
