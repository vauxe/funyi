import { errorMessage } from "./error-message.js";
import type { OverlayHost } from "./host-contract.js";
import type { OverlayDragFinished } from "./overlay-events.js";
import type { OverlayMode, ResizeDirection, ResizeHandle } from "./overlay-contract.js";
import { FrameScheduler, PointerSession } from "./pointer-session.js";

const HISTORY_AUTO_HEIGHT = 300;
const MAX_BUFFERED_NATIVE_DRAG_FINISHES = 16;

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

type OverlayGestureState =
  | { phase: "idle" }
  | {
      earlyNativeDragFinishedEvents: Map<number, OverlayDragFinished>;
      earlyNativeDragFinishedError: OverlayDragFinished | null;
      phase: "drag-starting";
      pointerEnded: boolean;
      token: number;
    }
  | { phase: "drag-active"; token: number }
  | { phase: "drag-native"; dragId: number; token: number }
  | { phase: "drag-ending"; token: number }
  | { phase: "resize-starting"; pointerEnded: boolean; token: number }
  | { phase: "resize-active"; token: number }
  | { phase: "resize-ending"; token: number };

export class OverlayController {
  private modeValue: OverlayMode = "compact";
  private gesture: OverlayGestureState = { phase: "idle" };

  private readonly dragSession: PointerSession;
  private readonly resizeSession: PointerSession;
  private readonly dragUpdateScheduler: FrameScheduler;
  private readonly resizeUpdateScheduler: FrameScheduler;
  private readonly dragMoveUpdates: GestureUpdateQueue;
  private readonly resizeMoveUpdates: GestureUpdateQueue;

  constructor(
    private readonly host: OverlayHost,
    private readonly elements: OverlayControllerElements,
    private readonly callbacks: OverlayControllerCallbacks,
  ) {
    this.dragMoveUpdates = new GestureUpdateQueue(
      () => this.host.updateOverlayDrag(),
      (token) => this.canUpdateDrag(token),
      (token, error) => this.reportDragUpdateError(token, error),
    );
    this.resizeMoveUpdates = new GestureUpdateQueue(
      () => this.host.updateOverlayResize(),
      (token) => this.canUpdateResize(token),
      (token, error) => this.reportResizeUpdateError(token, error),
    );
    this.dragUpdateScheduler = new FrameScheduler(() => {
      const token = this.dragSession.activeToken;
      if (token === null || !this.canUpdateDrag(token)) {
        return;
      }
      this.dragMoveUpdates.request(token);
    });
    this.resizeUpdateScheduler = new FrameScheduler(() => {
      const token = this.resizeSession.activeToken;
      if (token === null || !this.canUpdateResize(token)) {
        return;
      }
      this.resizeMoveUpdates.request(token);
    });
    this.dragSession = new PointerSession(elements.root, "is-dragging", {
      onStaleSession: (token) => this.handleStaleDragSession(token),
      onMove: () => this.dragUpdateScheduler.schedule(),
      onEnd: (_event, token) => void this.finishDrag(token),
    });
    this.resizeSession = new PointerSession(elements.root, "is-resizing", {
      onStaleSession: (token) => this.handleStaleResizeSession(token),
      onMove: () => this.resizeUpdateScheduler.schedule(),
      onEnd: (_event, token) => void this.finishResize(token),
    });
  }

  get mode(): OverlayMode {
    return this.modeValue;
  }

  bind(): void {
    window.addEventListener("resize", () => this.syncModeFromWindowHeight());
    this.syncModeFromWindowHeight();
    void this.bindGestures();
  }

  private async bindGestures(): Promise<void> {
    try {
      await this.host.listenOverlayDragFinished((event) => this.handleNativeDragFinished(event));
    } catch (error) {
      this.callbacks.onError(error);
      return;
    }
    this.elements.dragSurface.addEventListener("pointerdown", (event) => void this.startDrag(event));
    for (const handle of this.elements.resizeHandles) {
      handle.element.addEventListener("pointerdown", (event) => void this.startResize(event, handle.direction));
    }
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
    if (!this.prepareForGestureStart()) {
      return;
    }
    blurActiveEditableControl();
    event.preventDefault();
    if (!this.dragSession.start(event, this.elements.dragSurface)) {
      return;
    }
    const token = this.dragSession.activeToken;
    if (token === null) {
      return;
    }

    this.gesture = {
      earlyNativeDragFinishedEvents: new Map(),
      earlyNativeDragFinishedError: null,
      phase: "drag-starting",
      pointerEnded: false,
      token,
    };
    try {
      const dragId = await this.host.startOverlayDrag();
      if (this.gesture.phase !== "drag-starting" || this.gesture.token !== token) {
        return;
      }
      const earlyNativeDragFinishedEvents = this.gesture.earlyNativeDragFinishedEvents;
      const earlyNativeDragFinishedError = this.gesture.earlyNativeDragFinishedError;
      if (earlyNativeDragFinishedError) {
        this.finishNativeDrag(earlyNativeDragFinishedError, token);
        return;
      }
      if (dragId !== null) {
        this.gesture = { phase: "drag-native", dragId, token };
        const earlyFinishedEvent = earlyNativeDragFinishedEvents.get(dragId);
        if (earlyFinishedEvent) {
          this.finishNativeDrag(earlyFinishedEvent, token);
          return;
        }
        this.callbacks.onClearError();
        return;
      }
      const finishAfterStart = this.gesture.pointerEnded;
      this.gesture = { phase: "drag-active", token };
      this.callbacks.onClearError();
      if (finishAfterStart) {
        await this.finishDrag(token);
      }
    } catch (error) {
      this.reportDragError(token, error);
    }
  }

  private async finishDrag(token: number | null): Promise<void> {
    if (token === null || !dragStateHasToken(this.gesture, token)) {
      return;
    }

    if (this.gesture.phase === "drag-starting") {
      this.gesture = { ...this.gesture, pointerEnded: true };
      return;
    }

    if (this.gesture.phase === "drag-native") {
      this.dragUpdateScheduler.cancel();
      return;
    }

    if (this.gesture.phase !== "drag-active") {
      return;
    }

    this.gesture = { phase: "drag-ending", token };
    try {
      this.dragUpdateScheduler.cancel();
      await this.dragMoveUpdates.drain();
      if (!gestureIs(this.gesture, "drag-ending", token)) {
        return;
      }
      let finishError: unknown;
      let hasFinishError = false;
      try {
        await this.host.updateOverlayDrag();
      } catch (error) {
        finishError = error;
        hasFinishError = true;
      }
      if (!gestureIs(this.gesture, "drag-ending", token)) {
        return;
      }
      try {
        await this.host.endOverlayDrag();
      } catch (error) {
        finishError = hasFinishError ? combinedGestureError(finishError, error) : error;
        hasFinishError = true;
      }
      if (gestureIs(this.gesture, "drag-ending", token)) {
        if (hasFinishError) {
          this.callbacks.onError(finishError);
        } else {
          this.callbacks.onClearError();
        }
      }
    } catch (error) {
      if (gestureIs(this.gesture, "drag-ending", token)) {
        this.callbacks.onError(error);
      }
    } finally {
      this.clearDrag(token);
    }
  }

  private clearDrag(token?: number | null): void {
    if (token != null && !dragStateHasToken(this.gesture, token) && this.dragSession.activeToken !== token) {
      return;
    }
    this.dragUpdateScheduler.cancel();
    this.dragMoveUpdates.cancelQueued();
    this.dragSession.clear(token);
    if (token == null) {
      if (isDragState(this.gesture)) {
        this.gesture = { phase: "idle" };
      }
      return;
    }
    if (dragStateHasToken(this.gesture, token)) {
      this.gesture = { phase: "idle" };
    }
  }

  private handleNativeDragFinished(event: OverlayDragFinished): void {
    if (event.dragId === null) {
      if (event.error === undefined) {
        return;
      }
      if (this.gesture.phase === "drag-native") {
        this.finishNativeDrag(event, this.gesture.token);
        return;
      }
      if (this.gesture.phase === "drag-starting") {
        this.gesture = { ...this.gesture, earlyNativeDragFinishedError: event };
      }
      return;
    }
    if (this.gesture.phase === "drag-native" && this.gesture.dragId === event.dragId) {
      this.finishNativeDrag(event, this.gesture.token);
      return;
    }
    if (this.gesture.phase !== "drag-starting") {
      return;
    }
    this.gesture.earlyNativeDragFinishedEvents.set(event.dragId, event);
    if (this.gesture.earlyNativeDragFinishedEvents.size > MAX_BUFFERED_NATIVE_DRAG_FINISHES) {
      const oldest = this.gesture.earlyNativeDragFinishedEvents.keys().next().value;
      if (oldest !== undefined) {
        this.gesture.earlyNativeDragFinishedEvents.delete(oldest);
      }
    }
  }

  private finishNativeDrag(event: OverlayDragFinished, token: number): void {
    this.clearDrag(token);
    if (event.error) {
      this.callbacks.onError(new Error(event.error));
      return;
    }
    this.callbacks.onClearError();
  }

  private async startResize(event: PointerEvent, direction: ResizeDirection): Promise<void> {
    if (event.button !== 0 || !this.prepareForGestureStart()) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    const surface = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (!surface || !this.resizeSession.start(event, surface)) {
      return;
    }
    const token = this.resizeSession.activeToken;
    if (token === null) {
      return;
    }

    this.gesture = { phase: "resize-starting", pointerEnded: false, token };
    try {
      await this.host.startOverlayResize(direction);
      if (this.gesture.phase !== "resize-starting" || this.gesture.token !== token) {
        return;
      }
      const finishAfterStart = this.gesture.pointerEnded;
      this.gesture = { phase: "resize-active", token };
      this.callbacks.onClearError();
      if (finishAfterStart) {
        await this.finishResize(token);
      }
    } catch (error) {
      this.reportResizeError(token, error);
    }
  }

  private async finishResize(token: number | null): Promise<void> {
    if (token === null || !resizeStateHasToken(this.gesture, token)) {
      return;
    }
    if (this.gesture.phase === "resize-starting") {
      this.gesture = { ...this.gesture, pointerEnded: true };
      return;
    }

    if (this.gesture.phase !== "resize-active") {
      return;
    }

    this.gesture = { phase: "resize-ending", token };
    try {
      this.resizeUpdateScheduler.cancel();
      await this.resizeMoveUpdates.drain();
      if (!gestureIs(this.gesture, "resize-ending", token)) {
        return;
      }
      await this.host.endOverlayResize();
      if (gestureIs(this.gesture, "resize-ending", token)) {
        this.callbacks.onClearError();
      }
    } catch (error) {
      if (gestureIs(this.gesture, "resize-ending", token)) {
        this.callbacks.onError(error);
      }
    } finally {
      this.clearResize(token);
    }
  }

  private clearResize(token?: number | null): void {
    if (token != null && !resizeStateHasToken(this.gesture, token) && this.resizeSession.activeToken !== token) {
      return;
    }
    this.resizeUpdateScheduler.cancel();
    this.resizeMoveUpdates.cancelQueued();
    this.resizeSession.clear(token);
    if (token == null) {
      if (isResizeState(this.gesture)) {
        this.gesture = { phase: "idle" };
      }
      return;
    }
    if (resizeStateHasToken(this.gesture, token)) {
      this.gesture = { phase: "idle" };
    }
  }

  private reportDragUpdateError(token: number, error: unknown): void {
    if (!gestureIs(this.gesture, "drag-active", token)) {
      return;
    }
    void this.abortDrag(token, { drainUpdates: false, error });
  }

  private reportResizeUpdateError(token: number, error: unknown): void {
    if (!gestureIs(this.gesture, "resize-active", token)) {
      return;
    }
    void this.abortResize(token, { drainUpdates: false, error });
  }

  private reportDragError(token: number, error: unknown): void {
    if (!dragStateHasToken(this.gesture, token)) {
      return;
    }
    this.callbacks.onError(error);
    this.clearDrag(token);
  }

  private reportResizeError(token: number, error: unknown): void {
    if (!resizeStateHasToken(this.gesture, token)) {
      return;
    }
    this.callbacks.onError(error);
    this.clearResize(token);
  }

  private handleStaleDragSession(token: number): void {
    if (gestureIs(this.gesture, "drag-active", token)) {
      void this.abortDrag(token);
      return;
    }
    this.clearDrag(token);
  }

  private handleStaleResizeSession(token: number): void {
    if (gestureIs(this.gesture, "resize-active", token)) {
      void this.abortResize(token);
      return;
    }
    this.clearResize(token);
  }

  private async abortDrag(token: number, reason?: { drainUpdates?: boolean; error: unknown }): Promise<void> {
    if (!gestureIs(this.gesture, "drag-active", token)) {
      return;
    }
    this.gesture = { phase: "drag-ending", token };
    this.dragUpdateScheduler.cancel();
    this.dragMoveUpdates.cancelQueued();
    if (reason?.drainUpdates !== false) {
      await this.dragMoveUpdates.drain();
      if (!gestureIs(this.gesture, "drag-ending", token)) {
        return;
      }
    }
    if (reason) {
      this.callbacks.onError(reason.error);
    }
    try {
      await this.host.endOverlayDrag();
      if (!reason && gestureIs(this.gesture, "drag-ending", token)) {
        this.callbacks.onClearError();
      }
    } catch (endError) {
      if (gestureIs(this.gesture, "drag-ending", token)) {
        this.callbacks.onError(endError);
      }
    } finally {
      this.clearDrag(token);
    }
  }

  private async abortResize(token: number, reason?: { drainUpdates?: boolean; error: unknown }): Promise<void> {
    if (!gestureIs(this.gesture, "resize-active", token)) {
      return;
    }
    this.gesture = { phase: "resize-ending", token };
    this.resizeUpdateScheduler.cancel();
    this.resizeMoveUpdates.cancelQueued();
    if (reason?.drainUpdates !== false) {
      await this.resizeMoveUpdates.drain();
      if (!gestureIs(this.gesture, "resize-ending", token)) {
        return;
      }
    }
    if (reason) {
      this.callbacks.onError(reason.error);
    }
    try {
      await this.host.endOverlayResize();
      if (!reason && gestureIs(this.gesture, "resize-ending", token)) {
        this.callbacks.onClearError();
      }
    } catch (endError) {
      if (gestureIs(this.gesture, "resize-ending", token)) {
        this.callbacks.onError(endError);
      }
    } finally {
      this.clearResize(token);
    }
  }

  private prepareForGestureStart(): boolean {
    if (this.gesture.phase === "idle" || this.gesture.phase === "resize-active") {
      this.resizeSession.clearIfCaptureLost();
    }
    if (this.resizeSession.isActive) {
      return false;
    }
    if (this.gesture.phase === "idle" || this.gesture.phase === "drag-active") {
      this.dragSession.clearIfCaptureLost();
    }
    if (this.dragSession.isActive) {
      return false;
    }
    return this.gesture.phase === "idle";
  }

  private canUpdateDrag(token: number): boolean {
    return gestureIs(this.gesture, "drag-active", token);
  }

  private canUpdateResize(token: number): boolean {
    return gestureIs(this.gesture, "resize-active", token);
  }
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  // #settings-panel is a child of the drag surface; pointerdowns on its padding,
  // labels, status text, or explicitly selectable regions must not start a window drag.
  return (
    target instanceof Element &&
    Boolean(target.closest("button,input,select,textarea,a,label,#settings-panel,[data-overlay-drag-ignore]"))
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

class GestureUpdateQueue {
  private inFlight: Promise<void> | null = null;
  private queuedToken: number | null = null;

  constructor(
    private readonly runUpdate: () => Promise<void>,
    private readonly canUpdate: (token: number) => boolean,
    private readonly onUpdateError: (token: number, error: unknown) => void,
  ) {}

  request(token: number): void {
    if (!this.canUpdate(token)) {
      return;
    }
    if (this.inFlight) {
      this.queuedToken = token;
      return;
    }
    this.start(token);
  }

  cancelQueued(): void {
    this.queuedToken = null;
  }

  async drain(): Promise<void> {
    this.cancelQueued();
    const inFlight = this.inFlight;
    if (inFlight) {
      await inFlight.catch(() => {});
    }
  }

  private start(token: number): void {
    if (!this.canUpdate(token)) {
      return;
    }
    const operation = this.runUpdate()
      .catch((error: unknown) => {
        this.onUpdateError(token, error);
      })
      .finally(() => {
        if (this.inFlight === operation) {
          this.inFlight = null;
        }
        const queuedToken = this.queuedToken;
        this.queuedToken = null;
        if (queuedToken !== null) {
          this.start(queuedToken);
        }
      });
    this.inFlight = operation;
  }
}

function gestureIs(state: OverlayGestureState, phase: OverlayGestureState["phase"], token: number): boolean {
  return state.phase === phase && "token" in state && state.token === token;
}

function isDragState(state: OverlayGestureState): boolean {
  return (
    state.phase === "drag-starting" ||
    state.phase === "drag-active" ||
    state.phase === "drag-native" ||
    state.phase === "drag-ending"
  );
}

function isResizeState(state: OverlayGestureState): boolean {
  return state.phase === "resize-starting" || state.phase === "resize-active" || state.phase === "resize-ending";
}

function dragStateHasToken(state: OverlayGestureState, token: number): boolean {
  return isDragState(state) && "token" in state && state.token === token;
}

function resizeStateHasToken(state: OverlayGestureState, token: number): boolean {
  return isResizeState(state) && "token" in state && state.token === token;
}

function combinedGestureError(first: unknown, second: unknown): Error {
  return new Error(`${errorMessage(first)}; ${errorMessage(second)}`);
}
