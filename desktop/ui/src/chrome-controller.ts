import { type Clock, FinishTimeout } from "./finish-timeout.js";
import type { SessionState } from "./session-state.js";

export const DEFAULT_CHROME_IDLE_MS = 3_000;

type ChromeVisibility = "visible" | "hidden";

export interface ChromeControllerDeps {
  // The app shell carries `data-chrome` and doubles as the activity surface:
  // pointer or keyboard events anywhere inside it count as "the user is here".
  root: HTMLElement;
  // Pin chrome open while a transient interaction is mid-flight (e.g. the
  // settings panel is open) so a configuration pause does not tuck it away.
  shouldStayVisible?(): boolean;
  clock?: Clock;
  idleMs?: number;
}

// Auto-hides the control layer (caption controls + status line) after a spell of
// inactivity, in every session state — including before a session starts, so an
// untouched overlay settles into captions-only. CSS keys off the `data-chrome`
// attribute (see styles.css); this controller owns the state and the idle timer.
// Keyboard focus is handled by a `:focus-within` escape hatch in CSS, so a
// focused control stays visible even between activity pokes.
export class ChromeController {
  private readonly idle: FinishTimeout;
  private visibility: ChromeVisibility = "visible";
  private lastState: SessionState | null = null;

  constructor(private readonly deps: ChromeControllerDeps) {
    this.idle = new FinishTimeout(deps.clock, deps.idleMs ?? DEFAULT_CHROME_IDLE_MS);
  }

  init(): void {
    this.apply();
    const { root } = this.deps;
    const onActivity = (): void => this.handleActivity();
    root.addEventListener("pointermove", onActivity);
    root.addEventListener("pointerdown", onActivity);
    root.addEventListener("keydown", onActivity);
    root.addEventListener("focusin", onActivity);
    // Armed from launch: the controls tuck away after the first idle spell even
    // when no session is running yet.
    this.armIdle();
  }

  // A genuine session-state transition is a meaningful event — surface the
  // controls briefly so the change is noticed, then resume the idle countdown.
  // `onStateChange` can re-fire with the same state (audio availability nudges
  // it), so only act on an actual change.
  setSessionState(state: SessionState): void {
    if (state === this.lastState) {
      return;
    }
    this.lastState = state;
    this.handleActivity();
  }

  private handleActivity(): void {
    this.reveal();
    this.armIdle();
  }

  private armIdle(): void {
    this.idle.schedule(() => this.handleIdle());
  }

  private handleIdle(): void {
    if (this.deps.shouldStayVisible?.()) {
      // Still busy (panel open, etc.) — re-arm instead of hiding right now.
      this.armIdle();
      return;
    }
    this.setVisibility("hidden");
  }

  private reveal(): void {
    this.setVisibility("visible");
  }

  private setVisibility(visibility: ChromeVisibility): void {
    if (visibility === this.visibility) {
      return;
    }
    this.visibility = visibility;
    this.apply();
  }

  private apply(): void {
    this.deps.root.setAttribute("data-chrome", this.visibility);
  }
}
