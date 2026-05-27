type TimerBehavior = "manual" | "real";
type WindowListener = (event?: unknown) => void;

export interface FakeWindowRuntimeOptions {
  extras?: Record<string, unknown>;
  timerBehavior?: TimerBehavior;
}

export class FakeWindowRuntime {
  readonly pendingTimeouts = new Map<number, () => void>();

  private readonly extras: Record<string, unknown>;
  private readonly listeners = new Map<string, WindowListener[]>();
  private readonly realTimeouts = new Map<number, ReturnType<typeof globalThis.setTimeout>>();
  private readonly timerBehavior: TimerBehavior;
  private installedWindow: Record<string, unknown> | null = null;
  private nextTimeoutId = 1;

  constructor({ extras = {}, timerBehavior = "manual" }: FakeWindowRuntimeOptions = {}) {
    this.extras = extras;
    this.timerBehavior = timerBehavior;
  }

  install(): this {
    this.installedWindow = {
      ...this.extras,
      innerHeight: this.extras.innerHeight ?? 180,
      addEventListener: (type: string, listener: WindowListener): void => {
        const typeListeners = this.listeners.get(type) || [];
        typeListeners.push(listener);
        this.listeners.set(type, typeListeners);
      },
      clearTimeout: (id: number): void => {
        this.clearTimeout(id);
      },
      dispatch: (type: string, event: unknown): void => {
        this.dispatch(type, event);
      },
      removeEventListener: (type: string, listener: WindowListener): void => {
        const typeListeners = this.listeners.get(type) || [];
        this.listeners.set(
          type,
          typeListeners.filter((item) => item !== listener),
        );
      },
      setTimeout: (callback: () => void, delay: number): number => this.setTimeout(callback, delay),
    };
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: this.installedWindow,
      writable: true,
    });
    return this;
  }

  dispatch(type: string, event: unknown): void {
    for (const listener of this.listeners.get(type) || []) {
      listener(event);
    }
  }

  runNextTimeout(): void {
    const next = this.pendingTimeouts.entries().next().value as [number, () => void] | undefined;
    if (!next) {
      return;
    }
    const [id, callback] = next;
    this.pendingTimeouts.delete(id);
    callback();
  }

  setInnerHeight(height: number): void {
    if (this.installedWindow) {
      this.installedWindow.innerHeight = height;
    }
  }

  private clearTimeout(id: number): void {
    if (this.timerBehavior === "manual") {
      this.pendingTimeouts.delete(id);
      return;
    }
    const timeout = this.realTimeouts.get(id);
    if (timeout) {
      globalThis.clearTimeout(timeout);
      this.realTimeouts.delete(id);
    }
  }

  private setTimeout(callback: () => void, delay: number): number {
    const id = this.nextTimeoutId;
    this.nextTimeoutId += 1;

    if (this.timerBehavior === "manual") {
      this.pendingTimeouts.set(id, callback);
      return id;
    }

    const timeout = globalThis.setTimeout(() => {
      this.realTimeouts.delete(id);
      callback();
    }, delay);
    this.realTimeouts.set(id, timeout);
    return id;
  }
}

export function installFakeWindowRuntime(options?: FakeWindowRuntimeOptions): FakeWindowRuntime {
  return new FakeWindowRuntime(options).install();
}
