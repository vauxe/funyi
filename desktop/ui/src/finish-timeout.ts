export const DEFAULT_FINISH_TIMEOUT_MS = 120_000;

export interface Clock {
  clearTimeout(handle: unknown): void;
  setTimeout(callback: () => void | Promise<void>, delay: number): unknown;
}

export const DEFAULT_CLOCK: Clock = {
  clearTimeout: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
  setTimeout: (callback, delay) => globalThis.setTimeout(callback, delay),
};

export class FinishTimeout {
  private handle: unknown = null;

  constructor(
    private readonly clock: Clock = DEFAULT_CLOCK,
    private readonly delayMs = DEFAULT_FINISH_TIMEOUT_MS,
  ) {}

  clear(): void {
    if (this.handle === null) {
      return;
    }
    this.clock.clearTimeout(this.handle);
    this.handle = null;
  }

  schedule(callback: () => void | Promise<void>): void {
    this.clear();
    this.handle = this.clock.setTimeout(callback, this.delayMs);
  }
}
