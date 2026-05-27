import type { TauriRuntime } from "./tauri-runtime.js";
import { type FakeWindowRuntime, installFakeWindowRuntime } from "./test-window.fixture.js";

type TimerBehavior = "manual" | "real";
type TauriEventHandler = (event: { payload: unknown }) => void;

export interface TauriInvocation {
  command: string;
  args?: Record<string, unknown>;
}

interface FakeTauriRuntimeOptions {
  invoke?: (command: string, args?: Record<string, unknown>) => unknown | Promise<unknown>;
  timerBehavior?: TimerBehavior;
}

export class FakeTauriRuntime {
  readonly invocations: TauriInvocation[] = [];
  unlistenCount = 0;

  private readonly listeners = new Map<string, TauriEventHandler[]>();
  private windowRuntime: FakeWindowRuntime | null = null;

  constructor(private readonly options: FakeTauriRuntimeOptions = {}) {}

  install(): this {
    const runtime: TauriRuntime = {
      core: {
        invoke: async <TResult>(command: string, args?: Record<string, unknown>): Promise<TResult> => {
          this.invocations.push({ command, args });
          return (await this.options.invoke?.(command, args)) as TResult;
        },
      },
      event: {
        listen: async <TPayload>(
          event: string,
          handler: (event: { payload: TPayload }) => void,
        ): Promise<() => void> => {
          const listener = handler as TauriEventHandler;
          const handlers = this.listeners.get(event) || [];
          handlers.push(listener);
          this.listeners.set(event, handlers);
          return () => {
            this.unlistenCount += 1;
            this.listeners.set(
              event,
              (this.listeners.get(event) || []).filter((item) => item !== listener),
            );
          };
        },
      },
    };

    this.windowRuntime = installFakeWindowRuntime({
      extras: {
        __TAURI__: runtime,
      },
      timerBehavior: this.options.timerBehavior,
    });
    return this;
  }

  dispatchWindow(type: string, event: unknown): void {
    this.windowRuntime?.dispatch(type, event);
  }

  emit(event: string, payload: unknown): void {
    for (const handler of this.listeners.get(event) || []) {
      handler({ payload });
    }
  }
}

export function installFakeTauriRuntime(options?: FakeTauriRuntimeOptions): FakeTauriRuntime {
  return new FakeTauriRuntime(options).install();
}
