import type { OverlayHost } from "./host-contract.js";
import type { OverlayDragFinished } from "./overlay-events.js";
import type { ResizeDirection } from "./overlay-contract.js";

export type FakeOverlayMethod = Exclude<keyof OverlayHost, "listenOverlayDragFinished">;

export interface FakeOverlayInvocation {
  method: FakeOverlayMethod;
  args?: Record<string, unknown>;
}

export interface FakeOverlayHost extends OverlayHost {
  emitOverlayDragFinished(dragId: number | null, error?: string): void;
  invocations: FakeOverlayInvocation[];
}

export function createFakeOverlayHost(): FakeOverlayHost {
  const dragFinishedHandlers: Array<(event: OverlayDragFinished) => void> = [];
  const invocations: FakeOverlayInvocation[] = [];
  let nextDragId = 0;
  const invoke = async (method: FakeOverlayMethod, args?: Record<string, unknown>): Promise<void> => {
    invocations.push({ method, args });
  };

  return {
    invocations,
    closeOverlay: () => invoke("closeOverlay"),
    emitOverlayDragFinished: (dragId: number | null, error?: string) => {
      const event = error === undefined ? { dragId } : { dragId, error };
      for (const handler of [...dragFinishedHandlers]) {
        handler(event);
      }
    },
    endOverlayDrag: () => invoke("endOverlayDrag"),
    endOverlayResize: () => invoke("endOverlayResize"),
    listenOverlayDragFinished: async (handler: (event: OverlayDragFinished) => void) => {
      dragFinishedHandlers.push(handler);
      return () => {
        const index = dragFinishedHandlers.indexOf(handler);
        if (index >= 0) {
          dragFinishedHandlers.splice(index, 1);
        }
      };
    },
    minimizeOverlay: () => invoke("minimizeOverlay"),
    startOverlayDrag: async () => {
      await invoke("startOverlayDrag");
      const dragId = nextDragId;
      nextDragId += 1;
      return dragId;
    },
    startOverlayResize: (direction: ResizeDirection) => invoke("startOverlayResize", { direction }),
    updateOverlayDrag: () => invoke("updateOverlayDrag"),
    updateOverlayResize: () => invoke("updateOverlayResize"),
  };
}
