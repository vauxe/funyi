import type { OverlayHost } from "./host-contract.js";
import type { OverlayMode, ResizeDirection } from "./overlay-contract.js";

export type FakeOverlayMethod = keyof OverlayHost;

export interface FakeOverlayInvocation {
  method: FakeOverlayMethod;
  args?: Record<string, unknown>;
}

export interface FakeOverlayHost extends OverlayHost {
  invocations: FakeOverlayInvocation[];
}

export function createFakeOverlayHost(): FakeOverlayHost {
  const invocations: FakeOverlayInvocation[] = [];
  const invoke = async (method: FakeOverlayMethod, args?: Record<string, unknown>): Promise<void> => {
    invocations.push({ method, args });
  };

  return {
    invocations,
    closeOverlay: () => invoke("closeOverlay"),
    endOverlayDrag: () => invoke("endOverlayDrag"),
    endOverlayResize: () => invoke("endOverlayResize"),
    minimizeOverlay: () => invoke("minimizeOverlay"),
    setOverlayMode: (mode: OverlayMode) => invoke("setOverlayMode", { mode }),
    startOverlayDrag: () => invoke("startOverlayDrag"),
    startOverlayResize: (direction: ResizeDirection) => invoke("startOverlayResize", { direction }),
    updateOverlayDrag: () => invoke("updateOverlayDrag"),
    updateOverlayResize: () => invoke("updateOverlayResize"),
  };
}
