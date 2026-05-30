export const OVERLAY_DRAG_FINISHED_EVENT = "overlay-drag-finished";
const DEFAULT_OVERLAY_DRAG_ERROR = "overlay drag release failed";

export interface OverlayDragFinished {
  dragId: number | null;
  error?: string;
}

export function parseOverlayDragFinished(payload: unknown): OverlayDragFinished {
  if (!isRecord(payload)) {
    throw new Error("overlay drag finished payload must be an object");
  }
  const dragId = parseOverlayDragId(payload.dragId, "overlay drag finished payload dragId");
  if (payload.error === undefined) {
    return { dragId };
  }
  if (typeof payload.error !== "string") {
    throw new Error("overlay drag finished payload error must be a string");
  }
  if (payload.error.length === 0) {
    return { dragId, error: DEFAULT_OVERLAY_DRAG_ERROR };
  }
  return { dragId, error: payload.error };
}

export function parseOverlayDragId(value: unknown, label = "overlay drag id"): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return value;
}

export function overlayDragFinishedParseError(error: unknown): OverlayDragFinished {
  const message = error instanceof Error ? error.message : String(error);
  return {
    dragId: null,
    error: message || DEFAULT_OVERLAY_DRAG_ERROR,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
