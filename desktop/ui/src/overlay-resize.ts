import type { ResizeDirection } from "./overlay-contract.js";

export const DEFAULT_COMPACT_HEIGHT = 180;
export const MIN_COMPACT_HEIGHT = 128;

interface CompactResizeAnchor {
  direction: ResizeDirection;
  startY: number;
  startHeight: number;
}

export function nextCompactHeight(
  resize: CompactResizeAnchor,
  currentY: number,
  fallback: number,
): number {
  const verticalDirection = compactVerticalResizeDirection(resize.direction);
  if (!verticalDirection) {
    return fallback;
  }

  const deltaY = currentY - resize.startY;
  const nextHeight = resize.startHeight + (verticalDirection === "South" ? deltaY : -deltaY);
  return clampCompactHeight(nextHeight, fallback);
}

function compactVerticalResizeDirection(direction: ResizeDirection): "North" | "South" | null {
  if (direction.includes("North")) {
    return "North";
  }
  if (direction.includes("South")) {
    return "South";
  }
  return null;
}

function clampCompactHeight(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.round(Math.max(MIN_COMPACT_HEIGHT, value));
}
