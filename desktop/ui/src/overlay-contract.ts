export type OverlayMode = "compact" | "history";

export const RESIZE_DIRECTION_ATTRIBUTE = "data-resize-direction";
export const RESIZE_HANDLE_SELECTOR = `[${RESIZE_DIRECTION_ATTRIBUTE}]`;

export const RESIZE_DIRECTIONS = [
  "North",
  "East",
  "South",
  "West",
  "NorthEast",
  "NorthWest",
  "SouthEast",
  "SouthWest",
] as const;

export type ResizeDirection = (typeof RESIZE_DIRECTIONS)[number];

export function isResizeDirection(value: string | undefined): value is ResizeDirection {
  return RESIZE_DIRECTIONS.includes(value as ResizeDirection);
}
