function clampMs(ms: number): number {
  return Math.max(0, Math.trunc(ms));
}

function pad(value: number, width: number): string {
  return String(value).padStart(width, "0");
}

// mm:ss.mmm for the in-window history clock (minutes are not wrapped at an hour).
export function formatClock(ms: number): string {
  const total = clampMs(ms);
  const seconds = Math.trunc(total / 1000);
  const minutes = Math.trunc(seconds / 60);
  return `${pad(minutes, 2)}:${pad(seconds % 60, 2)}.${pad(total % 1000, 3)}`;
}
