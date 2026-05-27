export type AudioLevelState = "silent" | "low" | "live";

export interface AudioStatsState {
  hasDroppedFrames: boolean;
  level: AudioLevelState;
}

export function pcmLevelDb(bytes: Uint8Array): number | null {
  if (bytes.length < 2) {
    return null;
  }

  let sumSquares = 0;
  let samples = 0;
  for (let offset = 0; offset + 1 < bytes.length; offset += 2) {
    const low = bytes[offset] ?? 0;
    const high = bytes[offset + 1] ?? 0;
    let sample = low | (high << 8);
    if (sample >= 0x8000) {
      sample -= 0x10000;
    }
    const normalized = sample / 32768;
    sumSquares += normalized * normalized;
    samples += 1;
  }
  if (samples === 0 || sumSquares === 0) {
    return null;
  }
  return 20 * Math.log10(Math.sqrt(sumSquares / samples));
}

export function formatAudioStats(levelDb: number | null, droppedAudioFrames: number): string {
  const level = formatAudioLevel(levelDb);
  if (droppedAudioFrames <= 0) {
    return level;
  }
  return `${level}, dropped ${droppedAudioFrames}`;
}

export function parseAudioStatsState(value: string): AudioStatsState {
  const droppedFrames = droppedFrameCount(value);
  return {
    hasDroppedFrames: droppedFrames > 0,
    level: audioLevelState(value),
  };
}

export function isAudible(levelDb: number | null): levelDb is number {
  return levelDb !== null && levelDb >= -80;
}

function formatAudioLevel(levelDb: number | null): string {
  if (!isAudible(levelDb)) {
    return "Silent";
  }
  return `${Math.round(levelDb)}dB`;
}

function audioLevelState(value: string): AudioLevelState {
  if (/^silent$/i.test(value)) {
    return "silent";
  }
  const match = value.match(/(-?\d+)dB/i);
  const level = match ? Number.parseInt(match[1] || "", 10) : Number.NaN;
  if (!Number.isFinite(level) || level < -60) {
    return "silent";
  }
  return level < -42 ? "low" : "live";
}

function droppedFrameCount(value: string): number {
  const match = value.match(/\bdropped\s+(\d+)\b/i);
  return match ? Number.parseInt(match[1] || "0", 10) : 0;
}
