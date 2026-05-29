export type AudioLevelState = "silent" | "low" | "live";

// Structured capture stats carried through the status pipeline. Display formatting
// (if any) happens at the view layer; consumers compute from the numbers directly.
export interface AudioStats {
  levelDb: number | null;
  droppedFrames: number;
}

export interface AudioStatsState {
  hasDroppedFrames: boolean;
  level: AudioLevelState;
  volume: number;
}

export const EMPTY_AUDIO_STATS: AudioStats = { levelDb: null, droppedFrames: 0 };

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

export function audioStatsState({ levelDb, droppedFrames }: AudioStats): AudioStatsState {
  return {
    hasDroppedFrames: droppedFrames > 0,
    level: audioLevelState(levelDb),
    volume: audioVolume(levelDb),
  };
}

export function isAudible(levelDb: number | null): levelDb is number {
  return levelDb !== null && levelDb >= -80;
}

function audioLevelState(levelDb: number | null): AudioLevelState {
  if (levelDb === null || levelDb < -60) {
    return "silent";
  }
  return levelDb < -42 ? "low" : "live";
}

function audioVolume(levelDb: number | null): number {
  if (levelDb === null || levelDb < -80) {
    return 0;
  }
  // levelDb >= -80 here, so (levelDb + 80) / 60 >= 0; only the upper bound can clamp.
  const normalized = Math.min(1, (levelDb + 80) / 60);
  return Math.round(normalized * 100) / 100;
}
