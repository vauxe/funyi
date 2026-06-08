import { isInteger } from "./runtime-guards.js";
import { formatClock } from "./time-format.js";

export interface TranscriptLine {
  startMs: number | null;
  text: string;
  translation: string | null;
}

export interface TranscriptExportOptions {
  translationEnabled: boolean;
}

// Render committed transcript lines as plain text for the clipboard. Each line
// becomes one block: an optional [mm:ss.mmm] timestamp with the source, and the
// translation on the next line prefixed with "->" so source and target never blur.
// Lines come from SubtitleDocument.exportLines(), the same stable projection used by SRT.
export function formatTranscript(
  lines: readonly TranscriptLine[],
  { translationEnabled }: TranscriptExportOptions,
): string {
  const blocks: string[] = [];
  for (const line of lines) {
    const source = line.text.trim();
    const translation = translationEnabled ? (line.translation ?? "").trim() : "";
    if (!source && !translation) {
      continue;
    }
    const block: string[] = [];
    if (source) {
      const stamp = isInteger(line.startMs) ? formatClock(line.startMs) : "";
      block.push(stamp ? `[${stamp}] ${source}` : source);
    }
    if (translation) {
      block.push(`-> ${translation}`);
    }
    blocks.push(block.join("\n"));
  }
  return blocks.join("\n");
}

export async function copyToClipboard(text: string): Promise<void> {
  const clipboard = (globalThis.navigator as Navigator | undefined)?.clipboard;
  if (!clipboard || typeof clipboard.writeText !== "function") {
    throw new Error("Clipboard is unavailable");
  }
  await clipboard.writeText(text);
}
