import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

export function readDesktopFile(...segments: string[]): string {
  return readFileSync(desktopPath(...segments), "utf8");
}

export function desktopFiles(...segments: string[]): string[] {
  const root = desktopPath(...segments);
  return relativeFiles(root).map((file) => file.replaceAll("\\", "/"));
}

function desktopPath(...segments: string[]): string {
  return join(process.cwd(), ...segments);
}

function relativeFiles(root: string, directory = root): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      return relativeFiles(root, path);
    }
    return relative(root, path);
  });
}
