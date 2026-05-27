import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const testDistDir = join(root, "ui", "test-dist");

rmSync(testDistDir, { force: true, recursive: true });
mkdirSync(testDistDir, { recursive: true });

execFileSync(
  process.execPath,
  [join(root, "node_modules", "typescript", "bin", "tsc"), "-p", "ui/tsconfig.test.json"],
  {
    cwd: root,
    stdio: "inherit",
  },
);
