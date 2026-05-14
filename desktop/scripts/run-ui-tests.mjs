import { execFileSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const testDistDir = join(root, "ui", "test-dist");
const testFiles = readdirSync(testDistDir)
  .filter((name) => name.endsWith(".test.js"))
  .sort()
  .map((name) => join(testDistDir, name));

execFileSync(process.execPath, ["--test", ...testFiles], {
  cwd: root,
  stdio: "inherit",
});
