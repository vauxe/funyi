import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const distDir = join(root, "ui", "dist");

rmSync(distDir, { force: true, recursive: true });
mkdirSync(distDir, { recursive: true });

execFileSync(process.execPath, [join(root, "node_modules", "typescript", "bin", "tsc"), "-p", "ui/tsconfig.json"], {
  cwd: root,
  stdio: "inherit",
});

const html = readFileSync(join(root, "ui", "index.html"), "utf8")
  .replace("./src/styles.css", "./styles.css")
  .replace("./dist/app.js", "./app.js");
writeFileSync(join(distDir, "index.html"), html);
copyFileSync(join(root, "ui", "src", "styles.css"), join(distDir, "styles.css"));
