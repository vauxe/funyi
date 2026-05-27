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

const html = [
  ["./src/styles.css", "./styles.css"],
  ["./dist/app.js", "./app.js"],
].reduce(
  (content, [from, to]) => replaceRequired(content, from, to),
  readFileSync(join(root, "ui", "index.html"), "utf8"),
);
writeFileSync(join(distDir, "index.html"), html);
copyFileSync(join(root, "ui", "src", "styles.css"), join(distDir, "styles.css"));

function replaceRequired(content, from, to) {
  if (!content.includes(from)) {
    throw new Error(`ui/index.html is missing required build placeholder: ${from}`);
  }
  return content.replace(from, to);
}
