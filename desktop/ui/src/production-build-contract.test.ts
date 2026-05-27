import test from "node:test";
import assert from "node:assert/strict";

import { htmlElements } from "./test-contract-parsers.fixture.js";
import { desktopFiles, readDesktopFile } from "./test-project-files.fixture.js";

test("production UI build includes only runtime artifacts", () => {
  const files = desktopFiles("ui", "dist");

  assert.ok(files.includes("app.js"));
  assert.ok(files.includes("index.html"));
  assert.ok(files.includes("styles.css"));
  assert.deepEqual(
    files.filter((file) => /(?:^|[\\/])(?:test-|.*\.(?:test|fixture)\.js$)/u.test(file)),
    [],
  );
});

test("production HTML references only production UI assets", () => {
  const html = readDesktopFile("ui", "dist", "index.html");
  const stylesheets = htmlElements(html, "link").map((link) => link.href).filter(Boolean);
  const scripts = htmlElements(html, "script").map((script) => script.src).filter(Boolean);

  assert.deepEqual(stylesheets, ["./styles.css"]);
  assert.deepEqual(scripts, ["./app.js"]);
  assert.equal(html.includes("./src/"), false);
  assert.equal(html.includes("./dist/"), false);
});

test("production styles do not depend on inline data assets", () => {
  const css = readDesktopFile("ui", "dist", "styles.css");

  assert.equal(/url\(\s*["']?data:/iu.test(css), false);
  assert.equal(css.includes("data:image"), false);
});
