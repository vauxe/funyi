import test from "node:test";
import assert from "node:assert/strict";

import { readDesktopFile } from "./test-project-files.fixture.js";

const TAURI_CONFIG = JSON.parse(readDesktopFile("src-tauri", "tauri.conf.json")) as {
  app?: {
    security?: {
      csp?: string;
    };
  };
};

test("Tauri CSP stays scoped to packaged UI assets and the local ASR websocket", () => {
  const directives = cspDirectives(TAURI_CONFIG.app?.security?.csp);

  assert.deepEqual(directives.get("default-src"), ["'self'", "tauri:", "asset:"]);
  assert.deepEqual(directives.get("connect-src"), ["ws://127.0.0.1:*", "ws://localhost:*"]);
  assert.deepEqual(directives.get("style-src"), ["'self'", "'unsafe-inline'"]);

  for (const source of directives.values()) {
    assert.equal(source.includes("*"), false);
    assert.equal(source.includes("data:"), false);
    assert.equal(source.includes("http:"), false);
    assert.equal(source.includes("https:"), false);
  }
});

function cspDirectives(csp: string | undefined): Map<string, string[]> {
  assert.ok(csp, "missing Tauri CSP");
  return new Map(
    csp.split(";").flatMap((directive): Array<readonly [string, string[]]> => {
      const [name, ...sources] = directive.trim().split(/\s+/u);
      return name ? [[name, sources]] : [];
    }),
  );
}
