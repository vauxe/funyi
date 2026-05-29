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

  // No directive may introduce a wildcard host or a remote/inline scheme beyond the
  // packaged UI + local websocket. Port wildcards like `ws://127.0.0.1:*` are allowed.
  for (const source of [...directives.values()].flat()) {
    assert.notEqual(source, "*", `wildcard source: ${source}`);
    assert.doesNotMatch(source, /^(?:data|http|https):/u, source);
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
