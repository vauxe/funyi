import test from "node:test";
import assert from "node:assert/strict";

import { readDesktopFile } from "./test-project-files.fixture.js";

const UI_STYLES = readDesktopFile("ui", "src", "styles.css");

test("live captions use separate Miku colors for source and translation text", () => {
  assert.match(standaloneCssRuleBody(UI_STYLES, ".caption-source"), /color:\s*var\(--miku-representative\);/u);
  assert.match(standaloneCssRuleBody(UI_STYLES, ".caption-translation"), /color:\s*var\(--miku-hot-pink\);/u);
});

test("history captions keep source and translation text on the same Miku color roles", () => {
  assert.match(standaloneCssRuleBody(UI_STYLES, ".history-source"), /color:\s*var\(--miku-representative\);/u);
  assert.match(
    standaloneCssRuleBody(UI_STYLES, ".history-item.is-latest .history-source"),
    /color:\s*var\(--miku-representative\);/u,
  );
  assert.match(standaloneCssRuleBody(UI_STYLES, ".history-translation"), /color:\s*var\(--miku-hot-pink\);/u);
});

function standaloneCssRuleBody(source: string, selector: string): string {
  const normalizedSource = source.replace(/\r\n?/gu, "\n");
  const matches = normalizedSource.matchAll(new RegExp(`^${escapeRegExp(selector)}\\s*\\{([\\s\\S]*?)\\n\\}`, "gmu"));
  const match = [...matches].find(
    (candidate) => normalizedSource.slice((candidate.index ?? 0) - 2, candidate.index) !== ",\n",
  );
  assert.ok(match?.[1], `missing standalone CSS rule ${selector}`);
  return match[1];
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}
