import test from "node:test";
import assert from "node:assert/strict";

import { cssPxValue, cssRuleBody, rustNumberConst } from "./test-contract-parsers.fixture.js";
import { readDesktopFile } from "./test-project-files.fixture.js";

interface TauriWindowConfig {
  height?: number;
  minHeight?: number;
  minWidth?: number;
  width?: number;
}

const RUST_OVERLAY_SOURCE = readDesktopFile("src-tauri", "src", "overlay.rs");
const UI_STYLES = readDesktopFile("ui", "src", "styles.css");
const TAURI_CONFIG = JSON.parse(readDesktopFile("src-tauri", "tauri.conf.json")) as {
  app?: { windows?: TauriWindowConfig[] };
};

// These are genuine cross-artifact invariants: the Rust overlay geometry constants,
// the Tauri window config, and the CSS shell must agree. Pure styling values are
// covered behaviorally elsewhere and intentionally not snapshotted here.
test("Tauri window dimensions match the shared overlay geometry contract", () => {
  const mainWindow = TAURI_CONFIG.app?.windows?.[0];
  assert.ok(mainWindow, "missing Tauri main window config");

  assert.equal(mainWindow.width, overlaySize("COLLAPSED_WINDOW_WIDTH"));
  assert.equal(mainWindow.height, overlaySize("COLLAPSED_WINDOW_HEIGHT"));
  assert.equal(mainWindow.minWidth, overlaySize("MIN_OVERLAY_WIDTH"));
  assert.equal(mainWindow.minHeight, overlaySize("MIN_OVERLAY_HEIGHT"));
});

test("CSS shell dimensions match the native overlay geometry contract", () => {
  assert.equal(cssPxValue(UI_STYLES, "body", "min-width"), overlaySize("MIN_OVERLAY_WIDTH"));
  assert.equal(cssPxValue(UI_STYLES, "body", "min-height"), overlaySize("MIN_OVERLAY_HEIGHT"));
});

test("caption strip fills the native window height so captions cannot overflow it", () => {
  assert.match(cssRuleBody(UI_STYLES, ".caption-strip"), /height:\s*100vh;/u);
});

test("caption background image fades as one clipped layer behind content", () => {
  const strip = cssRuleBody(UI_STYLES, ".caption-strip");
  const imageLayer = cssRuleBody(UI_STYLES, ".caption-strip::before");

  assert.match(strip, /isolation:\s*isolate;/u);
  assert.match(strip, /background-color:\s*rgba\(5,\s*18,\s*20,\s*var\(--caption-bg-surface-opacity\)\);/u);
  assert.match(strip, /background-image:\s*none;/u);
  assert.match(imageLayer, /inset:\s*0;/u);
  assert.match(imageLayer, /border-radius:\s*inherit;/u);
  assert.match(imageLayer, /background-image:\s*var\(--caption-bg-image\);/u);
  assert.match(imageLayer, /opacity:\s*var\(--caption-bg-image-opacity\);/u);
});

test("transport pause pseudo-elements share the same grid cell", () => {
  const pauseRule = cssRuleBody(UI_STYLES, ".transport-control.is-pause::before,\n.transport-control.is-pause::after");

  assert.match(pauseRule, /grid-area:\s*1\s*\/\s*1/u);
});

test("narrow overlay transport controls fit beside the minimum service URL", () => {
  const narrowStyles = cssMediaBlock(UI_STYLES, "max-width: 420px");
  const minimumInnerWidth = overlaySize("MIN_OVERLAY_WIDTH") - 24;
  const minimumServiceUrlWidth = 98;
  const controlWidth = 18 + 5 * 24 + 5 * 3;

  assert.ok(controlWidth <= minimumInnerWidth - minimumServiceUrlWidth);
  assert.match(narrowStyles, /\.caption-controls\s*\{[\s\S]*gap:\s*3px;/u);
  assert.match(
    narrowStyles,
    /\.caption-controls \.icon-button,[\s\S]*\.caption-controls #transport-button\s*\{[\s\S]*width:\s*24px;/u,
  );
  assert.match(
    narrowStyles,
    /\.caption-controls \.icon-button,[\s\S]*\.caption-controls #transport-button\s*\{[\s\S]*height:\s*24px;/u,
  );
  assert.match(narrowStyles, /\.caption-controls \.volume-indicator\s*\{[\s\S]*width:\s*18px;[\s\S]*height:\s*24px;/u);
});

test("narrow overlays keep the service URL visible on the upper status row", () => {
  const narrowStyles = cssMediaBlock(UI_STYLES, "max-width: 420px");

  assert.match(narrowStyles, /\.status-controls\s*\{[\s\S]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s+92px;/u);
  assert.match(narrowStyles, /\.status-field-service\s*\{[\s\S]*grid-column:\s*1;/u);
  assert.match(narrowStyles, /\.status-field-service\s*\{[\s\S]*grid-row:\s*1;/u);
  assert.match(narrowStyles, /\.status-field-service\s*\{[\s\S]*max-width:\s*clamp\(98px,\s*34vw,\s*146px\);/u);
  assert.match(narrowStyles, /\.language-settings\s*\{[\s\S]*grid-column:\s*1;/u);
  assert.match(narrowStyles, /\.status-field-source\s*\{[\s\S]*grid-column:\s*2;/u);
});

function overlaySize(name: string): number {
  return rustNumberConst(RUST_OVERLAY_SOURCE, name);
}

function cssMediaBlock(source: string, query: string): string {
  const normalizedSource = source.replace(/\r\n?/gu, "\n");
  const start = normalizedSource.indexOf(`@media (${query}) {`);
  assert.ok(start >= 0, `missing CSS media block ${query}`);
  const bodyStart = normalizedSource.indexOf("{", start) + 1;
  let depth = 1;
  for (let index = bodyStart; index < normalizedSource.length; index += 1) {
    const char = normalizedSource[index];
    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return normalizedSource.slice(bodyStart, index);
      }
    }
  }
  assert.fail(`unterminated CSS media block ${query}`);
}
