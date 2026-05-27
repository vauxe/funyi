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

test("compact caption strip follows the actual window height", () => {
  assert.match(cssRuleBody(UI_STYLES, ".caption-strip"), /height:\s*100vh;/u);
  assert.doesNotMatch(UI_STYLES, /--compact-height/u);
});

test("compact caption text is constrained by layout instead of replay truncation", () => {
  assert.match(cssRuleBody(UI_STYLES, ".caption-source,\n.caption-translation"), /display:\s*-webkit-box;/u);
  assert.match(cssRuleBody(UI_STYLES, ".caption-source"), /-webkit-line-clamp:\s*2;/u);
  assert.match(cssRuleBody(UI_STYLES, ".caption-line.previous .caption-source"), /-webkit-line-clamp:\s*1;/u);
  assert.match(UI_STYLES, /\.caption-translation\s*\{\s*-webkit-line-clamp:\s*1;/u);
});

test("caption typography keeps a stable readable hierarchy", () => {
  assert.match(cssRuleBody(UI_STYLES, ".caption-source"), /font-size:\s*clamp\(24px,\s*14vh,\s*30px\);/u);
  assert.match(
    cssRuleBody(UI_STYLES, '.app-shell[data-overlay-mode="history"] .caption-source'),
    /font-size:\s*22px;/u,
  );
  assert.match(UI_STYLES, /\.caption-translation\s*\{[\s\S]*?font-size:\s*clamp\(19px,\s*11vh,\s*23px\);/u);
  assert.match(
    cssRuleBody(UI_STYLES, '.app-shell[data-overlay-mode="history"] .caption-translation'),
    /font-size:\s*18px;/u,
  );
  assert.match(UI_STYLES, /\.history-source\s*\{[\s\S]*?font-size:\s*15px;/u);
  assert.match(UI_STYLES, /\.history-translation\s*\{[\s\S]*?font-size:\s*13px;/u);
  assert.doesNotMatch(cssRuleBody(UI_STYLES, ".history-item.is-latest .history-source"), /font-size/u);
});

test("history keeps the latest line above the bottom fade mask", () => {
  const historyList = cssRuleBody(UI_STYLES, ".history-list");
  const latestItem = cssRuleBody(UI_STYLES, ".history-item.is-latest");

  assert.match(historyList, /padding:\s*8px 12px 24px 10px;/u);
  assert.match(historyList, /scroll-padding-bottom:\s*24px;/u);
  assert.match(latestItem, /scroll-margin-bottom:\s*24px;/u);
});

test("history list remains scrollable while short history anchors to the bottom", () => {
  const historyList = cssRuleBody(UI_STYLES, ".history-list");
  const firstItem = cssRuleBody(UI_STYLES, ".history-item:first-child");

  assert.match(historyList, /display:\s*flex;/u);
  assert.match(historyList, /flex-direction:\s*column;/u);
  assert.doesNotMatch(historyList, /align-content:\s*end;/u);
  assert.match(firstItem, /margin-top:\s*auto;/u);
});

test("history rows keep source and translation separated under overflow", () => {
  const historyItem = cssRuleBody(UI_STYLES, ".history-item");

  assert.match(historyItem, /flex:\s*0 0 auto;/u);
  assert.match(historyItem, /row-gap:\s*5px;/u);
});

function overlaySize(name: string): number {
  return rustNumberConst(RUST_OVERLAY_SOURCE, name);
}
