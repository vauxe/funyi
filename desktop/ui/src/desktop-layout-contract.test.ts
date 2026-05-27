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

test("narrow overlay shares the URL row with window controls", () => {
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.caption-strip\s*\{[\s\S]*?grid-template-rows:\s*minmax\(0,\s*1fr\) auto;/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.status-line\s*\{[\s\S]*?grid-row:\s*2;/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.caption-controls\s*\{[\s\S]*?grid-row:\s*2;[\s\S]*?align-self:\s*start;[\s\S]*?justify-self:\s*end;[\s\S]*?z-index:\s*8;/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.status-controls\s*\{[\s\S]*?display:\s*grid;[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1fr\) 92px;[\s\S]*?grid-template-rows:\s*repeat\(2,\s*22px\);/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.status-field-service\s*\{[\s\S]*?grid-column:\s*1;[\s\S]*?grid-row:\s*1;[\s\S]*?max-width:\s*132px;/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.status-field-source\s*\{[\s\S]*?grid-column:\s*2;[\s\S]*?grid-row:\s*2;/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.language-settings\s*\{[\s\S]*?grid-column:\s*1;[\s\S]*?grid-row:\s*2;/u,
  );
  assert.match(
    UI_STYLES,
    /@media \(max-width: 420px\) \{[\s\S]*?\.status-field-service,\s*\.language-settings,\s*\.status-field-source\s*\{[\s\S]*?min-width:\s*0;/u,
  );
});

test("volume indicator uses dynamic bars without resizing the controls row", () => {
  const indicator = cssRuleBody(UI_STYLES, ".volume-indicator");
  const bar = cssRuleBody(UI_STYLES, ".volume-bar");

  assert.match(indicator, /grid-template-columns:\s*repeat\(3,\s*3px\);/u);
  assert.match(indicator, /width:\s*22px;/u);
  assert.match(indicator, /opacity:\s*0\.82;/u);
  assert.match(bar, /transform-origin:\s*bottom;/u);
  assert.match(bar, /transform 90ms linear;/u);
  assert.match(UI_STYLES, /\.volume-bar:nth-child\(2\)\s*\{[\s\S]*?scaleY\(var\(--volume-bar-mid\)\)/u);
  assert.match(UI_STYLES, /@media \(prefers-reduced-motion:\s*reduce\)/u);
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
