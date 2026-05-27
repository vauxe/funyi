import test from "node:test";
import assert from "node:assert/strict";

import { cssPxValue, rustNumberConst } from "./test-contract-parsers.fixture.js";
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
  assert.equal(cssPxValue(UI_STYLES, ".app-shell", "--compact-height"), overlaySize("COLLAPSED_WINDOW_HEIGHT"));
});

function overlaySize(name: string): number {
  return rustNumberConst(RUST_OVERLAY_SOURCE, name);
}
