import test from "node:test";
import assert from "node:assert/strict";

import { AUDIO_CAPTURE_ERROR_EVENT, AUDIO_FRAME_EVENT } from "./audio-capture-events.js";
import { DESKTOP_COMMANDS } from "./desktop-host.js";
import { RESIZE_DIRECTIONS } from "./overlay-contract.js";
import { OVERLAY_DRAG_FINISHED_EVENT } from "./overlay-events.js";
import {
  rustEnumVariants,
  rustStringConst,
  rustTauriCommandNames,
  rustTauriCommandReturnType,
  rustTauriGenerateHandlerNames,
} from "./test-contract-parsers.fixture.js";
import { readDesktopFile } from "./test-project-files.fixture.js";

const TAURI_MAIN_SOURCE = readNativeSource("main.rs");
const RUST_AUDIO_SOURCE = readNativeSource("audio/mod.rs");
const RUST_OVERLAY_SOURCE = readNativeSource("overlay.rs");
const RUST_OVERLAY_WINDOW_SOURCE = readNativeSource("overlay_window.rs");

test("front-end Tauri commands are backed by Rust commands", () => {
  const rustCommandNames = rustTauriCommandNames(TAURI_MAIN_SOURCE);
  const rustHandlerNames = rustTauriGenerateHandlerNames(TAURI_MAIN_SOURCE);

  assert.deepEqual([...Object.values(DESKTOP_COMMANDS)].sort(), rustCommandNames.sort());
  assert.deepEqual(rustCommandNames.sort(), rustHandlerNames.sort());
});

test("native audio event names match the Rust event contract", () => {
  assert.equal(rustStringConst(RUST_AUDIO_SOURCE, "AUDIO_FRAME_EVENT"), AUDIO_FRAME_EVENT);
  assert.equal(rustStringConst(RUST_AUDIO_SOURCE, "AUDIO_CAPTURE_ERROR_EVENT"), AUDIO_CAPTURE_ERROR_EVENT);
});

test("native overlay event names match the Rust event contract", () => {
  assert.equal(rustStringConst(RUST_OVERLAY_WINDOW_SOURCE, "OVERLAY_DRAG_FINISHED_EVENT"), OVERLAY_DRAG_FINISHED_EVENT);
});

test("native overlay drag start return type matches the frontend mode split", () => {
  assert.equal(
    rustTauriCommandReturnType(TAURI_MAIN_SOURCE, DESKTOP_COMMANDS.startOverlayDrag),
    "Result<Option<u32>, String>",
  );
});

test("front-end resize values match Rust serde enums", () => {
  assert.deepEqual([...RESIZE_DIRECTIONS].sort(), rustEnumVariants(RUST_OVERLAY_SOURCE, "ResizeDirection").sort());
});

function readNativeSource(relativePath: string): string {
  return readDesktopFile("src-tauri", "src", relativePath);
}
