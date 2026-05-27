import test from "node:test";
import assert from "node:assert/strict";

import {
  AUDIO_CAPTURE_ERROR_EVENT,
  AUDIO_FRAME_EVENT,
} from "./audio-capture-events.js";
import { DESKTOP_COMMANDS } from "./desktop-host.js";
import { OVERLAY_MODES, RESIZE_DIRECTIONS } from "./overlay-contract.js";
import {
  rustEnumVariants,
  rustStringConst,
  rustTauriCommandNames,
} from "./test-contract-parsers.fixture.js";
import { readDesktopFile } from "./test-project-files.fixture.js";

const TAURI_MAIN_SOURCE = readNativeSource("main.rs");
const RUST_AUDIO_SOURCE = readNativeSource("audio/mod.rs");
const RUST_OVERLAY_SOURCE = readNativeSource("overlay.rs");

test("front-end Tauri commands are backed by Rust commands", () => {
  const rustCommandNames = rustTauriCommandNames(TAURI_MAIN_SOURCE);

  assert.deepEqual(
    [...Object.values(DESKTOP_COMMANDS)].sort(),
    rustCommandNames.sort(),
  );
});

test("native audio event names match the Rust event contract", () => {
  assert.equal(rustStringConst(RUST_AUDIO_SOURCE, "AUDIO_FRAME_EVENT"), AUDIO_FRAME_EVENT);
  assert.equal(
    rustStringConst(RUST_AUDIO_SOURCE, "AUDIO_CAPTURE_ERROR_EVENT"),
    AUDIO_CAPTURE_ERROR_EVENT,
  );
});

test("front-end overlay values match Rust serde enums", () => {
  assert.deepEqual(
    [...OVERLAY_MODES].sort(),
    rustEnumVariants(RUST_OVERLAY_SOURCE, "OverlayMode").map((variant) => variant.toLowerCase()).sort(),
  );
  assert.deepEqual(
    [...RESIZE_DIRECTIONS].sort(),
    rustEnumVariants(RUST_OVERLAY_SOURCE, "ResizeDirection").sort(),
  );
});

function readNativeSource(relativePath: string): string {
  return readDesktopFile("src-tauri", "src", relativePath);
}
