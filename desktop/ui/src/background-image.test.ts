import test from "node:test";
import assert from "node:assert/strict";

import { base64ToBytes, objectUrlFromStored, resolveStoredMime } from "./background-image.js";

test("decodes base64 into the original byte sequence", () => {
  const bytes = base64ToBytes(btoa(String.fromCharCode(80, 78, 71, 0, 255)));

  assert.deepEqual([...bytes], [80, 78, 71, 0, 255]);
});

test("returns an empty array for empty input", () => {
  assert.equal(base64ToBytes("").length, 0);
});

test("mints a same-origin blob URL from a stored payload", () => {
  const url = objectUrlFromStored({ mime: "image/png", data: btoa("png-bytes") });

  assert.match(url, /^blob:/u);
  URL.revokeObjectURL(url);
});

test("does not throw on an unexpected stored MIME (falls back internally)", () => {
  const url = objectUrlFromStored({ mime: "text/html", data: btoa("<script>") });

  assert.match(url, /^blob:/u);
  URL.revokeObjectURL(url);
});

test("downgrades a non-image stored MIME to JPEG and passes known image types through", () => {
  assert.equal(resolveStoredMime("text/html"), "image/jpeg");
  assert.equal(resolveStoredMime("application/octet-stream"), "image/jpeg");
  assert.equal(resolveStoredMime("image/png"), "image/png");
  assert.equal(resolveStoredMime("image/webp"), "image/webp");
});
