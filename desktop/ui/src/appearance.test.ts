import test from "node:test";
import assert from "node:assert/strict";

import {
  applyAppearance,
  backgroundImageCss,
  clampOpacity,
  DEFAULT_CAPTION_OPACITY,
  opacityToSlider,
  sliderToOpacity,
} from "./appearance.js";
import { asDomElement, FakeElement } from "./test-dom.fixture.js";

test("clamps opacity to the readable range and falls back on bad input", () => {
  assert.equal(clampOpacity(0.5), 0.5);
  assert.equal(clampOpacity(0), 0.2);
  assert.equal(clampOpacity(2), 1);
  assert.equal(clampOpacity(Number.NaN), DEFAULT_CAPTION_OPACITY);
});

test("converts between slider percent and opacity", () => {
  assert.equal(sliderToOpacity(50), 0.5);
  assert.equal(opacityToSlider(0.72), 72);
  assert.equal(opacityToSlider(sliderToOpacity(85)), 85);
});

test("builds a quoted CSS image value or none", () => {
  assert.equal(backgroundImageCss(null), "none");
  assert.equal(backgroundImageCss("blob:abc"), 'url("blob:abc")');
  assert.equal(backgroundImageCss('blob:a"b'), 'url("blob:a%22b")');
});

test("rejects non-blob image URLs to keep the CSS url() sink safe", () => {
  assert.equal(backgroundImageCss("https://evil.example/x.png"), "none");
  assert.equal(backgroundImageCss('data:image/png;base64,AAAA");}body{display:none'), "none");
});

test("applies opacity and image as CSS custom properties", () => {
  const root = new FakeElement();

  applyAppearance(asDomElement(root), { opacity: 0.5, imageUrl: "blob:abc" });

  assert.equal(root.styleValues.get("--caption-bg-opacity"), "0.50");
  assert.equal(root.styleValues.get("--caption-bg-image"), 'url("blob:abc")');
});
