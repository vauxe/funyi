// Caption-panel appearance: background opacity and an optional background image.
// The values are written as CSS custom properties onto the app shell (they inherit
// down to the caption strip), so applying them needs only a handle we already own.

export const DEFAULT_CAPTION_OPACITY = 0.72;
const MIN_CAPTION_OPACITY = 0.2;
const MAX_CAPTION_OPACITY = 1;
const OPACITY_VARIABLE = "--caption-bg-opacity";
const IMAGE_VARIABLE = "--caption-bg-image";

export interface AppearanceState {
  opacity: number;
  imageUrl: string | null;
}

export function clampOpacity(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_CAPTION_OPACITY;
  }
  return Math.min(MAX_CAPTION_OPACITY, Math.max(MIN_CAPTION_OPACITY, value));
}

export function sliderToOpacity(sliderValue: number): number {
  return clampOpacity(sliderValue / 100);
}

export function opacityToSlider(opacity: number): number {
  return Math.round(clampOpacity(opacity) * 100);
}

// Only same-origin blob: URLs (minted by URL.createObjectURL) are ever rendered as
// the background; reject anything else so a tampered value cannot reach the CSS
// url() sink. The quote-escape is belt-and-suspenders since blob: URLs are opaque.
export function backgroundImageCss(imageUrl: string | null): string {
  if (!imageUrl?.startsWith("blob:")) {
    return "none";
  }
  return `url("${imageUrl.replace(/"/gu, "%22")}")`;
}

export function applyAppearance(root: HTMLElement, state: AppearanceState): void {
  root.style.setProperty(OPACITY_VARIABLE, clampOpacity(state.opacity).toFixed(2));
  root.style.setProperty(IMAGE_VARIABLE, backgroundImageCss(state.imageUrl));
}
