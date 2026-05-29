import { APP_ELEMENT_SELECTORS } from "./app-dom.js";
import { RESIZE_DIRECTION_ATTRIBUTE, RESIZE_DIRECTIONS } from "./overlay-contract.js";
import { FakeDocument, FakeElement, installFakeDocument, installFakeElementConstructors } from "./test-dom.fixture.js";

export const APP_ELEMENT_IDS = [
  ...Object.values(APP_ELEMENT_SELECTORS).map(selectorId),
  ...RESIZE_DIRECTIONS.map(resizeHandleId),
] as readonly string[];

interface FakeAppDocumentOptions {
  elementIds?: readonly string[];
  installConstructors?: boolean;
  resizeDirections?: readonly string[];
}

export function installFakeAppDocument({
  elementIds = APP_ELEMENT_IDS,
  installConstructors = true,
  resizeDirections = RESIZE_DIRECTIONS,
}: FakeAppDocumentOptions = {}): Record<string, FakeElement> {
  const elements = Object.fromEntries(elementIds.map((id) => [id, new FakeElement(elementTag(id), id)]));
  for (const direction of resizeDirections) {
    const id = resizeHandleId(direction);
    elements[id] ||= new FakeElement(elementTag(id), id);
    elements[id].setAttribute(RESIZE_DIRECTION_ATTRIBUTE, direction);
  }

  installFakeDocument(new FakeDocument(elements));
  if (installConstructors) {
    installFakeElementConstructors();
  }
  return elements;
}

export function resizeHandleId(direction: string): string {
  return `resize-${direction.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase()}`;
}

function selectorId(selector: string): string {
  if (!selector.startsWith("#")) {
    throw new Error(`Fake app document only supports id selectors: ${selector}`);
  }
  return selector.slice(1);
}

function elementTag(id: string): string {
  if (id === "audio-source" || id === "language" || id === "translation-target-language") {
    return "select";
  }
  if (id === "app-shell") {
    return "main";
  }
  if (id === "caption-strip" || id === "history-list" || id === "settings-panel") {
    return "section";
  }
  if (id.endsWith("button")) {
    return "button";
  }
  if (id.startsWith("resize-")) {
    return "div";
  }
  if (id === "settings-status") {
    return "p";
  }
  if (id === "session-status" || id === "volume-indicator") {
    return "span";
  }
  if (id.includes("source") || id.includes("translation")) {
    return "div";
  }
  return "input";
}
