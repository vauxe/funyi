import assert from "node:assert/strict";

export type HtmlAttributes = Record<string, string>;

export function cssPxValue(source: string, selector: string, property: string): number {
  const body = cssRuleBody(source, selector);
  const match = new RegExp(`${escapeRegExp(property)}:\\s*([0-9.]+)px\\s*;`, "u").exec(body);
  assert.ok(match?.[1], `missing CSS pixel property ${selector} ${property}`);
  return Number(match[1]);
}

export function htmlAttributeValues(source: string, attributeName: string): string[] {
  return [...source.matchAll(new RegExp(`\\b${attributeName}="([^"]+)"`, "gu"))]
    .map((match) => match[1])
    .filter((value): value is string => Boolean(value));
}

export function htmlElementById(source: string, id: string): HtmlAttributes {
  const element = htmlElements(source).find((attributes) => attributes.id === id);
  assert.ok(element, `missing HTML element #${id}`);
  return element;
}

export function htmlElements(source: string, tagName?: string): HtmlAttributes[] {
  const tagPattern = tagName ? escapeRegExp(tagName) : "[a-z][a-z0-9-]*";
  return [...source.matchAll(new RegExp(`<${tagPattern}\\b([\\s\\S]*?)>`, "giu"))].map((match) =>
    parseHtmlAttributes(match[1] || ""),
  );
}

export function rustEnumVariants(source: string, name: string): string[] {
  const match = new RegExp(`pub enum ${name} \\{([\\s\\S]*?)\\n\\}`, "u").exec(source);
  assert.ok(match?.[1], `missing Rust enum ${name}`);
  return [...match[1].matchAll(/^ {4}([A-Z][A-Za-z0-9]*),$/gmu)]
    .map((variant) => variant[1])
    .filter((variant): variant is string => Boolean(variant));
}

export function rustNumberConst(source: string, name: string): number {
  const match = new RegExp(`const ${name}: f64 = ([0-9.]+);`, "u").exec(source);
  assert.ok(match?.[1], `missing Rust numeric constant ${name}`);
  return Number(match[1]);
}

export function rustStringConst(source: string, name: string): string {
  const match = new RegExp(`pub const ${name}: &str = "([^"]+)";`, "u").exec(source);
  assert.ok(match?.[1], `missing Rust string constant ${name}`);
  return match[1];
}

export function rustTauriCommandNames(source: string): string[] {
  return [...source.matchAll(/#\[tauri::command\]\s*fn\s+([a-z0-9_]+)/gu)]
    .map((match) => match[1])
    .filter((name): name is string => Boolean(name));
}

export function cssRuleBody(source: string, selector: string): string {
  const normalizedSource = source.replace(/\r\n?/gu, "\n");
  const normalizedSelector = selector.replace(/\r\n?/gu, "\n");
  const match = new RegExp(`${escapeRegExp(normalizedSelector)}\\s*\\{([\\s\\S]*?)\\n\\}`, "u").exec(normalizedSource);
  assert.ok(match?.[1], `missing CSS rule ${selector}`);
  return match[1];
}

function parseHtmlAttributes(source: string): HtmlAttributes {
  return Object.fromEntries(
    [...source.matchAll(/\s([a-z][a-z0-9-:]*)(?:="([^"]*)")?/giu)]
      .map((match) => [match[1], match[2] ?? ""] as const)
      .filter(([name]) => Boolean(name)),
  );
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}
