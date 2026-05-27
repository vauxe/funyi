type BrowserGlobalName = "document" | "Element" | "HTMLElement" | "WebSocket" | "window";

export function clearBrowserGlobals(...names: BrowserGlobalName[]): void {
  for (const name of names) {
    Reflect.deleteProperty(globalThis, name);
  }
}

export function clearDomGlobals(): void {
  clearBrowserGlobals("document", "Element", "HTMLElement");
}
