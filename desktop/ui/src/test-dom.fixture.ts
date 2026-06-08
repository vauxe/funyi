type Listener = (event?: unknown) => void;

export class FakeElement {
  attributes = new Map<string, string>();
  blurred = false;
  checked = false;
  children: FakeElement[] = [];
  className = "";
  clicks = 0;
  dataset: Record<string, string> = {};
  disabled = false;
  files: Blob[] | null = null;
  listeners = new Map<string, Listener[]>();
  pointerCapture: number | null = null;
  parentElement: FakeElement | null = null;
  scrollCalls: Array<{ behavior?: ScrollBehavior; block?: ScrollLogicalPosition }> = [];
  scrollHeight = 0;
  scrollTop = 0;
  styleValues = new Map<string, string>();
  textContent = "";
  title = "";
  value = "";

  readonly classList = {
    add: (name: string): void => {
      const classes = this.classSet();
      classes.add(name);
      this.className = [...classes].join(" ");
    },
    remove: (name: string): void => {
      const classes = this.classSet();
      classes.delete(name);
      this.className = [...classes].join(" ");
    },
    toggle: (name: string, enabled: boolean): void => {
      const classes = this.classSet();
      if (enabled) {
        classes.add(name);
      } else {
        classes.delete(name);
      }
      this.className = [...classes].join(" ");
    },
  };

  readonly style = {
    getPropertyValue: (name: string): string => this.styleValues.get(name) || "",
    setProperty: (name: string, value: string): void => {
      this.styleValues.set(name, value);
    },
  };

  constructor(
    readonly tagName = "div",
    readonly id = "",
  ) {}

  get lastElementChild(): FakeElement | null {
    return this.children.at(-1) || null;
  }

  addEventListener(type: string, listener: Listener): void {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  append(...children: FakeElement[]): void {
    for (const child of children) {
      child.parentElement = this;
    }
    this.children.push(...children);
    if (this.tagName === "select" && !this.value && children[0]?.value) {
      this.value = children[0].value;
    }
  }

  blur(): void {
    this.blurred = true;
  }

  click(): void {
    this.clicks += 1;
    this.dispatch("click", {});
  }

  closest(selector: string): FakeElement | null {
    const selectors = selector.split(",").map((item) => item.trim().toLowerCase());
    let element: FakeElement | null = this;
    while (element) {
      const current = element;
      if (selectors.some((item) => current.matchesClosestSelector(item))) {
        return element;
      }
      element = element.parentElement;
    }
    return null;
  }

  dispatch(type: string, event: unknown): void {
    for (const listener of this.listeners.get(type) || []) {
      listener(event);
    }
  }

  hasPointerCapture(pointerId: number): boolean {
    return this.pointerCapture === pointerId;
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  releasePointerCapture(pointerId: number): void {
    if (this.pointerCapture === pointerId) {
      this.pointerCapture = null;
    }
  }

  replaceChildren(...children: FakeElement[]): void {
    for (const child of this.children) {
      child.parentElement = null;
    }
    this.children = [];
    if (this.tagName === "select") {
      this.value = "";
    }
    this.append(...children);
  }

  scrollIntoView(options: { behavior?: ScrollBehavior; block?: ScrollLogicalPosition }): void {
    this.scrollCalls.push(options);
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    const datasetKey = dataAttributeKey(name);
    if (datasetKey) {
      this.dataset[datasetKey] = value;
    }
  }

  setPointerCapture(pointerId: number): void {
    this.pointerCapture = pointerId;
  }

  private classSet(): Set<string> {
    return new Set(this.className.split(/\s+/).filter(Boolean));
  }

  private matchesClosestSelector(selector: string): boolean {
    if (selector.startsWith("[") && selector.endsWith("]")) {
      return this.attributes.has(selector.slice(1, -1));
    }
    if (selector.startsWith("#")) {
      const id = selector.slice(1);
      return this.id === id || this.attributes.get("id") === id;
    }
    return this.tagName.toLowerCase() === selector;
  }
}

function dataAttributeKey(name: string): string | null {
  if (!name.startsWith("data-")) {
    return null;
  }
  return name.slice("data-".length).replace(/-([a-z])/g, (_match, letter: string) => letter.toUpperCase());
}

export class FakeDocument {
  activeElement: FakeElement | null = null;

  constructor(readonly elements: Record<string, FakeElement> = {}) {}

  createElement(tagName: string): FakeElement {
    return new FakeElement(tagName);
  }

  querySelector(selector: string): FakeElement | null {
    return selector.startsWith("#") ? this.elements[selector.slice(1)] || null : null;
  }

  querySelectorAll(selector: string): FakeElement[] {
    const datasetKey = dataSelectorKey(selector);
    if (!datasetKey) {
      return [];
    }
    return Object.values(this.elements).filter((element) => datasetKey in element.dataset);
  }
}

function dataSelectorKey(selector: string): string | null {
  if (!/^\[data-[a-z0-9-]+\]$/u.test(selector)) {
    return null;
  }
  return dataAttributeKey(selector.slice(1, -1));
}

export function installFakeDocument(document: FakeDocument = new FakeDocument()): FakeDocument {
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: document,
    writable: true,
  });
  return document;
}

export function installedFakeDocument(): FakeDocument {
  return globalThis.document as unknown as FakeDocument;
}

export function installFakeElementConstructors(): void {
  Object.defineProperty(globalThis, "Element", {
    configurable: true,
    value: FakeElement,
    writable: true,
  });
  Object.defineProperty(globalThis, "HTMLElement", {
    configurable: true,
    value: FakeElement,
    writable: true,
  });
}

export function asDomElement<TElement extends Element = HTMLElement>(element: FakeElement): TElement {
  return element as unknown as TElement;
}
