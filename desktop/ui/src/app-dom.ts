import { isResizeDirection, RESIZE_HANDLE_SELECTOR, type ResizeHandle } from "./overlay-contract.js";

export const APP_ELEMENT_SELECTORS = {
  appShell: "#app-shell",
  audioSource: "#audio-source",
  backgroundButton: "#background-button",
  backgroundClearButton: "#background-clear-button",
  backgroundFile: "#background-file",
  captionAnnouncer: "#caption-announcer",
  captionOpacity: "#caption-opacity",
  captionOpacityValue: "#caption-opacity-value",
  captionStrip: "#caption-strip",
  closeButton: "#close-button",
  currentSource: "#current-source",
  currentTranslation: "#current-translation",
  exportButton: "#export-button",
  historyList: "#history-list",
  language: "#language",
  minimizeButton: "#minimize-button",
  serverUrl: "#server-url",
  sessionButton: "#session-button",
  sessionStatus: "#session-status",
  settingsButton: "#settings-button",
  settingsPanel: "#settings-panel",
  settingsStatus: "#settings-status",
  translationTargetLanguage: "#translation-target-language",
  volumeIndicator: "#volume-indicator",
} as const;

export interface AppElements {
  appShell: HTMLElement;
  captionAnnouncer: HTMLElement;
  captionStrip: HTMLElement;
  serverUrl: HTMLInputElement;
  language: HTMLSelectElement;
  translationTargetLanguage: HTMLSelectElement;
  audioSource: HTMLSelectElement;
  sessionButton: HTMLButtonElement;
  minimizeButton: HTMLButtonElement;
  closeButton: HTMLButtonElement;
  sessionStatus: HTMLSpanElement;
  volumeIndicator: HTMLElement;
  currentSource: HTMLDivElement;
  currentTranslation: HTMLDivElement;
  historyList: HTMLElement;
  settingsButton: HTMLButtonElement;
  settingsPanel: HTMLElement;
  captionOpacity: HTMLInputElement;
  captionOpacityValue: HTMLOutputElement;
  backgroundButton: HTMLButtonElement;
  backgroundClearButton: HTMLButtonElement;
  backgroundFile: HTMLInputElement;
  exportButton: HTMLButtonElement;
  settingsStatus: HTMLElement;
  resizeHandles: ResizeHandle[];
}

export function getAppElements(): AppElements {
  return {
    appShell: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.appShell),
    captionAnnouncer: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.captionAnnouncer),
    captionStrip: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.captionStrip),
    serverUrl: requireElement<HTMLInputElement>(APP_ELEMENT_SELECTORS.serverUrl),
    language: requireElement<HTMLSelectElement>(APP_ELEMENT_SELECTORS.language),
    translationTargetLanguage: requireElement<HTMLSelectElement>(APP_ELEMENT_SELECTORS.translationTargetLanguage),
    audioSource: requireElement<HTMLSelectElement>(APP_ELEMENT_SELECTORS.audioSource),
    sessionButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.sessionButton),
    minimizeButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.minimizeButton),
    closeButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.closeButton),
    sessionStatus: requireElement<HTMLSpanElement>(APP_ELEMENT_SELECTORS.sessionStatus),
    volumeIndicator: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.volumeIndicator),
    currentSource: requireElement<HTMLDivElement>(APP_ELEMENT_SELECTORS.currentSource),
    currentTranslation: requireElement<HTMLDivElement>(APP_ELEMENT_SELECTORS.currentTranslation),
    historyList: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.historyList),
    settingsButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.settingsButton),
    settingsPanel: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.settingsPanel),
    captionOpacity: requireElement<HTMLInputElement>(APP_ELEMENT_SELECTORS.captionOpacity),
    captionOpacityValue: requireElement<HTMLOutputElement>(APP_ELEMENT_SELECTORS.captionOpacityValue),
    backgroundButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.backgroundButton),
    backgroundClearButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.backgroundClearButton),
    backgroundFile: requireElement<HTMLInputElement>(APP_ELEMENT_SELECTORS.backgroundFile),
    exportButton: requireElement<HTMLButtonElement>(APP_ELEMENT_SELECTORS.exportButton),
    settingsStatus: requireElement<HTMLElement>(APP_ELEMENT_SELECTORS.settingsStatus),
    resizeHandles: collectResizeHandles(),
  };
}

function requireElement<TElement extends Element>(selector: string): TElement {
  const element = document.querySelector<TElement>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function collectResizeHandles(): ResizeHandle[] {
  const handles = Array.from(document.querySelectorAll<HTMLElement>(RESIZE_HANDLE_SELECTOR));
  if (handles.length === 0) {
    throw new Error(`Missing required resize handles: ${RESIZE_HANDLE_SELECTOR}`);
  }

  return handles.map((element) => {
    const direction = element.dataset.resizeDirection;
    if (!isResizeDirection(direction)) {
      throw new Error(`Invalid resize direction: ${direction || "(empty)"}`);
    }
    return { element, direction };
  });
}
