import { applyAppearance, DEFAULT_CAPTION_OPACITY, opacityToSlider, sliderToOpacity } from "./appearance.js";
import type { PreparedBackground } from "./background-image.js";
import { errorMessage } from "./error-message.js";
import type { PreferencesStore, StoredBackground } from "./preferences.js";

export interface SettingsControllerElements {
  // The app shell doubles as the appearance target (CSS vars inherit down to the
  // caption strip) and the host for escape / outside-click close.
  root: HTMLElement;
  settingsButton: HTMLButtonElement;
  settingsPanel: HTMLElement;
  captionOpacity: HTMLInputElement;
  backgroundButton: HTMLButtonElement;
  backgroundFile: HTMLInputElement;
  backgroundClearButton: HTMLButtonElement;
  exportButton: HTMLButtonElement;
  settingsStatus: HTMLElement;
}

export interface SettingsControllerDeps {
  elements: SettingsControllerElements;
  preferences: PreferencesStore;
  buildTranscript(): string;
  copyText(text: string): Promise<void>;
  prepareBackground(file: Blob): Promise<PreparedBackground>;
  objectUrlFromStored(background: StoredBackground): string;
  revokeObjectUrl?(url: string): void;
}

export class SettingsController {
  private opacity = DEFAULT_CAPTION_OPACITY;
  private imageUrl: string | null = null;
  private open = false;

  constructor(private readonly deps: SettingsControllerDeps) {}

  init(): void {
    const prefs = this.deps.preferences.load();
    this.opacity = prefs.captionOpacity ?? DEFAULT_CAPTION_OPACITY;
    this.deps.elements.captionOpacity.value = String(opacityToSlider(this.opacity));
    this.restoreBackground();
    this.refreshAppearance();
    this.bind();
    this.setOpen(false);
  }

  private bind(): void {
    const { elements } = this.deps;
    elements.settingsButton.addEventListener("click", (event) => {
      event.stopPropagation();
      this.setOpen(!this.open);
    });
    elements.captionOpacity.addEventListener("input", () => this.handleOpacityInput());
    elements.backgroundButton.addEventListener("click", () => elements.backgroundFile.click());
    elements.backgroundFile.addEventListener("change", () => void this.handleBackgroundFile());
    elements.backgroundClearButton.addEventListener("click", () => this.clearBackground());
    elements.exportButton.addEventListener("click", () => void this.handleExport());
    elements.root.addEventListener("keydown", (event) => this.handleKeydown(event as KeyboardEvent));
    elements.root.addEventListener("pointerdown", (event) => this.handleRootPointerDown(event as PointerEvent));
  }

  private setOpen(open: boolean): void {
    this.open = open;
    this.deps.elements.settingsPanel.setAttribute("data-open", String(open));
    this.deps.elements.settingsButton.setAttribute("aria-expanded", String(open));
    if (!open) {
      this.setStatus("");
    }
  }

  private handleOpacityInput(): void {
    const slider = Number.parseInt(this.deps.elements.captionOpacity.value, 10);
    if (!Number.isFinite(slider)) {
      return;
    }
    this.opacity = sliderToOpacity(slider);
    this.refreshAppearance();
    this.deps.preferences.save({ captionOpacity: this.opacity });
  }

  private async handleBackgroundFile(): Promise<void> {
    const file = this.deps.elements.backgroundFile.files?.[0];
    // Reset so re-selecting the same file fires another change event.
    this.deps.elements.backgroundFile.value = "";
    if (!file) {
      return;
    }
    try {
      const prepared = await this.deps.prepareBackground(file);
      this.deps.preferences.saveBackground(prepared.stored);
      this.replaceImageUrl(prepared.objectUrl);
      this.refreshAppearance();
      this.setStatus("Background updated");
    } catch (error) {
      this.setStatus(backgroundError(error));
    }
  }

  private clearBackground(): void {
    this.deps.preferences.saveBackground(null);
    this.replaceImageUrl(null);
    this.refreshAppearance();
    this.setStatus("Background cleared");
  }

  private async handleExport(): Promise<void> {
    const text = this.deps.buildTranscript();
    if (!text) {
      this.setStatus("Nothing to copy yet");
      return;
    }
    try {
      await this.deps.copyText(text);
      this.setStatus("Copied to clipboard");
    } catch (error) {
      this.setStatus(`Copy failed: ${errorMessage(error)}`);
    }
  }

  private handleKeydown(event: KeyboardEvent): void {
    if (this.open && event.key === "Escape") {
      this.setOpen(false);
    }
  }

  private handleRootPointerDown(event: PointerEvent): void {
    if (!this.open) {
      return;
    }
    const target = event.target;
    if (target instanceof Element && target.closest("#settings-panel,#settings-button")) {
      return;
    }
    this.setOpen(false);
  }

  private restoreBackground(): void {
    const stored = this.deps.preferences.loadBackground();
    if (!stored) {
      return;
    }
    try {
      this.imageUrl = this.deps.objectUrlFromStored(stored);
    } catch {
      // A corrupt payload would fail on every launch; drop it so it self-heals.
      this.imageUrl = null;
      this.deps.preferences.saveBackground(null);
    }
  }

  private replaceImageUrl(next: string | null): void {
    if (this.imageUrl && this.imageUrl !== next) {
      this.deps.revokeObjectUrl?.(this.imageUrl);
    }
    this.imageUrl = next;
  }

  private refreshAppearance(): void {
    applyAppearance(this.deps.elements.root, { opacity: this.opacity, imageUrl: this.imageUrl });
  }

  private setStatus(text: string): void {
    this.deps.elements.settingsStatus.textContent = text;
  }
}

function backgroundError(error: unknown): string {
  return /quota|exceeded/iu.test(errorMessage(error)) ? "Image too large to save" : "Couldn't set background";
}
