import type { AudioAdapter } from "./audio-adapter.js";
import type { AppElements } from "./app-dom.js";
import { AudioSourceSelect } from "./audio-source-select.js";
import type { AudioSource } from "./audio-source.js";
import { objectUrlFromStored, prepareBackgroundImage } from "./background-image.js";
import { CaptionView } from "./caption-view.js";
import { errorMessage } from "./error-message.js";
import type { OverlayHost } from "./host-contract.js";
import { LanguageControls } from "./language-controls.js";
import { LiveSession } from "./live-session.js";
import { OverlayController } from "./overlay-controller.js";
import type { PreferencesStore } from "./preferences.js";
import { readyEventTranslationEnabled } from "./realtime-events.js";
import type { LiveSessionClient, LiveSessionClientCallbacks } from "./session-client.js";
import type { SessionState } from "./session-state.js";
import { buildSessionStartOptions } from "./session-start-options.js";
import { SessionControlsView } from "./session-controls-view.js";
import { NO_AUDIO_SOURCE_MESSAGE } from "./session-status.js";
import { SettingsController } from "./settings-controller.js";
import { StatusController } from "./status-controller.js";
import { SubtitleDocument } from "./subtitle-document.js";
import { copyToClipboard, formatTranscript } from "./transcript-export.js";

export interface FunyiAppOptions {
  audio: AudioAdapter;
  createClient(options: LiveSessionClientCallbacks): LiveSessionClient;
  dom: AppElements;
  overlay: OverlayHost;
  preferences: PreferencesStore;
}

export class FunyiApp {
  private readonly audioSourceSelect: AudioSourceSelect;
  private readonly captionView: CaptionView;
  private readonly languageControls: LanguageControls;
  private readonly liveSession: LiveSession;
  private readonly overlayController: OverlayController;
  private readonly preferences: PreferencesStore;
  private readonly sessionControlsView: SessionControlsView;
  private readonly settingsController: SettingsController;
  private readonly statusController: StatusController;
  private subtitleDocument = new SubtitleDocument();

  constructor(private readonly options: FunyiAppOptions) {
    const { audio, createClient, dom, overlay, preferences } = options;
    this.preferences = preferences;
    this.audioSourceSelect = new AudioSourceSelect(dom.audioSource);
    this.languageControls = new LanguageControls(dom.language, dom.translationTargetLanguage);
    this.sessionControlsView = new SessionControlsView({
      appShell: dom.appShell,
      audioSource: dom.audioSource,
      language: dom.language,
      serverUrl: dom.serverUrl,
      sessionButton: dom.sessionButton,
      sessionStatus: dom.sessionStatus,
      translationTargetLanguage: dom.translationTargetLanguage,
      volumeIndicator: dom.volumeIndicator,
    });
    this.statusController = new StatusController({
      render: (summary) => this.sessionControlsView.renderStatus(summary),
    });
    this.captionView = new CaptionView({
      currentSource: dom.currentSource,
      currentTranslation: dom.currentTranslation,
      historyList: dom.historyList,
      announcer: dom.captionAnnouncer,
    });
    this.overlayController = new OverlayController(
      overlay,
      {
        root: dom.appShell,
        dragSurface: dom.captionStrip,
        resizeHandles: dom.resizeHandles,
      },
      {
        onClearError: () => this.statusController.clearOverlayError(),
        onError: (error) => this.statusController.setOverlayError(error),
        onModeApplied: (mode) => {
          if (mode === "history") {
            this.captionView.scrollHistoryToLatest("auto");
          }
        },
      },
    );
    this.liveSession = new LiveSession({
      createClient,
      audio,
      onReady: (event) => {
        this.subtitleDocument.setTranslationEnabled(readyEventTranslationEnabled(event));
        this.render();
      },
      onStateChange: (state, detail) => this.setControlsState(state, detail),
      onStatus: (key, value) => this.statusController.setStatus(key, value),
      onTranscriptEvent: (event) => {
        this.subtitleDocument.applyEvent(event);
        this.render();
      },
    });
    this.settingsController = new SettingsController({
      elements: {
        root: dom.appShell,
        settingsButton: dom.settingsButton,
        settingsPanel: dom.settingsPanel,
        captionOpacity: dom.captionOpacity,
        captionOpacityValue: dom.captionOpacityValue,
        backgroundButton: dom.backgroundButton,
        backgroundFile: dom.backgroundFile,
        backgroundClearButton: dom.backgroundClearButton,
        exportButton: dom.exportButton,
        settingsStatus: dom.settingsStatus,
      },
      preferences: this.preferences,
      buildTranscript: () =>
        formatTranscript(this.captionView.collectTranscriptLines(), {
          translationEnabled: this.languageControls.translationEnabled,
        }),
      copyText: copyToClipboard,
      prepareBackground: prepareBackgroundImage,
      objectUrlFromStored,
      revokeObjectUrl: (url) => URL.revokeObjectURL(url),
    });
  }

  async boot(): Promise<void> {
    const { dom } = this.options;
    this.languageControls.render();
    this.applyStoredLanguagePreferences();
    await this.populateAudioSources();
    this.applyStoredAudioSource();
    this.settingsController.init();
    this.overlayController.bind();
    dom.sessionButton.addEventListener("click", () => void this.toggleSession());
    dom.minimizeButton.addEventListener("click", () => void this.overlayController.minimize());
    dom.closeButton.addEventListener("click", () => void this.closeOverlay());
    dom.serverUrl.addEventListener("change", () =>
      this.preferences.save({ serverUrl: dom.serverUrl.value.trim() || null }),
    );
    dom.audioSource.addEventListener("change", () =>
      this.preferences.save({ audioSourceId: dom.audioSource.value || null }),
    );
    dom.language.addEventListener("change", () => {
      // ASR language does not change what is displayed, so no re-render here.
      this.preferences.save({ asrLanguage: this.languageControls.asrLanguage });
      this.liveSession.setLanguageConfig({ language: this.languageControls.asrLanguage });
    });
    dom.translationTargetLanguage.addEventListener("change", () => this.applyTranslationTarget());
    this.render();
  }

  private applyStoredLanguagePreferences(): void {
    // Set values directly without dispatching `change`: session start and rendering
    // read languageControls live, so restored values take effect without re-saving
    // or firing the persistence handlers.
    const { dom } = this.options;
    const prefs = this.preferences.load();
    if (prefs.serverUrl) {
      dom.serverUrl.value = prefs.serverUrl;
    }
    if (prefs.asrLanguage) {
      setSelectValueIfPresent(dom.language, prefs.asrLanguage);
    }
    if (prefs.targetLanguage) {
      setSelectValueIfPresent(dom.translationTargetLanguage, prefs.targetLanguage);
    }
  }

  private applyStoredAudioSource(): void {
    const storedId = this.preferences.load().audioSourceId;
    if (storedId && this.audioSourceSelect.hasAvailableSource) {
      setSelectValueIfPresent(this.options.dom.audioSource, storedId, { requireEnabled: true });
    }
  }

  private async toggleSession(): Promise<void> {
    if (this.liveSession.getState() === "idle") {
      await this.startSession();
      return;
    }
    await this.liveSession.stop();
  }

  private async closeOverlay(): Promise<void> {
    try {
      await this.liveSession.stop({ sendFinish: false });
      await this.overlayController.close();
    } catch (error) {
      this.statusController.setOverlayError(error);
    }
  }

  private async populateAudioSources(): Promise<void> {
    let sources: AudioSource[];
    try {
      sources = await this.options.audio.listSources();
    } catch (error) {
      this.audioSourceSelect.render([]);
      this.statusController.setStatus("captureStatus", errorMessage(error));
      this.liveSession.setAudioAvailable(false);
      return;
    }

    this.audioSourceSelect.render(sources);
    if (!this.audioSourceSelect.hasAvailableSource) {
      this.statusController.setStatus(
        "captureStatus",
        this.audioSourceSelect.unavailableDetail || NO_AUDIO_SOURCE_MESSAGE,
      );
    }
    this.liveSession.setAudioAvailable(this.audioSourceSelect.hasAvailableSource);
  }

  private async startSession(): Promise<void> {
    const { dom } = this.options;
    if (!this.audioSourceSelect.hasAvailableSource) {
      this.statusController.setStatus("captureStatus", NO_AUDIO_SOURCE_MESSAGE);
      return;
    }
    if (!this.liveSession.canStart()) {
      return;
    }
    const startOptions = buildSessionStartOptions({
      url: dom.serverUrl.value,
      audioSourceId: dom.audioSource.value,
      audioSourceKind: this.audioSourceSelect.selectedKind,
      asrLanguage: this.languageControls.asrLanguage,
      targetLanguage: this.languageControls.targetLanguage,
    });
    if (!startOptions.ok) {
      this.statusController.setStatus("captureStatus", startOptions.message);
      return;
    }
    this.resetSessionState();
    await this.liveSession.start(startOptions.options);
  }

  private resetSessionState(): void {
    this.subtitleDocument = new SubtitleDocument({ translationEnabled: this.languageControls.translationEnabled });
    this.captionView.reset();
    this.statusController.setStatus("captureStatus", "");
    this.liveSession.resetStats();
    this.render();
  }

  private applyTranslationTarget(): void {
    this.preferences.save({ targetLanguage: this.languageControls.targetLanguage || null });
    this.subtitleDocument.setTranslationEnabled(this.languageControls.translationEnabled);
    this.liveSession.setLanguageConfig({ target_language: this.languageControls.targetLanguage || null });
    this.render();
  }

  private render(): void {
    this.captionView.render(this.subtitleDocument, {
      historyVisible: this.overlayController.mode === "history",
      translationLanguage: this.languageControls.targetLanguage,
    });
  }

  private setControlsState(state: SessionState, { canStart }: { canStart: boolean }): void {
    this.sessionControlsView.renderState(state, { canStart });
    this.statusController.setSessionState(state);
  }
}

export function createFunyiApp(options: FunyiAppOptions): FunyiApp {
  return new FunyiApp(options);
}

// Restore a stored select value only when the option still exists (the language
// lists or available audio sources may have changed between launches).
function setSelectValueIfPresent(
  select: HTMLSelectElement,
  value: string,
  { requireEnabled = false }: { requireEnabled?: boolean } = {},
): void {
  const present = Array.from(select.children).some((child) => {
    const option = child as HTMLOptionElement;
    return option.value === value && (!requireEnabled || !option.disabled);
  });
  if (present) {
    select.value = value;
  }
}
