import type { AudioAdapter } from "./audio-adapter.js";
import type { AppElements } from "./app-dom.js";
import { AudioSourceSelect, type SelectableAudioSource } from "./audio-source-select.js";
import type { AudioSource } from "./audio-source.js";
import { objectUrlFromStored, prepareBackgroundImage } from "./background-image.js";
import { CaptionView } from "./caption-view.js";
import { ChromeController } from "./chrome-controller.js";
import { errorMessage } from "./error-message.js";
import type { OverlayHost } from "./host-contract.js";
import { LanguageControls } from "./language-controls.js";
import { LiveSession } from "./live-session.js";
import { OverlayController } from "./overlay-controller.js";
import type { PreferencesStore } from "./preferences.js";
import { readyEventTranslationEnabled } from "./realtime-events.js";
import type { LiveSessionClient, LiveSessionClientCallbacks } from "./session-client.js";
import type { SessionState } from "./session-state.js";
import { buildSessionStartOptions, INVALID_AUDIO_SOURCE_MESSAGE } from "./session-start-options.js";
import { SessionControlsView } from "./session-controls-view.js";
import { NO_AUDIO_SOURCE_MESSAGE } from "./session-status.js";
import { SettingsController } from "./settings-controller.js";
import { StatusController } from "./status-controller.js";
import { SubtitleDocument } from "./subtitle-document.js";
import { copyToClipboard, formatTranscript } from "./transcript-export.js";
import { transcribeFile } from "./transcription-client.js";

export const OFFLINE_FILE_SOURCE_ID = "__funyi_offline_file__";

const OFFLINE_FILE_SOURCE: SelectableAudioSource = {
  detail: "Transcribe a local audio file.",
  id: OFFLINE_FILE_SOURCE_ID,
  isAvailable: true,
  kind: "file",
  name: "File",
};

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
  private readonly chromeController: ChromeController;
  private readonly languageControls: LanguageControls;
  private readonly liveSession: LiveSession;
  private readonly overlayController: OverlayController;
  private readonly preferences: PreferencesStore;
  private readonly sessionControlsView: SessionControlsView;
  private readonly settingsController: SettingsController;
  private readonly statusController: StatusController;
  private currentLiveAudioSourceId: string | null = null;
  private offlineAbort: AbortController | null = null;
  private offlineFileSelectionVersion = 0;
  private offlineTranscription: Promise<void> | null = null;
  private usedOfflineFileSelectionVersion = -1;
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
      sessionStatus: dom.sessionStatus,
      stopButton: dom.stopButton,
      transportButton: dom.transportButton,
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
    this.chromeController = new ChromeController({
      root: dom.appShell,
      // An open settings panel keeps the controls pinned: a pause mid-adjustment
      // should not tuck the panel's own toggle away beneath it.
      shouldStayVisible: () => this.settingsController.isOpen,
    });
  }

  async boot(): Promise<void> {
    const { dom } = this.options;
    this.languageControls.render();
    this.applyStoredLanguagePreferences();
    await this.populateAudioSources();
    this.applyStoredAudioSource();
    this.settingsController.init();
    this.chromeController.init();
    this.overlayController.bind();
    dom.transportButton.addEventListener("click", () => void this.toggleTransport());
    dom.stopButton.addEventListener("click", () => void this.stopSession());
    dom.minimizeButton.addEventListener("click", () => void this.overlayController.minimize());
    dom.closeButton.addEventListener("click", () => void this.closeOverlay());
    dom.serverUrl.addEventListener("change", () =>
      this.preferences.save({ serverUrl: dom.serverUrl.value.trim() || null }),
    );
    dom.audioSource.addEventListener("change", () => void this.applyAudioSourceChange());
    dom.offlineFile.addEventListener("change", () => this.applyOfflineFileChange());
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
    if (storedId && storedId !== OFFLINE_FILE_SOURCE_ID && this.audioSourceSelect.hasAvailableSource) {
      setSelectValueIfPresent(this.options.dom.audioSource, storedId, { requireEnabled: true });
    }
  }

  private async toggleTransport(): Promise<void> {
    const state = this.liveSession.getState();
    if (state === "idle") {
      await this.startSession();
      return;
    }
    if (state === "paused") {
      await this.liveSession.resume();
      return;
    }
    if (state === "running") {
      await this.liveSession.pause();
    }
  }

  private async stopSession(): Promise<void> {
    if (this.offlineAbort !== null) {
      await this.cancelOfflineTranscription();
      return;
    }
    await this.liveSession.stop();
  }

  private async closeOverlay(): Promise<void> {
    try {
      await this.cancelOfflineTranscription();
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
      this.audioSourceSelect.render([OFFLINE_FILE_SOURCE]);
      this.statusController.setStatus("captureStatus", errorMessage(error));
      this.liveSession.setAudioAvailable(true);
      return;
    }

    this.audioSourceSelect.render([...sources, OFFLINE_FILE_SOURCE]);
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
    const selectedKind = this.audioSourceSelect.selectedKind;
    if (selectedKind === "file") {
      await this.startFileTranscription();
      return;
    }
    const startOptions = buildSessionStartOptions({
      url: dom.serverUrl.value,
      audioSourceId: dom.audioSource.value,
      audioSourceKind: selectedKind,
      asrLanguage: this.languageControls.asrLanguage,
      targetLanguage: this.languageControls.targetLanguage,
    });
    if (!startOptions.ok) {
      this.statusController.setStatus("captureStatus", startOptions.message);
      return;
    }
    this.resetSessionState();
    if (await this.liveSession.start(startOptions.options)) {
      this.currentLiveAudioSourceId = startOptions.options.audioSourceId;
    }
  }

  private async applyAudioSourceChange(): Promise<void> {
    const { dom } = this.options;
    const audioSourceId = dom.audioSource.value || null;
    const audioSourceKind = this.audioSourceSelect.selectedKind;

    const state = this.liveSession.getState();
    if (state !== "running" && state !== "paused") {
      if (audioSourceKind === "file") {
        this.openOfflineFilePicker();
        return;
      }
      if (audioSourceKind !== null) {
        this.preferences.save({ audioSourceId });
      }
      return;
    }

    if (audioSourceKind === "file") {
      this.restoreCurrentLiveAudioSource();
      this.statusController.setStatus("captureStatus", "File transcription unavailable while captions are running.");
      return;
    }
    if (!audioSourceId || !audioSourceKind) {
      this.statusController.setStatus("captureStatus", INVALID_AUDIO_SOURCE_MESSAGE);
      return;
    }

    await this.liveSession.switchAudioSource({ audioSourceId, audioSourceKind });
    this.currentLiveAudioSourceId = audioSourceId;
    this.preferences.save({ audioSourceId });
  }

  private resetSessionState(): void {
    this.subtitleDocument = new SubtitleDocument({ translationEnabled: this.languageControls.translationEnabled });
    this.captionView.reset();
    this.statusController.setStatus("captureStatus", "");
    this.statusController.setStatus("connectionStatus", "");
    this.liveSession.resetStats();
    this.render();
  }

  private applyOfflineFileChange(): void {
    if (this.audioSourceSelect.selectedKind !== "file") {
      return;
    }
    const fileSelected = Boolean(this.selectedOfflineFile());
    if (fileSelected) {
      this.offlineFileSelectionVersion += 1;
      this.statusController.setStatus("captureStatus", "");
    }
    this.statusController.setStatus("connectionStatus", fileSelected ? "File selected." : "");
  }

  private async startFileTranscription(): Promise<void> {
    if (this.offlineTranscription !== null) {
      return;
    }
    const file = this.selectedOfflineFile();
    if (!file || this.usedOfflineFileSelectionVersion === this.offlineFileSelectionVersion) {
      this.openOfflineFilePicker();
      this.statusController.setStatus("connectionStatus", "Choose an audio file.");
      return;
    }

    const abort = new AbortController();
    this.usedOfflineFileSelectionVersion = this.offlineFileSelectionVersion;
    this.offlineAbort = abort;
    this.resetSessionState();
    this.setControlsState("connecting", { canStart: false });
    this.statusController.setStatus("connectionStatus", "Transcribing file...");

    const transcription = this.runFileTranscription(file, abort);
    this.offlineTranscription = transcription;
    await transcription;
  }

  private async runFileTranscription(file: Blob, abort: AbortController): Promise<void> {
    try {
      const snapshot = await transcribeFile({
        file,
        language: this.languageControls.asrLanguage,
        realtimeUrl: this.options.dom.serverUrl.value,
        signal: abort.signal,
        targetLanguage: this.languageControls.targetLanguage,
      });
      if (this.offlineAbort !== abort) {
        return;
      }
      this.subtitleDocument.replaceSnapshot(snapshot);
      this.render();
      this.statusController.setStatus("connectionStatus", "File transcript ready.");
    } catch (error) {
      if (this.offlineAbort !== abort) {
        return;
      }
      this.statusController.setStatus(
        "connectionStatus",
        abort.signal.aborted ? "File transcription cancelled." : errorMessage(error),
      );
    } finally {
      if (this.offlineAbort === abort) {
        this.offlineAbort = null;
        this.offlineTranscription = null;
        this.setControlsState("idle", { canStart: this.audioSourceSelect.hasAvailableSource });
      }
    }
  }

  private async cancelOfflineTranscription(): Promise<void> {
    const transcription = this.offlineTranscription;
    if (this.offlineAbort !== null) {
      this.offlineAbort.abort();
    }
    await transcription;
  }

  private selectedOfflineFile(): Blob | null {
    const files = this.options.dom.offlineFile.files;
    return files && files.length > 0 ? files[0] || null : null;
  }

  private openOfflineFilePicker(): void {
    this.options.dom.offlineFile.value = "";
    this.options.dom.offlineFile.click();
  }

  private restoreCurrentLiveAudioSource(): void {
    if (this.currentLiveAudioSourceId) {
      setSelectValueIfPresent(this.options.dom.audioSource, this.currentLiveAudioSourceId, { requireEnabled: true });
    }
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
    this.chromeController.setSessionState(state);
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
