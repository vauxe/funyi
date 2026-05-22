import { AsrClient } from "./asr-client.js";
import { LiveSession, type RealtimeEvent, type SessionState } from "./live-session.js";
import { SubtitleDocument, type SubtitleLine } from "./subtitle-document.js";
import {
  decodeBase64Pcm,
  listAudioSources,
  listenAudioCaptureErrors,
  listenAudioFrames,
  startAudioCapture,
  stopAudioCapture,
} from "./native-audio.js";

type StatusKey = "connectionStatus" | "readyStatus" | "captureStatus" | "audioStats";
type OverlayMode = "compact" | "history";
type StatusTone = "idle" | "active" | "ok" | "warn" | "error";

const dom = {
  appShell: requireElement<HTMLElement>("#app-shell"),
  captionStrip: requireElement<HTMLElement>("#caption-strip"),
  serverUrl: requireElement<HTMLInputElement>("#server-url"),
  language: requireElement<HTMLInputElement>("#language"),
  audioSource: requireElement<HTMLSelectElement>("#audio-source"),
  translationEnabled: requireElement<HTMLInputElement>("#translation-enabled"),
  sessionButton: requireElement<HTMLButtonElement>("#session-button"),
  historyButton: requireElement<HTMLButtonElement>("#history-button"),
  closeButton: requireElement<HTMLButtonElement>("#close-button"),
  connectionStatus: requireElement<HTMLSpanElement>("#connection-status"),
  readyStatus: requireElement<HTMLSpanElement>("#ready-status"),
  captureStatus: requireElement<HTMLSpanElement>("#capture-status"),
  audioStats: requireElement<HTMLSpanElement>("#audio-stats"),
  previousSource: requireElement<HTMLDivElement>("#previous-source"),
  previousTranslation: requireElement<HTMLDivElement>("#previous-translation"),
  currentSource: requireElement<HTMLDivElement>("#current-source"),
  currentTranslation: requireElement<HTMLDivElement>("#current-translation"),
  historyList: requireElement<HTMLElement>("#history-list"),
};

let subtitleDocument = new SubtitleDocument();
let hasAvailableAudioSource = false;
let overlayMode: OverlayMode = "compact";
let sessionState: SessionState = "idle";
let activeDragPointerId: number | null = null;
let activeDragSurface: HTMLElement | null = null;
let pendingDragFrame: number | null = null;
let connectionStatusOwner: "overlay" | "session" | null = null;
let renderedHistoryLines: SubtitleLine[] = [];
let renderedHistoryTranslationEnabled = subtitleDocument.translationEnabled;
let overlayModeChanging = false;
const statusValues: Record<StatusKey, string> = {
  connectionStatus: "",
  readyStatus: "",
  captureStatus: "",
  audioStats: "",
};
const liveSession = new LiveSession({
  createClient: ({ url, ...callbacks }) => new AsrClient({ url, ...callbacks }),
  audio: {
    decodePcm: decodeBase64Pcm,
    listenCaptureErrors: listenAudioCaptureErrors,
    listenFrames: listenAudioFrames,
    startCapture: startAudioCapture,
    stopCapture: stopAudioCapture,
  },
  onReady: (event) => setStatus("readyStatus", readySummary(event)),
  onStateChange: setControlsState,
  onStatus: (key, value) => setStatus(key as StatusKey, value),
  onTranscriptEvent: (event) => {
    subtitleDocument.applyEvent(event);
    render();
  },
});

async function boot(): Promise<void> {
  await populateAudioSources();
  dom.captionStrip.addEventListener("pointerdown", (event) => void handleDragPointerDown(event, dom.captionStrip));
  dom.sessionButton.addEventListener("click", () => void toggleSession());
  dom.historyButton.addEventListener("click", () => void setOverlayMode(overlayMode === "history" ? "compact" : "history"));
  dom.closeButton.addEventListener("click", () => void closeOverlay());
  dom.translationEnabled.addEventListener("change", () => {
    subtitleDocument.setTranslationEnabled(dom.translationEnabled.checked);
    render();
  });
  render();
}

async function handleDragPointerDown(event: PointerEvent, surface: HTMLElement): Promise<void> {
  if (event.button !== 0 || isInteractiveTarget(event.target)) {
    return;
  }
  event.preventDefault();
  activeDragPointerId = event.pointerId;
  activeDragSurface = surface;
  dom.appShell.classList.add("is-dragging");
  activeDragSurface.setPointerCapture(event.pointerId);
  window.addEventListener("pointermove", handleDragPointerMove);
  window.addEventListener("pointerup", handleDragPointerEnd);
  window.addEventListener("pointercancel", handleDragPointerEnd);

  try {
    await invokeOverlayCommand("start_overlay_drag");
    clearOverlayCommandError();
  } catch (error) {
    setOverlayCommandError(error);
    clearActiveDrag();
  }
}

function handleDragPointerMove(event: PointerEvent): void {
  if (event.pointerId !== activeDragPointerId) {
    return;
  }
  scheduleDragUpdate();
}

function handleDragPointerEnd(event: PointerEvent): void {
  if (event.pointerId !== activeDragPointerId) {
    return;
  }
  void finishOverlayDrag();
}

async function finishOverlayDrag(): Promise<void> {
  try {
    cancelScheduledDragUpdate();
    await invokeOverlayCommand("update_overlay_drag");
    await invokeOverlayCommand("end_overlay_drag");
    clearOverlayCommandError();
  } catch (error) {
    setOverlayCommandError(error);
  } finally {
    clearActiveDrag();
  }
}

function clearActiveDrag(): void {
  if (activeDragPointerId === null) {
    return;
  }
  if (activeDragSurface?.hasPointerCapture(activeDragPointerId)) {
    activeDragSurface.releasePointerCapture(activeDragPointerId);
  }
  activeDragPointerId = null;
  activeDragSurface = null;
  cancelScheduledDragUpdate();
  dom.appShell.classList.remove("is-dragging");
  window.removeEventListener("pointermove", handleDragPointerMove);
  window.removeEventListener("pointerup", handleDragPointerEnd);
  window.removeEventListener("pointercancel", handleDragPointerEnd);
}

function scheduleDragUpdate(): void {
  if (pendingDragFrame !== null) {
    return;
  }

  const run = (): void => {
    pendingDragFrame = null;
    if (activeDragPointerId === null) {
      return;
    }
    void invokeOverlayCommand("update_overlay_drag").catch((error: unknown) => {
      setOverlayCommandError(error);
      clearActiveDrag();
    });
  };

  pendingDragFrame = typeof requestAnimationFrame === "function"
    ? requestAnimationFrame(run)
    : window.setTimeout(run, 16);
}

function cancelScheduledDragUpdate(): void {
  if (pendingDragFrame === null) {
    return;
  }
  if (typeof cancelAnimationFrame === "function") {
    cancelAnimationFrame(pendingDragFrame);
  } else {
    window.clearTimeout(pendingDragFrame);
  }
  pendingDragFrame = null;
}

async function setOverlayMode(mode: OverlayMode): Promise<void> {
  const previousMode = overlayMode;
  if (mode === previousMode || overlayModeChanging) {
    return;
  }

  overlayModeChanging = true;
  applyOverlayMode(mode);
  try {
    await invokeOverlayCommand("set_overlay_mode", { mode });
    clearOverlayCommandError();
  } catch (error) {
    applyOverlayMode(previousMode);
    setOverlayCommandError(error);
  } finally {
    overlayModeChanging = false;
  }
}

function applyOverlayMode(mode: OverlayMode): void {
  overlayMode = mode;
  dom.appShell.setAttribute("data-overlay-mode", mode);
  syncModeButton(dom.historyButton, mode === "history", "Hide history", "Show history");
  if (mode === "history") {
    scrollHistoryToLatest("smooth");
  }
}

function syncModeButton(button: HTMLButtonElement, expanded: boolean, expandedLabel: string, collapsedLabel: string): void {
  const label = expanded ? expandedLabel : collapsedLabel;
  button.classList.toggle("is-expanded", expanded);
  button.title = label;
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", String(expanded));
}

async function toggleSession(): Promise<void> {
  if (sessionState === "idle") {
    await startSession();
    return;
  }
  await liveSession.stop();
}

async function closeOverlay(): Promise<void> {
  try {
    await liveSession.stop({ sendFinish: false });
    await invokeOverlayCommand("close_overlay");
  } catch (error) {
    setOverlayCommandError(error);
  }
}

async function populateAudioSources(): Promise<void> {
  const sources = await listAudioSources();
  hasAvailableAudioSource = sources.some((source) => source.isAvailable);
  dom.audioSource.replaceChildren();
  for (const source of sources) {
    const option = document.createElement("option");
    option.value = source.id;
    option.textContent = source.isAvailable ? audioSourceLabel(source) : `${audioSourceLabel(source)} off`;
    option.disabled = !source.isAvailable;
    option.title = source.detail || "";
    dom.audioSource.append(option);
  }
  if (!hasAvailableAudioSource) {
    setStatus("captureStatus", sources[0]?.detail || "No native audio source available.");
  }
  liveSession.setAudioAvailable(hasAvailableAudioSource);
}

async function startSession(): Promise<void> {
  if (!hasAvailableAudioSource) {
    setStatus("captureStatus", "No native audio source available.");
    return;
  }
  if (!liveSession.canStart()) {
    return;
  }
  resetSessionState();
  await liveSession.start({
    url: dom.serverUrl.value.trim(),
    audioSourceId: dom.audioSource.value,
    startPayload: {
      type: "start",
      session_id: `desktop-${Date.now()}`,
      sample_rate: 16000,
      audio_format: "pcm_s16le",
      language: dom.language.value.trim() || undefined,
      translation: dom.translationEnabled.checked,
    },
  });
}

function resetSessionState(): void {
  subtitleDocument = new SubtitleDocument({ translationEnabled: dom.translationEnabled.checked });
  resetRenderedHistory();
  setStatus("readyStatus", "");
  setStatus("captureStatus", "");
  liveSession.resetStats();
  render();
}

function render(): void {
  const windowState = subtitleDocument.window();
  renderCaptionLine(windowState.previous, dom.previousSource, dom.previousTranslation);
  renderCaptionLine(windowState.current, dom.currentSource, dom.currentTranslation);
  renderHistory();
}

function renderCaptionLine(
  line: ReturnType<SubtitleDocument["window"]>["current"],
  sourceElement: HTMLElement,
  translationElement: HTMLElement,
): void {
  setTextIfChanged(sourceElement, line?.text || "");
  setTextIfChanged(translationElement, line?.translation || "");
}

function renderHistory(): void {
  const lines = subtitleDocument.stableLines;
  const hadNewLine = lines.length > renderedHistoryLines.length;
  const translationChanged = renderedHistoryTranslationEnabled !== subtitleDocument.translationEnabled;

  if (lines.length < renderedHistoryLines.length) {
    dom.historyList.replaceChildren();
    renderedHistoryLines = [];
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line) {
      continue;
    }
    const item = dom.historyList.children[index] as HTMLElement | undefined;
    const historyItem = item || createHistoryItem();
    if (!item) {
      dom.historyList.append(historyItem);
    }
    if (translationChanged || renderedHistoryLines[index] !== line) {
      updateHistoryItem(historyItem, line);
    }
    historyItem.classList.toggle("is-latest", index === lines.length - 1);
  }
  renderedHistoryLines = lines.slice();
  renderedHistoryTranslationEnabled = subtitleDocument.translationEnabled;
  if (overlayMode === "history" && hadNewLine) {
    scrollHistoryToLatest("smooth");
  }
}

function resetRenderedHistory(): void {
  dom.historyList.replaceChildren();
  renderedHistoryLines = [];
  renderedHistoryTranslationEnabled = subtitleDocument.translationEnabled;
}

function createHistoryItem(): HTMLElement {
  const item = document.createElement("article");
  item.className = "history-item";

  const time = document.createElement("div");
  time.className = "history-time";

  const source = document.createElement("div");
  source.className = "history-source";

  const translation = document.createElement("div");
  translation.className = "history-translation";

  item.append(time, source, translation);
  return item;
}

function updateHistoryItem(item: HTMLElement, line: SubtitleLine): void {
  const [time, source, translation] = Array.from(item.children) as HTMLElement[];
  if (time) {
    setTextIfChanged(time, formatRange(line.startMs, line.endMs, line.timingStatus));
  }
  if (source) {
    setTextIfChanged(source, line.text);
  }
  if (translation) {
    setTextIfChanged(
      translation,
      subtitleDocument.translationEnabled
        ? line.translation || line.translationMessage || ""
        : "",
    );
  }
}

function setTextIfChanged(element: HTMLElement, value: string): void {
  if (element.textContent !== value) {
    element.textContent = value;
  }
}

function scrollHistoryToLatest(behavior: ScrollBehavior): void {
  const target = dom.historyList.lastElementChild;
  if (typeof HTMLElement === "undefined" || !(target instanceof HTMLElement)) {
    return;
  }
  const scroll = (): void => target.scrollIntoView({ behavior, block: "end" });
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(scroll);
    return;
  }
  scroll();
}

function setControlsState(state: SessionState, { canStart }: { canStart: boolean }): void {
  const active = state === "connecting" || state === "running" || state === "finishing";
  sessionState = state;
  dom.appShell.setAttribute("data-state", state);
  dom.sessionButton.disabled = state === "idle" && !canStart;
  dom.sessionButton.classList.toggle("is-stop", active);
  dom.sessionButton.classList.toggle("is-finishing", state === "finishing");
  dom.sessionButton.title = active ? "Stop" : "Start";
  dom.sessionButton.setAttribute(
    "aria-label",
    state === "finishing" ? "Cancel final transcript" : active ? "Stop" : "Start",
  );
  dom.serverUrl.disabled = active;
  dom.language.disabled = active;
  dom.audioSource.disabled = active;
  dom.translationEnabled.disabled = active;
}

function setStatus(key: StatusKey, value: string): void {
  if (key === "connectionStatus" && connectionStatusOwner === "overlay") {
    connectionStatusOwner = value ? "session" : null;
  }
  statusValues[key] = value || "";
  renderStatusIndicator(key);
}

function setOverlayCommandError(error: unknown): void {
  connectionStatusOwner = "overlay";
  statusValues.connectionStatus = errorMessage(error);
  renderStatusIndicator("connectionStatus");
}

function clearOverlayCommandError(): void {
  if (connectionStatusOwner !== "overlay") {
    return;
  }
  connectionStatusOwner = null;
  statusValues.connectionStatus = "";
  renderStatusIndicator("connectionStatus");
}

function renderStatusIndicator(key: StatusKey): void {
  const element = dom[key];
  const value = statusValues[key];
  const { active, tone, level } = statusPresentation(key, value);
  element.textContent = active ? value : "";
  element.dataset.active = String(active);
  element.dataset.tone = tone;
  if (level) {
    element.dataset.level = level;
  } else {
    delete element.dataset.level;
  }
  element.title = value;
  element.setAttribute("aria-label", active ? element.textContent || value : "");
}

function statusPresentation(
  key: StatusKey,
  value: string,
): { active: boolean; tone: StatusTone; level?: string } {
  if (!value) {
    return { active: false, tone: "idle" };
  }
  if (key === "audioStats") {
    const level = audioLevelState(value);
    return {
      active: true,
      tone: level === "live" ? "ok" : level === "low" ? "warn" : "idle",
      level,
    };
  }
  if (key === "connectionStatus") {
    return { active: true, tone: statusTextHasError(value) ? "error" : "ok" };
  }
  if (key === "captureStatus") {
    return {
      active: true,
      tone: statusTextHasError(value) ? "error" : /silent/i.test(value) ? "warn" : "active",
    };
  }
  return { active: true, tone: "ok" };
}

function audioLevelState(value: string): "silent" | "low" | "live" {
  if (/^silent$/i.test(value)) {
    return "silent";
  }
  const match = value.match(/(-?\d+)dB/i);
  const level = match ? Number.parseInt(match[1] || "", 10) : Number.NaN;
  if (!Number.isFinite(level) || level < -60) {
    return "silent";
  }
  return level < -42 ? "low" : "live";
}

function audioSourceLabel(source: { kind?: string; name: string }): string {
  const prefix = source.kind === "microphone" ? "Mic" : source.kind === "system" ? "Sys" : "Src";
  const name = source.name
    .replace(/\s*\((default )?microphone\)\s*$/i, "")
    .replace(/\s*\(ScreenCaptureKit\)\s*$/i, "")
    .replace(/^System audio$/i, "Audio");
  return `${prefix} · ${name}`;
}

function statusTextHasError(value: string): boolean {
  return /error|failed|closed|timeout|timed out|lost|unavailable|no native|no audio/i.test(value);
}

function readySummary(event: RealtimeEvent): string {
  const sampleRate = Number(event.sample_rate || 0);
  const readyText = sampleRate > 0 ? `${sampleRate / 1000}k` : "Ready";
  return isRecord(event.translation) && event.translation.enabled
    ? `${readyText} · 译`
    : readyText;
}

function formatRange(startMs: number | null, endMs: number | null, status: string | null): string {
  const prefix = isInteger(startMs) && isInteger(endMs)
    ? `${formatClock(startMs)} - ${formatClock(endMs)}`
    : "pending";
  return status ? `${prefix} ${status}` : prefix;
}

function formatClock(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const millis = Math.floor(ms % 1000);
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

boot();

function requireElement<TElement extends Element>(selector: string): TElement {
  const element = document.querySelector<TElement>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function isInteger(value: unknown): value is number {
  return Number.isInteger(value);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function invokeOverlayCommand(command: string, args?: Record<string, unknown>): Promise<void> {
  await window.__TAURI__?.core.invoke(command, args);
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  return target instanceof Element
    && Boolean(target.closest("button,input,select,textarea,a"));
}
