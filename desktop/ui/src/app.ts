import { AsrClient } from "./asr-client.js";
import { LiveSession, type RealtimeEvent, type SessionState } from "./live-session.js";
import { SubtitleDocument } from "./subtitle-document.js";
import {
  decodeBase64Pcm,
  listAudioSources,
  listenAudioCaptureErrors,
  listenAudioFrames,
  startAudioCapture,
  stopAudioCapture,
} from "./native-audio.js";

type StatusKey = "connectionStatus" | "readyStatus" | "captureStatus" | "audioStats";

const dom = {
  serverUrl: requireElement<HTMLInputElement>("#server-url"),
  language: requireElement<HTMLInputElement>("#language"),
  context: requireElement<HTMLInputElement>("#context"),
  audioSource: requireElement<HTMLSelectElement>("#audio-source"),
  translationEnabled: requireElement<HTMLInputElement>("#translation-enabled"),
  startButton: requireElement<HTMLButtonElement>("#start-button"),
  stopButton: requireElement<HTMLButtonElement>("#stop-button"),
  flushButton: requireElement<HTMLButtonElement>("#flush-button"),
  exportSrtButton: requireElement<HTMLButtonElement>("#export-srt-button"),
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
  dom.startButton.addEventListener("click", () => void startSession());
  dom.stopButton.addEventListener("click", () => void liveSession.stop());
  dom.flushButton.addEventListener("click", () => liveSession.flush());
  dom.exportSrtButton.addEventListener("click", exportSrt);
  dom.translationEnabled.addEventListener("change", () => {
    subtitleDocument.setTranslationEnabled(dom.translationEnabled.checked);
    render();
  });
  render();
}

async function populateAudioSources(): Promise<void> {
  const sources = await listAudioSources();
  hasAvailableAudioSource = sources.some((source) => source.isAvailable);
  dom.audioSource.replaceChildren();
  for (const source of sources) {
    const option = document.createElement("option");
    option.value = source.id;
    option.textContent = source.isAvailable ? source.name : `${source.name} (unavailable)`;
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
      context: dom.context.value,
      translation: dom.translationEnabled.checked,
    },
  });
}

function resetSessionState(): void {
  subtitleDocument = new SubtitleDocument({ translationEnabled: dom.translationEnabled.checked });
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
  dom.exportSrtButton.disabled = subtitleDocument.stableLines.length === 0;
}

function renderCaptionLine(
  line: ReturnType<SubtitleDocument["window"]>["current"],
  sourceElement: HTMLElement,
  translationElement: HTMLElement,
): void {
  sourceElement.textContent = line?.text || "";
  translationElement.textContent = line?.translation || "";
}

function renderHistory(): void {
  dom.historyList.replaceChildren();
  for (const line of subtitleDocument.stableLines) {
    const item = document.createElement("article");
    item.className = "history-item";

    const time = document.createElement("div");
    time.className = "history-time";
    time.textContent = formatRange(line.startMs, line.endMs, line.timingStatus);

    const source = document.createElement("div");
    source.className = "history-source";
    source.textContent = line.text;

    const translation = document.createElement("div");
    translation.className = "history-translation";
    translation.textContent = line.translation || line.translationMessage || "";

    item.append(time, source, translation);
    dom.historyList.append(item);
  }
}

function exportSrt(): void {
  const srt = subtitleDocument.toSrt();
  const blob = new Blob([srt], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `funyi-${new Date().toISOString().replaceAll(":", "-")}.srt`;
  link.click();
  URL.revokeObjectURL(url);
}

function setControlsState(state: SessionState, { canStart }: { canStart: boolean }): void {
  const active = state === "connecting" || state === "running" || state === "finishing";
  dom.startButton.disabled = !canStart;
  dom.stopButton.disabled = state === "idle";
  dom.flushButton.disabled = state !== "running";
  dom.serverUrl.disabled = active;
  dom.language.disabled = active;
  dom.context.disabled = active;
  dom.audioSource.disabled = active;
  dom.translationEnabled.disabled = active;
}

function setStatus(key: StatusKey, value: string): void {
  dom[key].textContent = value || "";
}

function readySummary(event: RealtimeEvent): string {
  const translationInfo = isRecord(event.translation) && event.translation.enabled
    ? `, translation ${String(event.translation.target_language || "")}`
    : "";
  return `ready ${String(event.sample_rate || "")} Hz${translationInfo}`;
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
