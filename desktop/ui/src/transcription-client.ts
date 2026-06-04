import { errorMessage } from "./error-message.js";
import type { RealtimeEvent } from "./realtime-events.js";
import { parseTranscriptDocumentSnapshot, type TranscriptDocumentSnapshot } from "./transcription-document.js";
import { transcriptionStreamUrlFromRealtimeUrl } from "./server-url.js";

export interface OfflineTranscriptionRequest {
  file: Blob;
  language: string | null;
  onEvent?: (event: RealtimeEvent) => void;
  realtimeUrl: string;
  signal?: AbortSignal;
  targetLanguage: string;
}

export async function transcribeFile(request: OfflineTranscriptionRequest): Promise<TranscriptDocumentSnapshot> {
  const endpoint = transcriptionStreamUrlFromRealtimeUrl(request.realtimeUrl);
  if (!endpoint.ok) {
    throw new Error(endpoint.message);
  }

  const url = new URL(endpoint.url);
  const language = request.language?.trim();
  if (language) {
    url.searchParams.set("language", language);
  }
  const targetLanguage = request.targetLanguage.trim();
  if (targetLanguage) {
    url.searchParams.set("targetLanguage", targetLanguage);
  }
  const filename = fileName(request.file);
  if (filename) {
    url.searchParams.set("filename", filename);
  }

  const response = await fetch(url.href, {
    body: request.file,
    headers: {
      Accept: "application/x-ndjson",
      "Content-Type": request.file.type || "application/octet-stream",
    },
    method: "POST",
    signal: request.signal,
  });
  if (!response.ok) {
    const payload = await responseJson(response);
    throw new Error(errorPayloadMessage(payload) || `Transcription request failed (${response.status})`);
  }
  return await responseStreamSnapshot(response, request.onEvent);
}

function fileName(file: Blob): string {
  const value = (file as { name?: unknown }).name;
  return typeof value === "string" ? value.trim() : "";
}

async function responseJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function errorPayloadMessage(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  const error = (payload as { error?: unknown }).error;
  if (!error || typeof error !== "object") {
    return "";
  }
  const message = (error as { message?: unknown }).message;
  return typeof message === "string" ? message : "";
}

async function responseStreamSnapshot(
  response: Response,
  onEvent: ((event: RealtimeEvent) => void) | undefined,
): Promise<TranscriptDocumentSnapshot> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Transcription stream is not readable.");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let snapshot: TranscriptDocumentSnapshot | null = null;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parsed = consumeLines(buffer, onEvent);
      buffer = parsed.remaining;
      snapshot = parsed.snapshot ?? snapshot;
    }
    buffer += decoder.decode();
    const parsed = consumeLines(buffer ? `${buffer}\n` : buffer, onEvent);
    snapshot = parsed.snapshot ?? snapshot;
  } catch (error) {
    try {
      await reader.cancel(error);
    } catch {
      // The original stream error is more useful to the caller.
    }
    throw error;
  }

  if (!snapshot) {
    throw new Error("Transcription stream ended before a final transcript.");
  }
  return snapshot;
}

function consumeLines(
  input: string,
  onEvent: ((event: RealtimeEvent) => void) | undefined,
): { remaining: string; snapshot: TranscriptDocumentSnapshot | null } {
  const lines = input.split(/\r?\n/u);
  const remaining = lines.pop() ?? "";
  let snapshot: TranscriptDocumentSnapshot | null = null;
  for (const line of lines) {
    const value = line.trim();
    if (!value) {
      continue;
    }
    try {
      snapshot = processStreamPayload(JSON.parse(value), onEvent) ?? snapshot;
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(`Invalid transcription stream event: ${errorMessage(error)}`);
      }
      throw error;
    }
  }
  return { remaining, snapshot };
}

function processStreamPayload(
  payload: unknown,
  onEvent: ((event: RealtimeEvent) => void) | undefined,
): TranscriptDocumentSnapshot | null {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid transcription stream event: event must be an object.");
  }
  const event = payload as RealtimeEvent;
  if (event.type === "error") {
    throw new Error(streamErrorMessage(event) || "Transcription stream failed.");
  }
  onEvent?.(event);
  if (event.type !== "transcript_final") {
    return null;
  }
  try {
    return parseTranscriptDocumentSnapshot((event as { document?: unknown }).document);
  } catch (error) {
    throw new Error(`Invalid transcription response: ${errorMessage(error)}`);
  }
}

function streamErrorMessage(event: RealtimeEvent): string {
  const error = event.error;
  if (typeof error === "string") {
    return error;
  }
  if (!error || typeof error !== "object") {
    return "";
  }
  const message = (error as { message?: unknown }).message;
  return typeof message === "string" ? message : "";
}
