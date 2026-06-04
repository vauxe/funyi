import { errorMessage } from "./error-message.js";
import { parseTranscriptDocumentSnapshot, type TranscriptDocumentSnapshot } from "./transcription-document.js";
import { transcriptionUrlFromRealtimeUrl } from "./server-url.js";

export interface OfflineTranscriptionRequest {
  file: Blob;
  language: string | null;
  realtimeUrl: string;
  signal?: AbortSignal;
  targetLanguage: string;
}

export async function transcribeFile(request: OfflineTranscriptionRequest): Promise<TranscriptDocumentSnapshot> {
  const endpoint = transcriptionUrlFromRealtimeUrl(request.realtimeUrl);
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
      "Content-Type": request.file.type || "application/octet-stream",
    },
    method: "POST",
    signal: request.signal,
  });
  const payload = await responseJson(response);
  if (!response.ok) {
    throw new Error(errorPayloadMessage(payload) || `Transcription request failed (${response.status})`);
  }
  try {
    return parseTranscriptDocumentSnapshot(payload);
  } catch (error) {
    throw new Error(`Invalid transcription response: ${errorMessage(error)}`);
  }
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
