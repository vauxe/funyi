import type { StoredBackground } from "./preferences.js";

// Browser-only helpers for the caption background image. A user-chosen file is
// downscaled before it is persisted so the base64 payload stays small enough for
// localStorage, and it is displayed through a blob: URL (allowed by the Tauri CSP,
// unlike data: URLs).

const MAX_DIMENSION = 1600;
const OUTPUT_MIME = "image/jpeg";
const OUTPUT_QUALITY = 0.82;
const ALLOWED_MIME = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);

export interface PreparedBackground {
  stored: StoredBackground;
  objectUrl: string;
}

export async function prepareBackgroundImage(file: Blob): Promise<PreparedBackground> {
  const bitmap = await createImageBitmap(file);
  const blob = await downscaleToBlob(bitmap);
  const stored: StoredBackground = { mime: blob.type || OUTPUT_MIME, data: await blobToBase64(blob) };
  return { stored, objectUrl: URL.createObjectURL(blob) };
}

export function objectUrlFromStored(background: StoredBackground): string {
  const bytes = base64ToBytes(background.data);
  return URL.createObjectURL(new Blob([bytes], { type: resolveStoredMime(background.mime) }));
}

// Constrain the content type to known image MIMEs so a tampered store entry cannot
// mint a blob served under the app origin with an arbitrary type.
export function resolveStoredMime(mime: string): string {
  return ALLOWED_MIME.has(mime) ? mime : OUTPUT_MIME;
}

export function base64ToBytes(base64: string): Uint8Array<ArrayBuffer> {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

async function downscaleToBlob(bitmap: ImageBitmap): Promise<Blob> {
  try {
    const scale = Math.min(1, MAX_DIMENSION / Math.max(bitmap.width, bitmap.height, 1));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Canvas 2D context unavailable");
    }
    context.drawImage(bitmap, 0, 0, width, height);
    return await canvasToBlob(canvas);
  } finally {
    bitmap.close();
  }
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Failed to encode background image"))),
      OUTPUT_MIME,
      OUTPUT_QUALITY,
    );
  });
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read image"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Unexpected image read result"));
        return;
      }
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(blob);
  });
}
