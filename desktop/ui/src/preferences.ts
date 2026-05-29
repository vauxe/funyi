import { isRecord } from "./runtime-guards.js";

// Persistent client preferences. The store is a tiny key/value abstraction so the
// controller can be unit-tested against an in-memory map, while the desktop build
// backs it with localStorage (which survives across launches in the Tauri webview).

export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface Preferences {
  serverUrl: string | null;
  asrLanguage: string | null;
  targetLanguage: string | null;
  audioSourceId: string | null;
  captionOpacity: number | null;
}

export interface StoredBackground {
  mime: string;
  data: string;
}

const PREFERENCES_KEY = "funyi.preferences";
const BACKGROUND_KEY = "funyi.background";

const EMPTY_PREFERENCES: Preferences = {
  serverUrl: null,
  asrLanguage: null,
  targetLanguage: null,
  audioSourceId: null,
  captionOpacity: null,
};

export class PreferencesStore {
  constructor(private readonly store: KeyValueStore) {}

  load(): Preferences {
    const record = this.readRecord(PREFERENCES_KEY);
    if (!record) {
      return { ...EMPTY_PREFERENCES };
    }
    return {
      serverUrl: optionalString(record.serverUrl),
      asrLanguage: optionalString(record.asrLanguage),
      targetLanguage: optionalString(record.targetLanguage),
      audioSourceId: optionalString(record.audioSourceId),
      captionOpacity: optionalNumber(record.captionOpacity),
    };
  }

  save(patch: Partial<Preferences>): void {
    const next: Preferences = { ...this.load(), ...patch };
    this.writeRecord(PREFERENCES_KEY, pruneNulls(next));
  }

  loadBackground(): StoredBackground | null {
    const record = this.readRecord(BACKGROUND_KEY);
    const mime = optionalString(record?.mime);
    const data = optionalString(record?.data);
    return mime && data ? { mime, data } : null;
  }

  // Quota errors from a large image are allowed to surface so the caller can report
  // them; clearing (null) is always best-effort.
  saveBackground(background: StoredBackground | null): void {
    if (!background) {
      this.remove(BACKGROUND_KEY);
      return;
    }
    this.store.setItem(BACKGROUND_KEY, JSON.stringify(background));
  }

  private readRecord(key: string): Record<string, unknown> | null {
    let raw: string | null;
    try {
      raw = this.store.getItem(key);
    } catch {
      return null;
    }
    if (!raw) {
      return null;
    }
    try {
      const parsed: unknown = JSON.parse(raw);
      return isRecord(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  private writeRecord(key: string, value: Record<string, unknown>): void {
    try {
      this.store.setItem(key, JSON.stringify(value));
    } catch {
      // Best-effort: a failed small-preferences write must never break the app.
    }
  }

  private remove(key: string): void {
    try {
      this.store.removeItem(key);
    } catch {
      // ignore
    }
  }
}

export class MemoryKeyValueStore implements KeyValueStore {
  private readonly map = new Map<string, string>();

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }
}

export function createPreferencesStore(): PreferencesStore {
  return new PreferencesStore(resolveKeyValueStore());
}

function resolveKeyValueStore(): KeyValueStore {
  const candidate = (globalThis as typeof globalThis & { localStorage?: KeyValueStore }).localStorage;
  if (candidate && typeof candidate.getItem === "function") {
    return candidate;
  }
  return new MemoryKeyValueStore();
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

function optionalNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function pruneNulls(prefs: Preferences): Record<string, unknown> {
  const record: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(prefs)) {
    if (value !== null) {
      record[key] = value;
    }
  }
  return record;
}
