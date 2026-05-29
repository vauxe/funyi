import test from "node:test";
import assert from "node:assert/strict";

import { type KeyValueStore, MemoryKeyValueStore, PreferencesStore } from "./preferences.js";

test("returns null defaults when nothing is stored", () => {
  const store = new PreferencesStore(new MemoryKeyValueStore());

  assert.deepEqual(store.load(), {
    serverUrl: null,
    asrLanguage: null,
    targetLanguage: null,
    audioSourceId: null,
    captionOpacity: null,
  });
  assert.equal(store.loadBackground(), null);
});

test("merges partial saves and round-trips through storage", () => {
  const backing = new MemoryKeyValueStore();
  const store = new PreferencesStore(backing);

  store.save({ serverUrl: "ws://127.0.0.1:9000/ws/asr" });
  store.save({ targetLanguage: "Japanese", captionOpacity: 0.5 });

  const reloaded = new PreferencesStore(backing).load();
  assert.equal(reloaded.serverUrl, "ws://127.0.0.1:9000/ws/asr");
  assert.equal(reloaded.targetLanguage, "Japanese");
  assert.equal(reloaded.captionOpacity, 0.5);
});

test("null patches clear a previously stored value", () => {
  const store = new PreferencesStore(new MemoryKeyValueStore());

  store.save({ targetLanguage: "Japanese" });
  store.save({ targetLanguage: null });

  assert.equal(store.load().targetLanguage, null);
});

test("ignores malformed records and wrong-typed fields", () => {
  const backing = new MemoryKeyValueStore();
  backing.setItem("funyi.preferences", "not json");
  assert.equal(new PreferencesStore(backing).load().serverUrl, null);

  backing.setItem("funyi.preferences", JSON.stringify({ serverUrl: 5, captionOpacity: "x" }));
  const loaded = new PreferencesStore(backing).load();
  assert.equal(loaded.serverUrl, null);
  assert.equal(loaded.captionOpacity, null);
});

test("stores and clears the background image payload", () => {
  const store = new PreferencesStore(new MemoryKeyValueStore());

  store.saveBackground({ mime: "image/jpeg", data: "AAAA" });
  assert.deepEqual(store.loadBackground(), { mime: "image/jpeg", data: "AAAA" });

  store.saveBackground(null);
  assert.equal(store.loadBackground(), null);
});

test("a failed small-preferences write never throws but a background quota error surfaces", () => {
  const throwing: KeyValueStore = {
    getItem: () => null,
    setItem: () => {
      throw new Error("QuotaExceededError");
    },
    removeItem: () => {},
  };
  const store = new PreferencesStore(throwing);

  assert.doesNotThrow(() => store.save({ serverUrl: "ws://127.0.0.1:8000/ws/asr" }));
  assert.throws(() => store.saveBackground({ mime: "image/jpeg", data: "AAAA" }), /QuotaExceededError/);
});
