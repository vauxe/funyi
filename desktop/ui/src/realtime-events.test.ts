import test from "node:test";
import assert from "node:assert/strict";

import { parseRealtimeEventMessage, readyEventTranslationEnabled } from "./realtime-events.js";

test("parses realtime events only from JSON objects", () => {
  assert.deepEqual(parseRealtimeEventMessage('{"type":"ready"}'), { type: "ready" });
  assert.throws(() => parseRealtimeEventMessage("null"), /event payload must be an object/);
  assert.throws(() => parseRealtimeEventMessage("[]"), /event payload must be an object/);
  assert.throws(() => parseRealtimeEventMessage('"ready"'), /event payload must be an object/);
  assert.throws(() => parseRealtimeEventMessage('{"type":7}'), /event type must be a string/);
});

test("reads translation enabled from ready events defensively", () => {
  assert.equal(readyEventTranslationEnabled({ type: "ready", translation: { enabled: true } }), true);
  assert.equal(readyEventTranslationEnabled({ type: "ready", translation: { enabled: false } }), false);
  assert.equal(readyEventTranslationEnabled({ type: "ready", translation: null }), false);
  assert.equal(readyEventTranslationEnabled({ type: "ready" }), false);
});
