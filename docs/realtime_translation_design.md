# Realtime Translation Design

Status: source transcript events exist; translation runtime does not.

Source transcript semantics live in `@docs/realtime_asr_service.md`. Translation
subscribes to source events and must not change `Qwen3ASRModel`,
`RealtimeASRSession`, or `TranscriptStore`.

## Boundary

| Component | Owns |
|---|---|
| `TranslationRuntime` | preview debounce, stable queue, stale-drop, timeouts |
| `HYMTTranslator` | model load, prompt, generation |
| `realtime_server.py` | transport, event order, serialized sends |

Do not put translation in `RealtimeASRSession` or source transcript segments.

## Protocol Additions

`ready.translation` when enabled:

- `enabled`
- `target_language`
- `model`
- `preview`
- `stable`

Events:

- `translation_preview`: replace-style translation of current source `partial`.
- `translation_stable`: append-style translation keyed by stable source
  `segment_id`.
- `translation_status`: `timeout`, `failed`, `skipped_backlog`, or
  `stale_revision`.

Rules:

- send source `transcript_update` before translating its stable appends;
- drop preview results whose `source_revision` is stale;
- preview failures are silent;
- stable translation uses bounded FIFO and explicit status on timeout/overflow;
- never expose stack traces or model internals.

## Scheduler

- HY-MT is bounded text-to-text generation, not streaming decode.
- Run at most one generation at a time in v1.
- Preview: debounce 800-1200ms, latest-only, `max_new_tokens=96-128`.
- Stable: queue size 4, timeout 5s, `max_new_tokens=256`.
- Stable work has priority over preview work that has not started.

## Finish

Translation-enabled `finish` order:

```text
stop preview scheduling
drop queued preview work
run session.finish()
send finish-created transcript_update events
translate only finish-created stable_appends, with timeout
send translation_stable or translation_status for those segments
send transcript_final
close WebSocket
```

Do not wait for old stable-translation backlog. ASR-only mode keeps current
`session.finish()` behavior.

## Sending

Use one sender task and one send lock:

```text
ASR/session task -> event_queue
translation worker -> event_queue
sender task -> websocket.send_text under send_lock
```

Never call `websocket.send_text` concurrently.

## Translator

`HYMTTranslator.translate(text, *, target_language, source_language="",
max_new_tokens=256) -> str` should be synchronous and small. Load once at
startup, fail fast if the local model is unavailable, and never download weights
from request handling.

Do not document exact prompt text as a contract until validated with the chosen
checkpoint.

## Implementation

1. Add `TranslationRuntime`.
2. Add startup flags: target language, model path, preview/stable toggles, queue
   size, timeout.
3. Add fake-translator unit tests for stale preview, queue order, timeout,
   overflow, finish backlog, and send serialization.
4. Add real-model smoke only after a smoke tool exists.
5. Add WebSocket E2E comparing ASR-only and ASR+translation latency, quality,
   GPU memory, and shutdown cleanup.

Private audio, transcripts, and generated translation outputs stay in
`local_data/` and `local_goldens/`.
