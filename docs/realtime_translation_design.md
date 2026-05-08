# Realtime Translation Design

Status: source transcript events and a synchronous HY-MT adapter exist;
WebSocket translation runtime is not wired yet.

Goal: add optional subtitle translation above `/ws/asr` without changing source
ASR semantics. ASR-only mode must keep its current behavior.

Source transcript semantics live in `@docs/realtime_asr_service.md`.

## Scope

v1 translates stable source segments only. It does not translate source
`partial`; preview translation can be added later after stable translation is
validated.

Do not change `Qwen3ASRModel`, `RealtimeASRSession`, `TranscriptStore`, or the
source transcript segment schema.

## Components

| Component | Owns |
|---|---|
| `TranslationRuntime` | stable queue, overflow, timeout, stale-drop |
| `HYMTTranslator` | model load, prompt, generation |
| `realtime_server.py` | transport, event order, serialized sends |

## Protocol Additions

`ready.translation` when enabled:

- `enabled`
- `target_language`
- `model`
- `stable`: `{ "enabled": true, "queue_size": 4, "timeout_ms": 5000 }`
- `preview`: `{ "enabled": false }`

Events:

```json
{
  "type": "translation_stable",
  "source_revision": 12,
  "source_segment_id": "seg_000001",
  "source_segment_index": 1,
  "target_language": "English",
  "text": "..."
}
```

```json
{
  "type": "translation_status",
  "scope": "stable",
  "code": "timeout",
  "source_revision": 12,
  "source_segment_id": "seg_000001",
  "source_segment_index": 1,
  "target_language": "English",
  "message": "translation timed out"
}
```

`translation_status.code` is one of:

- `timeout`
- `failed`
- `skipped_backlog`

Rules:

- send source `transcript_update` before translating its stable appends;
- `translation_stable` is keyed by the source `segment_id`;
- clients must tolerate translation events arriving after later source updates;
- every dropped stable segment gets `translation_status`;
- never expose stack traces or model internals.

## Scheduler

- HY-MT is bounded text-to-text generation, not streaming decode.
- Run at most one generation at a time in v1.
- Stable queue size defaults to 4.
- If the queue is full, drop the oldest queued segment, emit
  `skipped_backlog` for it, then enqueue the newest segment.
- Each stable translation has an end-to-end timeout, including time spent waiting
  for the generation slot.
- Timeout or translator failure emits `translation_status` and the worker moves
  to the next queued segment.
- `max_new_tokens` defaults to 512, matching `Qwen3ASRModel`.
- HY-MT loads with `attn_implementation="sdpa"` by default; generation length
  and sampling parameters stay unchanged.
- The default decode backend uses a fixed-shape static-cache loop to avoid
  per-token input and mask concatenation; `generate` remains available for
  fallback comparisons.

## Finish

Translation-enabled `finish` order:

```text
run session.finish()
split returned events into transcript_update events and transcript_final
send finish-created transcript_update events first
mark old queued stable translations skipped_backlog
translate only finish-created stable_appends, each with timeout
send translation_stable or translation_status for those finish-created segments
send transcript_final
close WebSocket
```

Do not wait for old stable-translation backlog. If a previous generation is
already running, finish-created translations still use their timeout budget; if
the generation slot does not become available in time, emit `timeout` for those
finish-created segments. ASR-only mode keeps current `session.finish()` behavior.

## Sending

Use one sender task:

```text
ASR/session task -> event_queue
translation worker -> event_queue
sender task -> websocket.send_text
```

Never call `websocket.send_text` outside the sender task.

## Translator

`HYMTTranslator.translate(text, *, target_language, source_language="",
max_new_tokens=512) -> str` should be synchronous and small. Load once at
startup, fail fast if the local model is unavailable, and never download weights
from request handling.

Startup config should include target language, model path, device, queue size,
timeout, and a stable-translation enable flag. Do not enable translation unless
the model has loaded successfully.

Do not document exact prompt text as a contract until validated with the chosen
checkpoint.

## Quality Gate

Translation changes must be validated against a fixed local case set covering
core subtitles, formatted text, ASR noise, and domain smoke cases.

The gate must check:

- no new quality errors versus the current accepted baseline;
- target language, empty output, length outliers, repetition loops, and required
  structural markers;
- `must_preserve` items such as protocol labels, fixed UI strings, numbers,
  units, and subtitle cue ids;
- per-case reference similarity, with a regression error when a candidate drops
  meaningfully below a baseline that has the metric.

Generation or prompt changes are not accepted on speed alone. If a change
improves formatting but regresses core meaning, keep the old prompt/generation
path and record the failed experiment outside git.

## Implementation

1. Add `TranslationRuntime`.
2. Add startup flags: target language, model path, device, stable enable flag,
   queue size, and timeout.
3. Add fake-translator unit tests for queue order, timeout, overflow, finish
   backlog, and send serialization.
4. Add real-model smoke only after a smoke tool exists.
5. Add WebSocket E2E comparing ASR-only and ASR+translation latency, quality,
   GPU memory, and shutdown cleanup.

Private audio, transcripts, and generated translation outputs stay in
`local_data/` and `local_goldens/`.

## Later Preview

Preview translation may be added after v1. It must be latest-only, debounced,
keyed by `source_revision`, and must silently drop stale results before sending.
