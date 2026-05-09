# Realtime Translation Pipeline

Status: source transcript events, a synchronous HY-MT adapter, the optional
WebSocket translation runtime, and the subtitle replay model are wired.
Model, prompt, or decode-path changes still need the translation quality gate.

Goal: add optional bilingual subtitles above `/ws/asr` without changing ASR
behavior. ASR-only mode keeps the current `ready`, `transcript_update`, and
`transcript_final` contract. Source transcript semantics stay in
`@docs/realtime_asr_service.md`.

## Boundary

Translation is a service-layer side track. It consumes emitted transcript events
and does not modify `Qwen3ASRModel`, `RealtimeASRSession`, `TranscriptStore`, or
the source segment schema.

v1 has two paths:

- stable: `stable_appends -> stable queue -> TranslationModelActor ->
  translation_stable`;
- preview: `partial -> latest-only debounce -> TranslationModelActor ->
  translation_preview`.

Stable translations are durable. Preview translations are temporary, replaceable,
and never enter history or export.

Out of scope:

- token-level HY-MT streaming;
- cross-request prefix cache;
- translation batching;
- ASR backend or stabilization changes.

## Flow

```text
audio frames
  -> RealtimeASRSession
  -> transcript_update / transcript_final
  -> update TranslationRuntime scheduler state without waiting for the model
  -> send source events immediately

TranslationRuntime
  -> TranslationModelActor
  -> HYMTTranslator.translate(...)
  -> event_queue

sender task
  -> websocket.send_text(...)

client SubtitleDocument
  -> previous stable line / current draft line / SRT history
```

Only the sender task writes to the WebSocket.

When `cuda_graph` is enabled, the service prewarms the ASR CUDA graph for the
fixed live20 profile before accepting WebSocket sessions. Prewarm failure is a
startup failure. Runtime ASR then replays the captured graph while HY-MT can
generate concurrently. If a request exceeds the prewarmed graph shape, ASR falls
back to non-graph decode for that call instead of capturing a new graph next to
HY-MT.

When translation is enabled, the service also prewarms HY-MT after the ASR
prewarm and before accepting WebSocket sessions. HY-MT warmup covers short,
medium, and long subtitle-shaped texts on the same single model actor executor
thread used at runtime. HY-MT warmup failure is a startup failure.

## Protocol

Session start uses service defaults. If the service has translation configured,
the client can disable translation for one session:

```json
{"type":"start","session_id":"local","translation":false}
```

Equivalent object form:

```json
{"type":"start","session_id":"local","translation":{"enabled":false}}
```

If `translation.target_language` is provided, it must match the service target
language for this single-model service.

`ready.translation` when enabled:

```json
{
  "enabled": true,
  "target_language": "English",
  "model": "tencent/HY-MT1.5-1.8B",
  "stable": { "enabled": true, "reliable": true, "queue_size": null, "timeout_ms": null },
  "preview": { "enabled": true, "debounce_ms": 700, "timeout_ms": 30000 }
}
```

`ready.translation` when the service has translation capability but the session
disabled it:

```json
{
  "enabled": false,
  "available": true,
  "target_language": "English"
}
```

Stable result:

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

Preview result:

```json
{
  "type": "translation_preview",
  "source_revision": 13,
  "target_language": "English",
  "text": "..."
}
```

Stable failure:

```json
{
  "type": "translation_status",
  "scope": "stable",
  "code": "failed",
  "source_revision": 12,
  "source_segment_id": "seg_000001",
  "source_segment_index": 1,
  "target_language": "English",
  "message": "translation failed"
}
```

`translation_status.code`: `failed`.
Never expose stack traces or model internals.

## Runtime Rules

For every `transcript_update`:

```text
update translation scheduler state
if partial exists: update_preview(revision, partial)
else: cancel_preview(revision)
enqueue each stable_appends item
send source event
```

Stable:

- stable history is reliable: every source stable segment gets exactly one
  `translation_stable` before `transcript_final`, unless the translator fails;
- stable generation runs one job at a time through the model actor;
- stable jobs are not dropped for backlog pressure and are not timed out by the
  service;
- translator failures emit `translation_status` for the affected segment.

Preview:

- debounce defaults to 700 ms;
- latest-only slot, no queue;
- preview generation is best-effort and goes through the same model actor;
- a newer `source_revision` makes older preview work stale;
- `transcript_update` without `partial` cancels older preview work;
- stale preview results are dropped silently;
- timed-out or finish-canceled preview model calls may keep running in the
  model actor thread, but their results are ignored.

Client replay:

- `SubtitleDocument` replays server events into one local document;
- `stable_appends` append immutable history;
- `partial` replaces the current draft line;
- `translation_stable` annotates a stable line by source segment id/index;
- `translation_preview` annotates the current draft only when `source_revision`
  matches;
- the compact subtitle window is `stable_lines[-1]` above `current` below;
- SRT/detail output uses stable history only, with translation as a second line
  in the same cue when translation display is enabled.

Scheduling:

- audio ingest and source event sending never wait for translation;
- service startup prewarms ASR graph capture before accepting sessions;
- translation startup prewarms HY-MT before accepting sessions;
- runtime ASR graph replay can overlap HY-MT generation;
- runtime ASR does not capture a new graph next to HY-MT; oversize requests
  fall back to non-graph decode;
- preview has priority over normal stable backlog because it is the lowest
  latency translation path;
- preview work that has not entered the model can be superseded by newer state;
- a preview model call that already entered HY-MT cannot be preempted and may
  delay later stable history or `finish`;
- stable backlog runs when no preview is ready and is drained during `finish`;
- finish-created stable jobs have priority during `finish`;
- if a stable job is already running when preview arrives, do not cancel the
  model actor call; drop the preview if it is stale by completion time;
- stable and preview share one `TranslationModelActor`, so the same translator
  instance is never entered concurrently;
- preview timeouts only discard the result; they do not interrupt the model
  call already running on the actor;
- if `--no-cuda-graph-prewarm` is used with translation, HY-MT calls share the
  CUDA graph capture lock to avoid runtime capture races; the validated default
  path is the prewarmed graph path, where actor serialization is enough for
  HY-MT model ownership.

## Finish

Translation-enabled `finish`:

```text
run session.finish()
send finish-created transcript_update events
enter translation finish mode
cancel pending or logically running preview work
wait for any already running stable translation to publish once
translate finish-created stable_appends before queued stable jobs
send translation_stable or translation_status for those stable jobs
send transcript_final
close WebSocket
```

Do not try to cancel the model actor thread already inside
`HYMTTranslator.translate`.
Running stable jobs are not retranslated; they publish once before
`transcript_final`. Already-running preview model calls may continue on the
model actor, and stable finish work waits for the actor before publishing.
Preview results after finish are ignored. ASR-only mode keeps the current
`session.finish()` behavior.

## Translator

`HYMTTranslator.translate(text, *, target_language, source_language="",
max_new_tokens=512) -> str` remains synchronous. Runtime calls it through the
single `TranslationModelActor` thread. Load the model once at startup and never
download weights from request handling.

Default runtime path:

- model: `tencent/HY-MT1.5-1.8B`;
- attention: `sdpa`;
- decode backend: `fixed_mask`;
- generation parameters unchanged from the accepted baseline.

Prompt, sampling, tokenizer/model, or decode-path changes require the
translation quality gate. Protocol/runtime-only changes do not.

## Validation

Protocol/runtime:

- fake-translator unit tests for preview priority, stable reliability under
  backlog, stable no-timeout behavior, preview debounce/cancel/stale-drop, and
  finish suppression;
- service-ordering unit tests for the invariant that an old preview is never
  queued after a newer source revision;
- subtitle document unit tests for window projection, SRT history, and
  translation visibility;
- WebSocket E2E for ASR-only parity and ASR+translation ordering.

Translation quality gate, only for model/prompt/generation/decode changes:

- no new quality errors versus the accepted baseline;
- target language, empty output, length outliers, repetition loops, and required
  structural markers;
- `must_preserve` items such as protocol labels, fixed UI strings, numbers,
  units, and subtitle cue ids;
- per-case reference similarity with regression failure on meaningful drops.

Private audio, transcripts, and generated outputs stay in `local_data/`,
`local_goldens/`, or `/tmp`.
