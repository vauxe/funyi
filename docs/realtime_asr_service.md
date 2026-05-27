# Realtime WebSocket ASR Service

Local single-user transcription service. It is not a public multi-user service.

This document owns the `/ws/asr` command and event contract. Streaming window
mechanics live in `@docs/streaming_runtime.md`, optimization flags live in
`@docs/performance_optimization.md`, and translation scheduling lives in
`@docs/realtime_translation_design.md`.

## Protocol

The service exposes:

- `GET /healthz` -> `{"status":"ok"}`
- `WS /ws/asr` for one active local realtime session.

The first WebSocket frame must be a JSON `start` command:

```json
{"type":"start","session_id":"local","sample_rate":16000,"audio_format":"pcm_s16le","language":"Chinese","context":""}
```

Accepted `start` fields are `type`, `session_id`, `sample_rate`,
`audio_format`, `language`, `context`, and `target_language`. Unknown fields are
rejected before `ready`. `sample_rate` defaults to `16000`; `audio_format`
defaults to `pcm_s16le`. The only accepted audio stream is mono little-endian
`pcm_s16le` at 16 kHz.

`language`, when non-empty, must be one of the supported Qwen3-ASR language
names and is normalized case-insensitively. Omit it, set it to `null`, or set it
to an empty string for auto language detection. Invalid languages are rejected
before `ready`.

If the service was started with `--translation-model`, `target_language`
selects the per-session translation target. Targets must be in the HY-MT
model-card language list:

```json
{"type":"start","session_id":"local","language":"Chinese","target_language":"English"}
```

Omit `target_language` to run transcription only. The service has no default
translation target, and empty `target_language` values are rejected in `start`.

After `ready`, send binary WebSocket frames containing 16 kHz mono
little-endian `pcm_s16le`. For low-latency captioning, send about 100 ms per
WebSocket audio frame. Frame size is transport cadence; the service accepts
each frame directly and ASR runs on the model streaming cadence.

Commands after `ready`:

```json
{"type":"flush"}
{"type":"finish"}
```

`flush` promotes the current ASR tail when possible and keeps the session open.
`finish` promotes the final tail, emits `transcript_final`, then closes the
WebSocket with close code `1000`.

`set_language` changes future transcription and translation settings:

```json
{"type":"set_language","language":"English","target_language":"Japanese"}
```

Omitted fields are unchanged. Null or empty `language` returns future ASR to
auto language detection; null or empty `target_language` disables future
translation. Non-empty targets require `--translation-model`. The server flushes
the current ASR tail before applying changed settings, and stable history is not
rewritten or retranslated.

## Server Events

This section is the single source of truth for `/ws/asr` server event payloads.
Other docs should link here instead of redefining these shapes.

All server events are JSON text frames with a `type` field. Clients should ignore
fields they do not need.

### `ready`

Sent once after the `start` command is accepted. Audio capture should begin only
after this event.

```json
{
  "type": "ready",
  "session_id": "local",
  "sample_rate": 16000,
  "audio_format": "pcm_s16le"
}
```

Optional `ready.timestamps`, present only when forced-aligner timestamps are
enabled:

```json
{
  "enabled": true,
  "model": "Qwen/Qwen3-ForcedAligner-0.6B",
  "source": "forced_aligner",
  "allowed_source_languages": ["Chinese", "English"],
  "stable": {
    "initial_status": "pending",
    "patch_event": "transcript_timing_update",
    "finish_timeout_ms": 30000,
    "pad_ms": 500
  }
}
```

Optional `ready.translation`, present only when the session has a
`target_language` in `start`:

```json
{
  "enabled": true,
  "target_language": "English",
  "model": "tencent/HY-MT1.5-1.8B",
  "stable": {
    "enabled": true,
    "reliable": true,
    "queue_size": null,
    "timeout_ms": null,
    "batch_size": 1
  },
  "preview": {
    "enabled": true,
    "debounce_ms": 700,
    "timeout_ms": 30000
  }
}
```

Enabling translation later with `set_language.target_language` does not resend
`ready`; later translation events carry their own `target_language`.

### `transcript_update`

Normal source-caption update. `stable_appends` is append-only history; `partial`
is the replaceable current tail, or `null`.

```json
{
  "type": "transcript_update",
  "revision": 7,
  "stable_base": 2,
  "stable_count": 3,
  "stable_appends": [
    {
      "id": "seg_000003",
      "index": 3,
      "start_ms": 1200,
      "end_ms": 2100,
      "text": "caption text",
      "language": "English"
    }
  ],
  "partial": {
    "start_ms": 2100,
    "end_ms": 2600,
    "text": "current",
    "language": "English"
  }
}
```

Each stable segment has `id`, `index`, `start_ms`, `end_ms`, `text`, and
`language`. In forced-aligner timestamp mode, new stable segments initially use
`start_ms: null`, `end_ms: null`, and `timing_status: "pending"`.
`stable_appends` are transcript history segments, not subtitle layout units.

Clients append `stable_appends`, replace `partial`, and require `stable_base` to
match local `stable_count`. Clients that need compact subtitles render the latest
stable/current text into a constrained view locally. If the base check fails,
reconnect and start a fresh session.

### `transcript_timing_update`

Forced-aligner timestamp patch for one existing stable segment.

```json
{
  "type": "transcript_timing_update",
  "source_segment_id": "seg_000003",
  "start_ms": 120,
  "end_ms": 860,
  "timing_status": "aligned"
}
```

`timing_status` is `aligned` or `failed`. Failed patches use `start_ms: null` and
`end_ms: null`. This event must not create, remove, reorder, or rewrite
transcript text. Clients that do not need timestamps can ignore it.

### `translation_preview`

Best-effort translation for the current `partial`. It is temporary and should be
displayed only when `source_revision` matches the current partial revision.

```json
{
  "type": "translation_preview",
  "source_revision": 13,
  "target_language": "English",
  "text": "..."
}
```

### `translation_stable`

Durable translation for one stable source segment.

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

### `translation_status`

Stable translation status for one source segment. Emitted when stable
translation fails.

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

`code` is `failed`.

### `transcript_final`

Final stable snapshot. The service emits this after final transcript,
timestamp, and stable-translation work that must complete before close.

```json
{
  "type": "transcript_final",
  "revision": 8,
  "stable_count": 3,
  "segments": []
}
```

`segments` uses the same stable-segment shape as `transcript_update.stable_appends`.
After `transcript_final`, the service closes the WebSocket with code `1000`.

### `error`

Fatal error:

```json
{"type":"error","error":"message","fatal":true}
```

Startup validation failures send `error` with `fatal=true` and close the
WebSocket, usually with code `1003`. A second concurrent session is rejected
with code `1013`. Internal session failures close with code `1011`.

Recoverable command error after `ready`:

```json
{"type":"error","error":"message"}
```

Recoverable command errors do not automatically close the session.

## Timestamp Mode

By default, stable segments use sample-clock `start_ms` / `end_ms` values.
Starting the service with `--timestamp-model <model>` enables forced-aligner
timestamps. In that mode, stable-segment public timing is one forced-aligned
`start_ms` / `end_ms` pair, filled asynchronously.

Forced-aligner timestamps use the ForcedAligner model-card language list, but
`language` remains an ASR prompt and accepts the full Qwen3-ASR language list.
Segments whose detected or configured source language is outside the
ForcedAligner list are still transcribed; their timestamp patches are marked
`timing_status="failed"`. `ready.timestamps.allowed_source_languages` exposes
the source-language list that can produce aligned timestamps.

New stable segments are emitted immediately in `transcript_update` with pending
timing. When alignment finishes, the service sends `transcript_timing_update`
for the same stable segment.

For `finish`, timestamp-enabled sessions wait up to the configured
`--timestamp-finish-timeout-ms` for queued stable-segment timing before
`transcript_final`. Segments that still cannot be aligned keep `start_ms=null`,
`end_ms=null`, and use `timing_status="failed"` in the final snapshot.

## Transcript State

```text
stable_segments[0:stable_count]   append-only stable prefix
partial                           replace-only current tail, or null
revision                          monotonic event version
```

If the base check fails, reconnect and start a fresh session. The protocol does
not expose an in-session snapshot command. Do not merge divergent transcript
states.

## Boundaries

- `realtime_server.py`: one connection, start validation, PCM decode,
  `asyncio.to_thread(...)`, JSON send.
- `RealtimeASRSession`: lossless PCM ingestion, ASR cadence, sample clock, text
  stabilization, `TranscriptStore` writes, final flush.
- `TranscriptStore`: in-memory append-only source transcript.
- `RealtimeTimestampRuntime`: optional stable-segment forced alignment and
  `transcript_timing_update` patches.
- `RealtimeTranslationRuntime`: optional source-event consumer that emits
  translation preview/stable/status events without rewriting source transcript
  history.

Model windowing, prompt rollback, carried text prefixes, and spec decode belong
to the streaming runtime design in `@docs/streaming_runtime.md`. The session
must not treat model-carried prefix text as user-visible stable history.

## Rules

- transport frames are not ASR chunks or transcript segments;
- every accepted PCM sample must eventually be fed to the streaming ASR state;
- clients should use replaceable `partial` text for the live subtitle line;
- one WebSocket session owns one continuous ASR stream;
- `flush` promotes the current tail but does not start a new model epoch;
- `set_language` promotes the current tail, then starts future ASR from the new
  language setting;
- long speech may stabilize repeated text after `live_stability_delay_ms`;
- ASCII word fragments stay partial;
- stable history is never rewritten;
- timestamp and translation events annotate existing source segments; they do
  not rewrite source transcript history;
- bounded-window and final-tail selection rules live in
  `@docs/streaming_runtime.md`.

## Defaults

Protocol-visible service defaults:

- one active WebSocket session;
- stable history delay: `live_stability_delay_ms=12000`; clients should render
  replaceable `partial` in a local compact view for low-latency subtitles;
- forced-aligner timestamps are off unless `--timestamp-model` is set;
- timestamp mode defaults: `--timestamp-pad-ms=500`,
  `--timestamp-finish-timeout-ms=30000`, `--timestamp-local-files-only`;
- translation is available only when the service starts with
  `--translation-model`.

The local service runtime profile is live20. Its model-window settings are in
`@docs/streaming_runtime.md`; optimization flags and rejected paths are in
`@docs/performance_optimization.md`.

## Validation

```bash
uv run python -m unittest discover tests
```

Use `tools/ws_e2e_leak_check.py` after starting `realtime_server.py` for service
smoke, CER, update-gap, memory, and shutdown checks.
