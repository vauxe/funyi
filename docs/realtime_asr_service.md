# Realtime WebSocket ASR Service

Local single-user transcription service. It is not a public multi-user service.

This document owns the `/ws/asr` command and event contract. The infinite
streaming design goals live in `@docs/infinite_streaming_asr_design.md`;
library streaming mechanics live in `@docs/streaming_runtime.md`; optimization
flags live in `@docs/performance_optimization.md`; translation scheduling lives
in `@docs/realtime_translation_design.md`.

## Protocol

The service exposes:

- `GET /healthz` -> `{"status":"ok"}`
- `POST /api/transcriptions` for one-shot local file transcription.
- `POST /api/transcriptions/stream` for incremental local file transcription.
- `WS /ws/asr` for one active local realtime session.

Only one transcription session may be active at a time. A realtime WebSocket and
an offline file transcription share the same single-user mutex; concurrent
requests are rejected with `409` and `error.code: "busy"`.

## Offline File Transcription

`POST /api/transcriptions` accepts the media file as the raw request body and
returns a complete transcript snapshot. `POST /api/transcriptions/stream`
accepts the same request and returns newline-delimited JSON (`application/x-ndjson`)
so clients can render each completed source segment before the full document is
ready.
Both endpoints are intentionally not job APIs: the request owns the work and
completes when transcription is finished. Uploads are written to a temporary
file as they arrive; audio files that `libsndfile` can open are decoded through
project dependencies (`librosa` / `soundfile`), while audio or video containers
it cannot (mp4, mkv, mov, webm, m4a, ...) are transparently decoded to 16 kHz
mono via `ffmpeg`, which must be available on `PATH` for those formats. Either
way the audio track is transcribed in bounded chunks, with no fixed upload-size
or duration limit. The final JSON still contains the complete transcript, so
extremely long files remain bounded by local disk, runtime, and response size in
practice.

Query parameters are `language` (optional Qwen3-ASR language; empty means auto),
`context`, `targetLanguage` / `target_language` (optional HY-MT target, requiring
`--translation-model`), `timestamps` (default `true`), and `filename` (used only
to preserve a safe temporary suffix for media decoding).

Example:

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/transcriptions?language=Chinese&targetLanguage=English&filename=meeting.wav" \
  --data-binary @meeting.wav
```

Successful responses use `schemaVersion: 1`, `durationMs`, `language`, `text`,
and `segments[]` with `id`, `index`, `startMs`, `endMs`, `text`, `language`,
optional `timingStatus`, optional `translation`, and optional
`translationStatus` / `translationMessage`. Multi-cue translations may also
include `translationUnits[]`; single-cue translations stay on the segment.
`timingStatus` is `estimated` when cue timing is derived from ASR text and chunk
duration, and `aligned` after item-level forced alignment. Error responses use
`{"error":{"code":"...","message":"..."}}`.

The streaming endpoint emits realtime-compatible events: `transcript_update`
for each source segment, optional `translation_stable` when `targetLanguage` is
set, and a terminal `transcript_final` event. Source `transcript_update` events
are emitted after ASR/timestamp work and do not wait for HY-MT translation.
Translation events are a service-layer side track and include `source_revision`.
In `transcript_final`, the top-level `segments` field keeps the realtime
snake_case segment shape, while `document.segments` keeps the snapshot
camelCase shape returned by `/api/transcriptions` (`translation_status` maps to
`translationStatus`, and `translation_message` maps to `translationMessage`).
The side-track backlog is bounded; once it is full, later ASR chunks wait for
translation to drain instead of accumulating unbounded work. File-stream stable
translation is bounded by `--translation-stable-timeout-ms`; timeout emits
`translation_status` and the final document still completes. The final event
includes `revision`, `final_revision`, `stable_count`, and a
`document` field containing the same snapshot shape returned by
`/api/transcriptions`. Upload and validation errors still use normal HTTP error
responses before streaming starts; decode or model errors after streaming starts
are emitted as `type: "error"` events.

The first WebSocket frame must be a JSON `start` command:

```json
{"type":"start","session_id":"local","sample_rate":16000,"audio_format":"pcm_s16le","language":"Chinese","context":"","realtime_commit_mode":"aligned_windowed"}
```

Accepted `start` fields are `type`, `session_id`, `sample_rate`,
`audio_format`, `language`, `context`, `target_language`, and
`realtime_commit_mode`. Unknown fields are rejected before `ready`.
`sample_rate` defaults to `16000`; `audio_format` defaults to `pcm_s16le`.
The only accepted audio stream is mono little-endian `pcm_s16le` at 16 kHz.
`realtime_commit_mode` defaults to `aligned_windowed`; this is currently the
only supported realtime service commit mode.

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
WebSocket audio frame. Frame size is transport cadence; the ASR session decodes
on the model streaming cadence, and the timestamp runtime aligns stable
segments asynchronously. A single binary frame larger than the service limit
(about 500 s of audio) is rejected with a recoverable `error` and the session
stays open.

Commands after `ready`:

```json
{"type":"flush"}
{"type":"finish"}
```

`flush` drains the current ASR tail when possible and keeps the session open.
`finish` drains ASR, waits for pending timestamp work until the configured
timeout, emits `transcript_final`, then closes the WebSocket with close code
`1000`. `partial` is not durable; only segments already sent through
`stable_appends` are guaranteed durable.

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
  "audio_format": "pcm_s16le",
  "streaming": {
    "mode": "aligned_windowed",
    "requires": ["asr", "forced_aligner"],
    "stable": {
      "source": "asr_streaming_text_and_forced_aligner",
      "patch_event": "transcript_timing_update",
      "live_stability_delay_ms": 12000
    }
  }
}
```

`ready.timestamps` is present in the realtime service because forced alignment
is part of the service path. Stable transcript appends are initially timestamp
pending and are patched by `transcript_timing_update`:

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
  "model": "tencent/Hy-MT2-1.8B",
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
      "start_ms": null,
      "end_ms": null,
      "text": "caption text",
      "language": "English",
      "timing_status": "pending"
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
`language`. In realtime timestamp mode, new stable segments may include
`timing_status: "pending"` with `start_ms` and `end_ms` set to `null`; the
forced aligner later patches that segment to `aligned` or `failed`.
`stable_appends` are transcript history segments, not subtitle layout units.
For offline file streams, those transcript history segments are already
subtitle-shaped public `TranscriptSegment` values so the final document can be
exported directly as subtitles.

Clients append `stable_appends`, replace `partial`, and require `stable_base` to
match local `stable_count`. Clients that need compact subtitles render the latest
stable/current text into a constrained view locally. If the base check fails,
reconnect and start a fresh session.

### `transcript_timing_update`

Realtime timestamp patch for an existing stable source segment. It is emitted
after the forced aligner maps stable ASR text onto the source audio clock, and
may arrive after the `transcript_update` that created the segment.

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
transcript text.

### `transcript_status`

Recoverable or fatal source-transcript status. This event is reserved for
visible service/session failures that are not ordinary timestamp patches. It
does not append stable text.

```json
{
  "type": "transcript_status",
  "status": "source_model_error",
  "message": "ASR backend failed; the session cannot continue.",
  "fatal": true
}
```

Status values are service-owned. Clients may log or surface these statuses, but
should not rewrite transcript history in response to them.

### `translation_preview`

Best-effort translation for the current live translation unit. It is temporary
and should be displayed only when `source_revision` matches the current partial
revision. When previous `stable_appends` are still part of the open translation
unit, the source text translated by the backend is `pending stable source text +
partial.text`; clients that display source and preview translation together
should render the same composed source text.

```json
{
  "type": "translation_preview",
  "source_revision": 13,
  "target_language": "English",
  "text": "..."
}
```

### `translation_stable`

Durable translation for a stable source translation unit. The unit may cover
one or more adjacent stable source segments while a replaceable `partial` tail
is still active. `source_segment_id` / `source_segment_index` remain the
anchor segment for clients that do not consume coverage lists; when present,
`source_segment_ids` / `source_segment_indices` describe every covered source
segment.

```json
{
  "type": "translation_stable",
  "source_revision": 12,
  "source_segment_id": "seg_000002",
  "source_segment_index": 2,
  "source_segment_ids": ["seg_000001", "seg_000002"],
  "source_segment_indices": [1, 2],
  "target_language": "English",
  "text": "..."
}
```

Client replay rules:

- Prefer `source_segment_ids` / `source_segment_indices` when present. They
  describe one translation unit's source coverage; they do not fold the covered
  source cues into one source line.
- When both coverage lists are present, they are emitted in the same stable
  history order, have the same length, and each id/index pair refers to the
  same source segment. Clients should resolve by id first and use the paired
  index only as fallback if the id cannot be found.
- `source_segment_id` / `source_segment_index` are the anchor for older clients
  and for fallback lookup. They do not describe the full covered source text
  when coverage lists are present.
- Source history, copy, and subtitle export should keep the source segment list as
  emitted. Coverage lists describe which source cues a translation belongs to;
  they do not change the number of source subtitle cues.
- Covered source segments are adjacent in stable history. A client that renders a
  grouped translation overlay can use the first covered segment start and the
  last covered segment end when both are available.

### `translation_status`

Stable translation status for one stable source translation unit. Emitted when
stable translation fails. It uses the same anchor and optional coverage fields
as `translation_stable`.

```json
{
  "type": "translation_status",
  "scope": "stable",
  "code": "failed",
  "source_revision": 12,
  "source_segment_id": "seg_000002",
  "source_segment_index": 2,
  "source_segment_ids": ["seg_000001", "seg_000002"],
  "source_segment_indices": [1, 2],
  "target_language": "English",
  "message": "translation failed"
}
```

`code` is `failed` or `timeout`. `timeout` is emitted when file-stream stable
translation exceeds `--translation-stable-timeout-ms`.

### `transcript_final`

Terminal stable-history marker. The service emits this after final transcript
and stable-translation work that must complete before close. In aligned
realtime mode, stable text is replayed from prior `stable_appends`; final text
must not appear only in this event.

```json
{
  "type": "transcript_final",
  "revision": 8,
  "final_revision": 8,
  "stable_count": 1,
  "segments": [
    {"id": "seg_000001", "index": 1, "start_ms": 0, "end_ms": 1200, "text": "你好", "language": "Chinese"}
  ],
  "document": {
    "schemaVersion": 1,
    "durationMs": 1200,
    "language": "Chinese",
    "text": "你好",
    "segments": [
      {"id": "seg_000001", "index": 1, "startMs": 0, "endMs": 1200, "text": "你好", "language": "Chinese"}
    ]
  }
}
```

`/api/transcriptions/stream` includes top-level `segments`, using the same
stable-segment shape as `transcript_update.stable_appends`, and `document`, using
the same snapshot shape as `/api/transcriptions`. Unbounded realtime WebSocket
sessions may omit `segments` and do not include `document`; clients keep the
stable history they already replayed from `transcript_update`.

After a WebSocket `transcript_final`, the service closes the WebSocket with code
`1000`. The HTTP file-stream response ends after its terminal event.

### `error`

Fatal error:

```json
{"type":"error","error":{"code":"internal_error","message":"Offline transcription failed."},"fatal":true}
```

Startup validation failures send `error` with `fatal=true` and close the
WebSocket, usually with code `1003`. A second concurrent session is rejected
with code `1013`. Service misconfiguration and internal session failures close
with code `1011`. File-stream errors use the structured `error.code` /
`error.message` shape; WebSocket errors may use a string `error` message for
legacy realtime command failures.

Recoverable command error after `ready`:

```json
{"type":"error","error":"message"}
```

Recoverable command errors do not automatically close the session.

## Aligned Realtime Mode

Realtime ASR requires `--timestamp-model <model>`. Stable-segment public timing
is one forced-aligned `start_ms` / `end_ms` pair; sample-clock estimates are not
published as stable segment timestamps.

Realtime commits use the ForcedAligner model-card language list. Because the
service publishes stable text only with forced-aligner timestamp patches,
`language` must be one of `ready.timestamps.allowed_source_languages`; unsupported
source languages are rejected before audio ingest.

New stable segments are emitted in `transcript_update` after ASR text stability.
They start with pending timestamps, then `transcript_timing_update` patches them
after forced alignment succeeds or fails.

For `finish`, the session waits up to the configured
`--timestamp-finish-timeout-ms` for pending forced-aligner work before
`transcript_final`. `finish` is terminal even when a timestamp patch fails; the
failed segment remains visible with `timing_status: "failed"`.

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
  timestamp runtime integration, and JSON send.
- `RealtimeConnectionSession`: speech epochs, model streaming ASR state,
  stable/partial transcript updates, and timestamp job hints.
- `RealtimeTimestampRuntime`: source-audio buffer, forced-aligner jobs, and
  timestamp patches for existing stable segments.
- `TranscriptStore`: in-memory append-only source transcript.
- `RealtimeTranslationRuntime`: optional source-event consumer that emits
  translation preview/stable/status events without rewriting source transcript
  history.

Model windowing, prompt rollback, carried text prefixes, and spec decode belong
to the streaming runtime design in `@docs/streaming_runtime.md`. The session
must not treat model-carried prefix text as user-visible stable history.

## Rules

- transport frames are not ASR chunks or transcript segments;
- every accepted PCM sample advances the source timeline and is available to
  timestamp alignment; speech-gated audio also enters ASR streaming state;
- clients should use replaceable `partial` text for the live subtitle line;
- one WebSocket session owns one continuous transcript history and source clock;
- `flush` drains the current ASR tail but does not end the WebSocket session;
- `set_language` first flushes the current tail, then starts future ASR from
  the new language setting;
- live stable text requires repeated ASR evidence; final timestamps require
  forced-aligner patches;
- ASCII word fragments stay partial;
- stable history is never rewritten;
- timestamp and translation events annotate existing source segments; they do
  not rewrite source transcript history;
- model-window and final-tail selection rules live in
  `@docs/streaming_runtime.md`.

## Defaults

Protocol-visible service defaults:

- one active WebSocket session;
- stable history requires ASR text stability, and final timestamp quality
  requires forced-aligner patches; clients
  should render replaceable `partial` in a local compact view for low-latency
  subtitles;
- forced-aligner model is required with `--timestamp-model`;
- timestamp defaults: `--timestamp-finish-timeout-ms=30000`,
  `--timestamp-local-files-only`;
- translation is available only when the service starts with
  `--translation-model`.
- direct `realtime_server.py` starts with translation disabled unless
  `--translation-model` is passed; `scripts/start_backend.sh` passes the default
  HY-MT2 model unless `FUNYI_TRANSLATION_MODEL=` disables it.
- startup prewarms enabled model paths before the HTTP/WebSocket interface is
  created: ASR cuda graph, translation target buckets, and forced-aligner
  timestamps. Prewarm failure fails startup instead of exposing a cold or
  partially initialized service.

The local service runtime profile uses the live20 model streaming preset
(`chunk_size_sec=0.5`, `max_window_sec=20`, `max_prefix_tokens=64`,
`spec_decode=True`) plus required forced-aligner timestamp patches.
Optimization flags and rejected paths are in
`@docs/performance_optimization.md`.

For frontend/audio debugging, start the backend with `--log-level debug`. Debug
logs include throttled PCM duration/RMS/peak summaries, ASR frame text, and key
outgoing transcript summaries. They are local runtime logs and can include
recognized transcript text; do not paste private transcript logs into public
issues.

Add `--save-debug-audio` when you need the backend-received audio written to
WAV for inspection. Files are saved under
`local_data/realtime_debug_audio/` by default, or under `--debug-audio-dir` when
set. The saved WAV is 16 kHz mono PCM after WebSocket decoding, before ASR or
forced alignment.

## Validation

```bash
uv run --group test pytest
```

Use `tools/ws_e2e_leak_check.py` after starting `realtime_server.py` for service
smoke, CER, update-gap, memory, and shutdown checks.
