# Realtime WebSocket ASR Service

Local single-user transcription service. It is not a public multi-user service.

## Protocol

Start:

```json
{"type":"start","session_id":"local","language":"Chinese","context":""}
```

`language`, when present, must be one of the supported Qwen3-ASR language
names. Invalid languages are rejected before `ready`.

The service default keeps stable history conservative:
`live_stability_delay_ms=12000`. Use `partial` updates for low-latency live
subtitle display. Lower `--live-stability-delay-ms` only when the service can
tolerate more aggressive stable commits.

If the service was started with `--translation-model`, `target_language`
selects the per-session translation target. Targets must be in the HY-MT
model-card language list:

```json
{"type":"start","session_id":"local","language":"Chinese","target_language":"English"}
```

Omit `target_language` to run transcription only. The service has no default
translation target, and empty `target_language` values are rejected.

Then send mono little-endian `pcm_s16le` at 16 kHz. For low-latency captioning,
send about 100 ms per WebSocket audio frame. Frame size is transport cadence;
the service accepts each frame directly and ASR runs on the model streaming
cadence.

Commands: `flush`, `finish`, `set_language`.

`set_language` changes future transcription and translation settings:

```json
{"type":"set_language","language":"English","target_language":"Japanese"}
```

Omitted fields are unchanged. Null or empty `language` returns future ASR to
auto language detection; null or empty `target_language` disables future
translation. Non-empty targets require `--translation-model`. The server flushes
the current ASR tail before applying changed settings, and stable history is not
rewritten or retranslated.

Events:

- `ready`
- `transcript_update`
- `transcript_timing_update` when forced-aligner timestamps are enabled
- `transcript_final`
- `error`

`transcript_update` is the only normal caption update. The frontend appends
`stable_appends`, replaces `partial`, and requires `stable_base` to match local
`stable_count`.

Long stable text is split into subtitle-sized stable segments without using
punctuation as a boundary signal.

## Timestamp Mode

By default, stable segments use sample-clock `start_ms` / `end_ms` values.
Starting the service with `--timestamp-model <model>` enables forced-aligner
timestamps. In that mode, stable-segment public timing is one forced-aligned
`start_ms` / `end_ms` pair, filled asynchronously.

Forced-aligner timestamps use the ForcedAligner model-card language list. When
`language` is explicitly set in `start`, the service rejects values outside that
list before `ready`. Auto-language sessions may still transcribe ASR-supported
languages outside the ForcedAligner list, but their timestamp patches are marked
`timing_status="failed"`. `ready.timestamps.allowed_source_languages` exposes
the accepted ForcedAligner source-language list.

New stable segments are emitted immediately with pending timing:

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
      "timing_status": "pending",
      "text": "现在开始",
      "language": "Chinese"
    }
  ],
  "partial": null
}
```

When alignment finishes, the service patches the same stable segment:

```json
{
  "type": "transcript_timing_update",
  "source_segment_id": "seg_000003",
  "start_ms": 120,
  "end_ms": 860,
  "timing_status": "aligned"
}
```

`transcript_timing_update` only patches timing metadata for an existing stable
segment identified by `source_segment_id`. It must not create, remove, reorder,
or rewrite transcript text. Clients that do not need timestamps can ignore it.

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

If the base check fails, reconnect or request a fresh snapshot. Do not merge
divergent transcript states.

## Boundaries

- `realtime_server.py`: one connection, start validation, PCM decode,
  `asyncio.to_thread(...)`, JSON send.
- `RealtimeASRSession`: lossless PCM ingestion, ASR cadence, sample clock, text
  stabilization, `TranscriptStore` writes, final flush.
- `TranscriptStore`: in-memory append-only source transcript.
- `RealtimeTimestampRuntime`: optional stable-segment forced alignment and
  `transcript_timing_update` patches.

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
- bounded-window recognition frames may be tail-only after the stable cursor,
  and prompt-carried text must not be stabilized as new evidence;
- stable translation history must not drop middle source segments;
- changing `target_language` cancels pending preview and queued stable
  translation work; already emitted transcript history is not retranslated;
- unaligned finalization promotes a final tail update when it extends the last
  visible partial; otherwise it promotes the last visible partial instead of
  dropping user-visible tail text;
- translation/export belong above stable segments.

## Defaults

Service entrypoint defaults:

- live20: `chunk_size_sec=0.5`, `unfixed_chunk_num=4`,
  `max_window_sec=20`, `max_prefix_tokens=64`
- `spec_decode=True`
- `cuda_graph=True`, `cuda_graph_len_bucket=64`
- startup CUDA graph prewarm for live20; prewarm failure is a startup failure
- `flashinfer=True`
- `fused_rmsnorm=True`, `fused_linears=True`
- `w8a16=True` for qkv/gate_up
- `live_stability_delay_ms=12000`
- forced-aligner timestamps are disabled unless `--timestamp-model` is set
- timestamp mode defaults: `--timestamp-pad-ms=500`,
  `--timestamp-finish-timeout-ms=30000`,
  `--timestamp-local-files-only`

Disable flags only for debugging, fallback, or comparison.
When translation is enabled with `--translation-model`, a session may provide
`target_language` in the start command or later via `set_language`. Targets are
accepted only when they are in the HY-MT model-card language list.
Stable translation batching is opt-in with `--translation-stable-batch-size`.
The default is `1` so preview latency and existing single-item runtime behavior
stay unchanged; larger values only batch queued stable segments with the same
source language.

## Validation

```bash
uv run python -m unittest discover tests
```

Use `tools/ws_e2e_leak_check.py` after starting `realtime_server.py` for service
smoke, CER, update-gap, memory, and shutdown checks.
