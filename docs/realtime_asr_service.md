# Realtime WebSocket ASR Service

This document describes the local single-user realtime ASR service. A local
client can connect on localhost. Do not design this as a public multi-user
service.

## Scope

The service does transcription only:

- accept one active WebSocket session;
- receive mono `pcm_s16le` audio at 16 kHz;
- emit realtime transcript updates;
- return a final structured source transcript when the session finishes.

Keep translation, polishing, summarization, persistence, diarization, and
multi-session scheduling out of the ASR WebSocket path until they are needed by
the product.

## Protocol

Client messages:

```json
{"type":"start","session_id":"local","language":"Chinese","context":""}
```

Then send binary audio frames containing little-endian signed 16-bit PCM.

Supported JSON commands after `start`:

```json
{"type":"flush"}
{"type":"finish"}
```

Server events:

```json
{"type":"ready","session_id":"local","sample_rate":16000,"audio_format":"pcm_s16le"}
{"type":"transcript_update","revision":1,"stable_base":0,"stable_count":0,"stable_appends":[],"partial":{...}}
{"type":"transcript_final","revision":2,"stable_count":1,"segments":[...]}
{"type":"error","error":"..."}
```

`transcript_update` is the only normal caption update. The frontend appends
`stable_appends` to stable history and replaces the current `partial` snapshot.
`transcript_final` closes the source transcript.

## Transcript Model

The service owns one source transcript state:

```text
stable_segments[0:stable_count]   stable prefix, append-only, never changes
partial                          replace-only snapshot, or null
```

Wire updates use a compact delta:

```text
stable_appends      newly stable source segments
partial             current mutable text snapshot, or null
```

Frontend application rule:

```text
require local.revision < event.revision
require local.stable_count == event.stable_base
append event.stable_appends
replace current partial with event.partial
set local.stable_count = event.stable_count
set local.revision = event.revision
```

If the base check fails, reconnect or request a fresh snapshot. Do not merge
divergent transcript states.

## Runtime Boundaries

`realtime_server.py` owns only transport:

- accept/reject the single local connection;
- validate the initial `start` command;
- decode binary PCM frames;
- call `RealtimeASRSession` off the event loop with `asyncio.to_thread`;
- serialize returned events to JSON.

`RealtimeASRSession` owns realtime ASR state:

- VAD, pre-roll, short-pause handling;
- confirmed vs undecided audio;
- ASR cadence using the low-latency preset;
- stable cursor movement;
- partial replacement;
- final flush.

`TranscriptStore` owns the in-memory source transcript state.

## State Rules

The session is either idle or has one active speech segment.

An active speech segment keeps:

- `tail_start_sample`
- `last_speech_end_sample`
- `stable_text_anchor`
- `previous_tail_text`
- `previous_tail_end_sample`
- ASR streaming state
- confirmed audio
- undecided endpoint-silence audio

Rules:

- binary transport chunks are not VAD frames, ASR chunks, or transcript
  segments;
- VAD endpoint, `flush`, and `finish` close the acoustic turn and move safe text
  to the stable prefix;
- long continuous speech may advance the stable prefix after
  `live_stability_delay_ms`
  only when the latest ASR hypothesis repeats a prefix from the previous
  partial tail;
- ASCII word fragments are kept partial rather than stabilized mid-word;
- the newest ASR tail remains in `partial` until a later hypothesis or
  turn close makes it stable;
- if ASR contradicts text that has already become stable, the stable prefix is
  not rewritten;
- if a live ASR hypothesis no longer aligns with the stable prefix, the existing
  partial is preserved until alignment resumes or the turn closes;
- if turn close, `flush`, or `finish` still cannot align the final ASR text with
  the stable prefix, `partial` is cleared and not promoted to stable;
- VAD decides acoustic activity, not transcript finality;
- undecided endpoint silence is not fed to ASR unless speech resumes;
- caption file export, if needed later, belongs in the frontend over stable
  segments.

## Defaults

The service entrypoint defaults to the validated local low-latency profile:

- `chunk_size_sec=1.0`
- `max_window_sec=20.0`
- `spec_decode=True`
- `cuda_graph=True`
- `flashinfer=True`
- `fused_rmsnorm=True`
- `fused_linears=True`
- `w8a16=True`

The transcript projection uses `live_stability_delay_ms=12000` by default as
the long-speech escape hatch. This is not a blind display timer and not a model
LocalAgreement contract: it only moves text when a later ASR hypothesis repeats
a prefix from the previous partial tail. Setting the delay to `0` removes only
the wait; repeated-prefix evidence is still required. During VAD endpoint,
`flush`, or `finish`, the session runs ASR finish for the active turn and
stabilizes the remaining aligned text.

Disable flags only for debugging, environment fallback, or quality comparison.

## Validation

Use unit tests for state-machine and protocol behavior:

```bash
uv run python -m unittest tests.test_realtime_asr
```

Use `tools/ws_e2e_leak_check.py` for service-level smoke and long-running
resource checks after starting `realtime_server.py`.
