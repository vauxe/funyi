# Realtime WebSocket ASR Service

This document describes the local single-user realtime ASR service. A local
client can connect on localhost. Do not design this as a public multi-user
service.

## Scope

The service does transcription only:

- accept one active WebSocket session;
- receive mono `pcm_s16le` audio at 16 kHz;
- emit live replacement captions and committed ASR segments;
- return final structured segments when the session finishes.

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
{"type":"partial","revision":1,"asr_epoch":1,"text":"...","start_ms":0,"end_ms":1000}
{"type":"committed","revision":2,"asr_epoch":1,"segment":{...}}
{"type":"final","revision":3,"segments":[...]}
{"type":"error","error":"..."}
```

`partial` is replace-style UI state. The frontend should replace the current
live caption with the latest partial text. `committed` is append-only history.

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
- live partials;
- committed VAD-ended speech segments;
- final flush.

`TranscriptStore` owns committed segments. It is intentionally in-memory for now.

## State Rules

The session is either idle or has one active speech segment.

An active speech segment keeps:

- `display_block_start_sample`
- `last_speech_end_sample`
- `text_anchor`
- ASR streaming state
- confirmed audio
- undecided endpoint-silence audio

Rules:

- binary transport chunks are not VAD frames, ASR chunks, or ASR segments;
- VAD endpoint, `flush`, and `finish` close the acoustic segment;
- display-duration rollover is disabled by default; if explicitly enabled, it
  commits only the current caption block and keeps ASR/VAD state active;
- undecided endpoint silence is not fed to ASR unless speech resumes;
- caption file export, if needed later, belongs in the frontend over committed
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

Disable flags only for debugging, environment fallback, or quality comparison.

## Validation

Use unit tests for state-machine and protocol behavior:

```bash
uv run python -m unittest tests.test_realtime_asr
```

Use `tools/ws_e2e_leak_check.py` for service-level smoke and long-running
resource checks after starting `realtime_server.py`.
