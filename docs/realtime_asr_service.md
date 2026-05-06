# Realtime WebSocket ASR Service

Local single-user transcription service. It is not a public multi-user service.

## Protocol

Start:

```json
{"type":"start","session_id":"local","language":"Chinese","context":""}
```

Then send mono little-endian `pcm_s16le` at 16 kHz.

Commands: `flush`, `finish`.

Events:

- `ready`
- `transcript_update`
- `transcript_final`
- `error`

`transcript_update` is the only normal caption update. The frontend appends
`stable_appends`, replaces `partial`, and requires `stable_base` to match local
`stable_count`.

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
- `RealtimeASRSession`: VAD, pre-roll, confirmed/undecided audio, ASR cadence,
  stable cursor, partial replacement, final flush.
- `TranscriptStore`: in-memory source transcript.

## Rules

- transport frames are not VAD frames, ASR chunks, or transcript segments;
- VAD decides acoustic activity, not text stability;
- long speech may stabilize repeated text after `live_stability_delay_ms`;
- ASCII word fragments stay partial;
- stable history is never rewritten;
- unaligned turn close clears partial and does not promote it;
- undecided endpoint silence is not fed to ASR unless speech resumes;
- translation/export belong above stable segments.

## Defaults

Service entrypoint defaults:

- live20: `chunk_size_sec=1.0`, `max_window_sec=20`, `max_prefix_tokens=64`
- `spec_decode=True`
- `cuda_graph=True`, `cuda_graph_len_bucket=64`
- `flashinfer=True`
- `fused_rmsnorm=True`, `fused_linears=True`
- `w8a16=True` for qkv/gate_up
- `live_stability_delay_ms=12000`

Disable flags only for debugging, fallback, or comparison.

## Validation

```bash
uv run python -m unittest tests.test_realtime_asr
```

Use `tools/ws_e2e_leak_check.py` after starting `realtime_server.py` for service
smoke, CER, update-gap, memory, and shutdown checks.
