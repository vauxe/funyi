# Infinite Streaming ASR Design

## Scope

This document describes design goals and module contracts for `/ws/asr`
`aligned_windowed` realtime ASR. It does not track implementation progress.
Runtime status belongs in validation reports or PR notes.

The service is transcription-first. Translation is optional and consumes source
transcript events. Library/offline defaults stay upstream-compatible.

## Problem

Qwen3-ASR is a finite-audio transcription model. It does not expose native
infinite-audio decoder state or token timestamps. A realtime service therefore
has to turn an unbounded PCM stream into bounded model calls while keeping the
public transcript append-only.

The failure mode to avoid is a second, service-level ASR windowing path that
decodes recent audio independently from the model streaming state. Those windows
can omit, rewrite, or reorder text at boundaries. A forced aligner can locate
text that ASR already produced, but it cannot recover text that ASR skipped.

## Goals

- Support unbounded audio duration with bounded model-audio context and bounded
  timestamp audio retention.
- Use one ASR text source: `Qwen3ASRModel` streaming state.
- Require the forced aligner in the realtime service so stable source segments
  receive source-clock timestamps.
- Keep source transcript semantics explicit: stable text is append-only,
  partial text is replace-only, and finalization is terminal.
- Keep one public time axis: connection source sample index.
- Avoid post-generation text repair as the primary correctness mechanism.

## Non-Goals

- Do not implement a new acoustic model.
- Do not fork a parallel service-level ASR window decoder.
- Do not use forced alignment as proof that ASR text is correct.
- Do not deduplicate repeated text after publication as the primary repeat fix.
- Do not make ASR-only service modes behave like `aligned_windowed`.

## Architecture

```text
WebSocket PCM
  -> realtime_server.py
  -> AudioTimelineBuffer
  -> SpeechGate
  -> RealtimeConnectionSession
  -> Qwen3ASRModel streaming state
  -> TranscriptStore
  -> RealtimeTimestampRuntime
  -> WebSocket events
```

Module contracts:

- `realtime_server.py` validates one local WebSocket session, decodes PCM,
  starts ASR/timestamp/translation runtimes, and owns send backpressure.
- `AudioTimelineBuffer` stores source-clock PCM for pending timestamp jobs. It
  does not know about text or transcript publication.
- `SpeechGate` turns source-clock PCM into speech epochs. It must preserve
  source sample positions when silence is skipped.
- `RealtimeConnectionSession` owns speech epochs, model streaming ASR state,
  stable/partial transcript writes, and timestamp job hints.
- `Qwen3ASRModel` owns finite-window model inference, rolling audio trimming,
  bounded prompt prefix, and recognition frames. It does not own public
  transcript history.
- `TranscriptStore` owns append-only stable segments, replace-only partial text,
  timing patches, and final markers.
- `RealtimeTimestampRuntime` consumes stable-segment jobs, crops source audio,
  runs the forced aligner, patches segment timestamps, and trims audio that no
  future timestamp job can need.

## ASR Text Path

The ASR text path is model streaming, not an independent latest-window decode:

```text
speech audio
  -> model streaming state
  -> RecognitionFrame
  -> TailSelector
  -> TextStabilizer
  -> TranscriptStore.append_stable_segment(...)
```

The model may keep bounded audio and bounded text prefix internally to preserve
upstream streaming semantics. That model prefix is inference state, not public
stable transcript history. Public stability comes only from the session
stabilizer and `TranscriptStore`.

## Timestamp Path

Stable text and timestamps are separate contracts:

```text
stable segment + source sample hint
  -> StableTimingJob
  -> crop source audio with pad
  -> forced aligner
  -> transcript_timing_update
```

New stable segments may be emitted with `timing_status="pending"` and
`start_ms/end_ms=null`. The timestamp runtime patches each segment to
`aligned` or `failed`. `finish` waits for pending timestamp jobs up to the
configured timeout before emitting `transcript_final`.

The forced aligner output is relative to the crop. The runtime must add the
crop source start before converting to milliseconds.

## Retention

Model audio retention is owned by `ASRStreamingState` and
`RollingWindowTrimPolicy`.

Timestamp audio retention is owned by `RealtimeTimestampRuntime`:

```text
trim_before = min(completed_timestamp_floor, oldest_queued_job_start) - pad
```

After a timestamp job is processed, no future stable segment may start before
that segment's source end. The timestamp runtime can therefore advance the
completed floor and trim older source audio while keeping pad and queued jobs
intact.

The service must not rely on a full-session PCM buffer to be correct.

## WebSocket Semantics

`@docs/realtime_asr_service.md` owns the canonical protocol. This design must
not define conflicting payloads.

Required visible semantics:

- unsupported commit modes are rejected before audio ingest;
- `aligned_windowed` requires both ASR and forced aligner;
- `transcript_update.stable_appends` is append-only;
- `transcript_update.partial` is replace-only;
- `transcript_timing_update` may only patch an existing stable segment;
- `transcript_final` is terminal and must not be the only place stable text
  appears.

## Validation

Changes to the realtime ASR/timestamp path require:

- focused unit tests for transcript, timing, retention, and WebSocket event
  invariants;
- fast 60s WebSocket E2E before long runs;
- comparable 600s WebSocket E2E with punctuation-stripped CER, repetition
  checks, event-contract checks, and timestamp quality summary;
- no important metric regression under the same command shape.
