# Streaming Runtime

Use when changing `streaming_transcribe`, bounded live captions, model-window
state, or streaming quality gates. WebSocket and client event semantics live in
`@docs/realtime_asr_service.md`.

The target architecture for unbounded streaming over a finite ASR model is in
`@docs/infinite_streaming_asr_design.md`.

## Goal

Qwen3-ASR is a finite-audio transcription model. Realtime captions are built by
running that model repeatedly over bounded audio windows, then stabilizing the
mutable hypotheses outside the model.

The streaming runtime must make an infinite PCM stream look like a sequence of
finite ASR requests without letting mutable model output rewrite user-visible
history.

## Invariants

- ASR hypotheses are mutable; transcript history is append-only.
- Model input audio is bounded when `max_window_sec` is set.
- Prompted text is context, not new evidence.
- Only the stabilizer may decide that text is stable.
- Text stability does not by itself prove that old audio can be dropped.
- `TranscriptStore` is the source of user-visible history.
- Sample indices are the timing source; convert to milliseconds only at output
  boundaries.
- Without timestamps or an aligner, text-to-audio cut points are approximate and
  must be quality-gated with CER and WebSocket E2E.

## Two Cursors

The core streaming state has two independent cursors:

```text
audio_trim_cursor       oldest sample that must still be retained
transcript_commit_cursor oldest text position not yet committed to history
```

They move for different reasons:

- `transcript_commit_cursor` advances when the text stabilizer finds a stable
  prefix across repeated hypotheses.
- `audio_trim_cursor` advances only when a trim policy proves that earlier audio
  is no longer needed.

Do not infer one cursor from the other. A stable text prefix does not identify
the exact audio cut point unless the system has timestamps or an aligner. A
trimmed audio prefix does not mean its text is user-visible stable history.

## Current Split

The runtime keeps these responsibilities separate:

| Layer | Responsibility |
|---|---|
| `Qwen3ASRModel` | public offline facade and streaming wrapper over finite-window ASR calls |
| `ASRStreamingState` / `RollingWindowTrimPolicy` | PCM buffering, bounded audio windows, prompt prefix, rollback/spec-decode state |
| `RecognitionFrame` | explicit model/session data contract separating prompt text from generated evidence |
| `TailSelector` | select the replaceable transcript tail from one recognition frame and the transcript cursor |
| `TextStabilizer` | LocalAgreement-style stable prefix and replaceable partial text |
| `SpeechGate` | Silero speech-turn helper and source-clock speech epochs |
| `RealtimeConnectionSession` | speech epochs, model streaming ASR state, TranscriptStore writes, and timestamp job hints |
| `RealtimeTimestampRuntime` | source-audio buffer, forced-aligner jobs, and `transcript_timing_update` patches |
| `realtime_server.py` | WebSocket transport, one active local connection, JSON send policy |

`ASRStreamingState`, `RecognitionFrame`, `TailSelector`, `TextStabilizer`, and
`RollingWindowTrimPolicy` live in
`qwen3_asr_runtime/streaming.py`. A separate `WindowedStreamingRecognizer` is
only worth adding if stateful audio/prefill reuse or another caller needs that
boundary; otherwise it would mostly wrap the current model methods.

`model.py` should not own user-visible committed history. If it needs to carry
text that was dropped from the prompt or audio window, call it
`carried_text_prefix`, not `committed_text`.

The realtime service should not run a second service-level ASR windowing path.
It uses the model streaming state as the only ASR text source, writes stable
transcript text from that source, and sends forced-aligner jobs for timestamp
patches on the same source clock.

When a stable update contains a long run of text, `RealtimeConnectionSession`
appends one stable transcript segment. Compact subtitle length is a client
layout policy, not transcript history and not ASR chunks.

## Non-Streaming Model To Infinite Stream

Each decode step uses:

```text
speech-turn PCM
  -> advance audio_now
  -> fixed ASR cadence chunks
  -> audio window from audio_trim_cursor and retention policy
  -> finite ASR request with prompt context
  -> RecognitionFrame
  -> TailSelector selects uncommitted tail from explicit frame fields
  -> text stabilizer
  -> append stable text, replace partial text
  -> trim policy may advance audio_trim_cursor
```

The recognizer exposes recognition frames:

```python
RecognitionFrame(
    window_start_sample: int,
    audio_end_sample: int,
    full_text: str,
    language: str,
    decoded_text: str = "",
    generated_text: str = "",
)
```

`full_text` is kept for compatibility and diagnostics. It may include text
forced through the prompt or carried across prefix trimming. `decoded_text` is
the current model-visible text after any carried prefix has been removed; it may
still include mutable prompt-prefix text that has not been user-visible stable.
`generated_text` is the continuation after the current prompt prefix has been
stripped; it is a fallback tail candidate, not stable history by itself.

`TailSelector` selects a replaceable tail by contract:

- if `full_text` starts with the stable transcript prefix, use the exact suffix;
- if `decoded_text` starts with the stable transcript prefix, use the exact
  suffix;
- if the audio window overlaps the stable cursor and a stable transcript suffix
  overlaps the start of `decoded_text` after normalizing only whitespace and
  punctuation, remove that overlap; this handles prompt-prefix tail that became
  stable after the previous decode;
- if the candidate already matches the last visible partial, keep it as the
  current tail instead of removing a stable-suffix overlap;
- if `window_start_sample >= stable_end_sample`, use `decoded_text`, not
  `full_text`, because carried text is outside the current audio window while
  decoded prompt-prefix text may still be mutable uncommitted tail;
- otherwise the frame is unaligned: live updates may replace partial text.
  These ASR-only final-tail rules do not define the aligned realtime service
  final contract; `/ws/asr` final output is stable-history only.

Do not use fuzzy full-history overlap as a substitute for explicit frame fields.
The only overlap rule is stable-suffix to decoded-prefix after punctuation/space
normalization, and only when the audio window still overlaps the stable cursor.
At or after the cursor boundary, keep the decoded prefix because it may be a real
repeated phrase rather than prompt echo. After the rolling window advances, a
correct frame may be tail-only.

The stabilizer keeps only a previous tail and emits:

```python
StableTextUpdate(
    stable_text: str,
    partial_text: str | None,
    stable_end_sample: int | None,
)
```

Live stable text comes from a repeated common prefix between consecutive
hypotheses, trimmed to a safe text boundary. This library-level stabilizer does
not own the aligned realtime service final contract; see
`@docs/realtime_asr_service.md`.

For realtime service sessions, the WebSocket session owns one continuous source
sample clock and a shared `TranscriptStore`. Accepted PCM is appended to the
timestamp runtime's source-audio buffer; speech audio is also fed through the
model streaming state. Stable ASR segments are emitted with `timing_status:
"pending"` and patched by forced alignment through `transcript_timing_update`.
Explicit `flush`/`finish` drains the ASR tail without rewriting stable history.

The trim policy is explicit:

| Policy | Behavior | Correctness |
|---|---|---|
| `ManualFlushTrim` | trim only on explicit flush or finish | safest without timestamps; long speech may grow |
| `RollingWindowTrim` | keep `max_window_sec` plus overlap | practical low-latency mode; CER/E2E gated |
| `TimestampTrim` | trim to stable token or word `end_sample` | strict bounded mode, requires timestamp/aligner |

If the model returns only text, `RollingWindowTrim` is an approximation. It must
never be described as a proof that no uncommitted audio was dropped.

## Forced Aligner Timestamps

Qwen3-ForcedAligner returns item-level spans, not transcript-segment spans. The
forced-align text processor first splits transcript text into alignable units:
CJK text is mostly character-level, mixed Latin text is kept as words, and
space-delimited languages are word-level after punctuation cleanup. It then
inserts two `<timestamp>` markers per unit. The model predicts timestamp token
classes at those marker positions; the runtime multiplies the classes by
`model.config.timestamp_segment_time`, repairs non-monotonic values, and returns
`ForcedAlignResult.items` with relative `start_time` / `end_time` in seconds.

Realtime timestamp mode aggregates stable-segment timestamps from item spans:

```text
stable segment text + ASR sample hint
  -> crop the hinted source-audio window with pad
  -> align(crop, segment_text, language)
  -> patch segment.start_ms/end_ms through transcript_timing_update
```

Do not expose sample-clock estimates as final public segment timestamps. A
newly stable realtime segment may be emitted with `timing_status="pending"` and
`start_ms/end_ms=null`; `finish` waits for pending timestamp jobs and final
segments must be either `aligned` or `failed`. The aligner output is relative to
the crop, so every patch must add the source window start sample before
converting to milliseconds.

## Current Runtime

Library streaming defaults stay upstream-compatible:

- `chunk_size_sec=2.0`
- `max_window_sec=None`
- `spec_decode=False`
- optimized load flags off
- each step re-feeds accumulated audio

This default path is hash-regressed by
`local_goldens/streaming_regression.json`. Optimized and bounded-live paths are
CER-gated, not hash-gated.

Setting `max_window_sec` enables bounded live audio in the library streaming
wrapper:

- model audio context is capped;
- prompt text carries context across windows;
- old text carried by the model is not user-visible stable history;
- omitted `max_prefix_tokens` becomes `192`.

Current implementation: `ASRStreamingState.carried_text_prefix` is the model
continuity prefix carried across text-prefix trimming. `committed_text` remains
a backward-compatible alias and must not be confused with `TranscriptStore`
history.

## Live Modes

| Mode | Window | Role |
|---|---:|---|
| live20 | 20s | legacy low-latency library/session preset |
| live30 | 30s | generic bounded-live prefix-mode baseline |
| live45 | 45s | stricter quality mode |

Keep `abs-delta` drift separate from `worse-delta` quality regression.

## Low-Latency Library Preset

Load-time flags:

```python
Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    cuda_graph=True,
    flashinfer=True,
    fused_rmsnorm=True,
    fused_linears=True,
    quantized_linears=True,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
```

Streaming kwargs:

```python
Qwen3ASRModel.low_latency_preset_kwargs()
```

returns `chunk_size_sec=0.5`, `unfixed_chunk_num=4`,
`unfixed_token_num=5`, `max_window_sec=20.0`, `max_prefix_tokens=64`,
and `spec_decode=True`.

The local service prewarms the same live20/64-token CUDA graph shape used by
the low-latency model streaming preset; library and tool graph-bucket defaults
stay `1`. The realtime service requires forced-aligned timestamp patches for
stable segments, while clients can render replaceable `partial` text for
low-latency live subtitles. This is a service/session policy, not a library
streaming default.

## Spec Decode

`spec_decode=True` verifies rollback draft tokens with a prefill over
`prompt + rollback_draft`. Accepted draft tokens skip decode steps. Under bf16,
prefill-path KV can drift from decode-path KV, so gate with streaming CER.

Implementation invariant: prefix trimming may carry old text forward, but it
must preserve the rolled-back token suffix as `draft_ids`.

## Validation

Use `@docs/validation_and_regression.md`.

Required gates by change type:

- default library streaming: exact streaming regression;
- bounded-live or prompt/window/stabilizer changes: streaming CER gates;
- `RealtimeConnectionSession`, `RealtimeTimestampRuntime`, or WebSocket
  behavior: WebSocket E2E;
- user-visible caption behavior: client replay assertions for append-only
  stable history, replace-only partial text, and compact display layout.

Property tests should cover:

- stable text is append-only;
- partial text is replace-only;
- model rewrites do not mutate stable history;
- `audio_trim_cursor` and `transcript_commit_cursor` advance independently;
- tail-only bounded-window frames after the stable cursor are finalized;
- repeated phrases are not dropped by overlap handling;
- bounded windows do not duplicate or skip text across window rolls;
- ASR-only/library finalization promotes the remaining current tail, a longer
  final tail update, or the last visible partial.

## Do Not Reopen Without New Evidence

- `chunk_size_sec=3`, `unfixed_token_num=3`, or `max_new_tokens=64`
- live30 `max_prefix_tokens=64` outside the service profile
- graph buckets `32` or `256`
- fuzzy full-history overlap as a substitute for explicit window state
- raw low-energy RMS filtering as a substitute for speech-turn state
- persistent reserved StaticCache/graph across steps without allocator control

Open architecture work:

- replace full-window re-feed with stateful audio/prefill reuse;
- introduce timestamp or aligner support if strict audio-text trimming becomes
  required;
- introduce a separate `WindowedStreamingRecognizer` only if another caller or
  stateful audio/prefill reuse makes the current model wrapper too broad.
