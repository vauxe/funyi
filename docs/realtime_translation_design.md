# Realtime Transcript And Translation Design

Status: source transcript event model implemented; translation runtime not
implemented yet.

This document defines the target event model for realtime captions and
translation. The root problem is not HY-MT integration. The root problem is text
state: streaming ASR produces changing hypotheses, while the user interface,
translation, and export need different stability guarantees.

The target model is:

```text
single transcript + monotonic stable cursor
```

Translation is added only after this transcript model is correct. `Qwen3ASRModel`
and the offline ASR runtime remain transcription-only.

## Core Model

The service owns one source transcript state:

```python
@dataclass
class TranscriptState:
    id: str
    revision: int
    stable_count: int
    stable_segments: list[StableSegment]
    partial: PartialSegment | None
```

Conceptually, `stable_count` splits the rendered transcript into two regions:

```text
stable_segments[0:stable_count]   stable prefix, append-only, never changes
partial                          replace-only snapshot, or null
```

This is the central invariant:

```text
revision increases monotonically
stable_count increases monotonically
stable prefix never changes
partial may be replaced on every update
```

There is no separate lifecycle for provisional segments. The newest text is not
finalized segment-by-segment. It is simply the replaceable `partial` until the
stable cursor moves past it.

## Why This Model

Realtime ASR text has two fundamental regions:

1. Text old enough and stable enough to treat as fact.
2. Recent text that is still likely to be rewritten by the model.

Anything more complex is an implementation detail. A `created / updated /
finalized` event model exposes too much process to the frontend. A legacy
event stream that alternates `partial` and `committed` events exposes too little
stability information.

The cursor model is smaller and more correct:

- stable history is append-only;
- recent uncertainty is replace-only;
- frontend logic is deterministic;
- translation has clear stable and preview boundaries;
- long continuous speech does not wait for VAD silence;
- ASR rewrites are absorbed by `partial`.

## Component Boundaries

| Component | Owns | Must Not Own |
|---|---|---|
| `RealtimeASRSession` | audio clock, VAD, ASR streaming state, transcript state | HY-MT |
| `TranscriptStore` | source transcript state and stable segment ids | target translations |
| `TranslationRuntime` | preview debounce, stable translation queue, stale-drop, timeouts | audio/VAD/ASR decode |
| `HYMTTranslator` | tokenizer/model/prompt/generation | WebSocket state |
| `realtime_server.py` | WebSocket transport, event composition, send serialization | model internals |

`TranscriptState` can initially live in `TranscriptStore` and
`realtime_session.py`. Extract a separate state-machine class only when the code
becomes harder to read.

## Segment Shape

Stable segments need durable ids because stable translation is keyed by segment.
Partial entries are snapshots and must not be persisted by id.

```python
@dataclass
class StableSegment:
    id: str
    index: int
    start_ms: int
    end_ms: int
    text: str
    language: str

@dataclass
class PartialSegment:
    start_ms: int
    end_ms: int
    text: str
    language: str
```

Rules:

- Stable ids are durable and append-only.
- Partial entries are replaced as a whole.
- Empty text must not create stable or partial entries.
- Segment times are monotonic.
- Internal timing should remain sample-based and convert to milliseconds only at
  event boundaries.
- Current ASR output has no word timestamps, so do not claim word-accurate
  segment timing.

## Protocol

### Ready

ASR-only:

```json
{
  "type": "ready",
  "session_id": "local",
  "sample_rate": 16000,
  "audio_format": "pcm_s16le"
}
```

With translation:

```json
{
  "type": "ready",
  "session_id": "local",
  "sample_rate": 16000,
  "audio_format": "pcm_s16le",
  "translation": {
    "enabled": true,
    "target_language": "English",
    "model": "tencent/HY-MT1.5-1.8B",
    "preview": true,
    "stable": true
  }
}
```

When translation is disabled, omit `translation`.

### Transcript Update

`transcript_update` is the only normal source-caption update event.

Wire format is a compact delta over the single transcript state:

```json
{
  "type": "transcript_update",
  "revision": 42,
  "stable_base": 7,
  "stable_count": 9,
  "stable_appends": [
    {
      "id": "seg_000008",
      "index": 8,
      "start_ms": 42000,
      "end_ms": 48600,
      "text": "we should first fix the transcript state model",
      "language": "English"
    },
    {
      "id": "seg_000009",
      "index": 9,
      "start_ms": 48600,
      "end_ms": 54800,
      "text": "then add translation as a subscriber",
      "language": "English"
    }
  ],
  "partial": {
    "start_ms": 54800,
    "end_ms": 58300,
    "text": "the newest words may still change",
    "language": "English"
  }
}
```

Frontend application rule:

```text
require local.revision < event.revision
require local.stable_count == event.stable_base
append event.stable_appends to stable history
set stable_count = event.stable_count
replace partial with event.partial
set revision = event.revision
```

If the base check fails, reconnect or request a fresh snapshot. Do not attempt
to merge divergent transcript states.

Rules:

- `stable_base` is the stable count before applying `stable_appends`.
- `stable_count == stable_base + len(stable_appends)`.
- `stable_appends` may be empty.
- `partial` is always the current replacement snapshot, or `null`.
- `partial: null` clears the live caption area.
- Stable segments are never resent as updates except in a final/full snapshot.

### Transcript Final

`transcript_final` closes the source transcript:

```json
{
  "type": "transcript_final",
  "revision": 50,
  "stable_count": 12,
  "segments": [...]
}
```

Rules:

- Before `transcript_final`, the session moves all remaining safe source text to
  stable segments.
- The final snapshot contains source transcript only.
- Translation output is not embedded in the source transcript snapshot.

## Frontend State

The frontend stores:

```text
stable_segments[]     append-only
partial               replace-only, or null
revision
stable_count
```

Rendering is simple:

```text
render stable_segments as history
render partial as current replaceable tail
```

The frontend does not need to understand VAD, ASR chunks, partial hypotheses, or
segment lifecycle events.

## Moving The Stable Cursor

The stable cursor should move conservatively. It is not driven only by VAD.

Inputs:

- latest ASR hypothesis text;
- stable prefix anchor inside the ASR hypothesis;
- previous partial tail;
- sample clock;
- VAD start/end hints;
- `flush` and `finish` commands.

Algorithm:

```text
on ASR hypothesis:
  tail_text = tail_after_stable_anchor(stable_anchor, full_asr_hypothesis)
  partial = tail_text

  if live_stability_delay_ms has elapsed:
      stable_prefix = common prefix(previous_tail_text, tail_text)
      trim stable_prefix back from any ASCII word fragment
      if stable_prefix is non-empty:
          move stable_prefix into stable_appends
          use previous_tail_end_sample as the stable segment end
          partial = tail_text minus stable_prefix

  remember tail_text and current ASR end sample for the next hypothesis

on VAD endpoint / flush / finish:
  run ASR finish for the active speech turn
  if final ASR text aligns with the stable prefix anchor:
      move remaining aligned tail_text into stable_appends
  clear partial

if a live hypothesis no longer aligns with the stable prefix anchor:
  do not mutate stable history
  do not clear the existing partial

if turn close still does not align:
  do not mutate stable history
  clear partial
```

Current source-transcript defaults:

| Setting | Value |
|---|---:|
| ASR cadence | 1s |
| live stability delay | 12s |
| stability evidence | prefix repeats in one later hypothesis |
| ASCII word handling | do not stabilize a partial word |
| live stable timestamp | previous hypothesis end sample |
| turn-close stable timestamp | VAD/flush/finish close sample |

Do not depend on punctuation. Realtime ASR may add punctuation late or rewrite
it. Without word timestamps or token confidence, the simplest correct policy is
to leave the newest hypothesis tail in `partial` and only stabilize text after
it is repeated by a later hypothesis or by turn close.
Setting `live_stability_delay_ms` to `0` removes only the wait; it does not make
the latest one-off hypothesis stable.

## VAD Role

VAD is still needed, but it is not the transcript stability policy.

VAD owns:

- speech start;
- pre-roll recovery;
- silence filtering;
- turn-end hint;
- reset points for `flush` and `finish`.

VAD must not own:

- stable cursor movement during long speech;
- subtitle segment duration;
- translation scheduling;
- final transcript semantics.

In short:

```text
VAD decides acoustic activity.
TranscriptState decides text stability.
```

## Scenario Behavior

### Short Speech With Pause

```text
updates while speaking:
  stable_appends = []
  partial = current text

VAD endpoint:
  stable_appends = whole utterance
  partial = null
```

The user sees live captions immediately. The stable segment appears when the
turn ends.

### Long Continuous Speech

```text
t=0-12s:
  stable_appends = []
  partial = recent hypothesis

t=12s + next ASR hypothesis:
  stable_appends = prefix repeated by previous and latest hypotheses
  partial = recent mutable tail

t=next threshold:
  stable_appends = next stable prefix
  partial = next mutable tail
```

The transcript advances without waiting for silence, while the newest text stays
replaceable.

### Recent ASR Rewrite

```text
old partial = "we should add translation first"
new partial = "we should fix the transcript first"
```

The frontend replaces `partial`. No segment update event is needed.

### Rewrite Older Than Stable Prefix

If the ASR later contradicts already stable text, the realtime service ignores
that rewrite for displayed history.

This is intentional. Correcting old stable history would require full-document
diffing, translation invalidation, and complex frontend merge rules. Reduce the
risk by moving the stable cursor conservatively, not by rewriting the stable
prefix.

### No Punctuation

Do not wait for punctuation. The long-speech threshold and repeated-prefix rule
advance stable history for ordinary unpunctuated speech. If the model keeps
rewriting with no common prefix, keep `partial` replaceable rather than
inventing false stability.

### Flush

`flush` acts like a forced turn boundary:

```text
finish current ASR turn
move remaining aligned tail_text to stable_appends
clear partial
reset VAD/turn state
```

### Finish

`finish` must be bounded:

```text
stop preview translation scheduling
drop queued preview work
session.finish()
  -> emit final transcript_update events if needed
  -> emit transcript_final
translate only finish-created stable_appends with timeout
send transcript_final
close
```

Do not wait for old translation backlog before closing.

## Translation

Translation subscribes to transcript events. It never mutates source transcript
state.

```text
stable_appends     -> stable translation, append-only
partial            -> preview translation, latest-only
```

### Translation Preview

`translation_preview` is replace-style and follows the latest `partial`:

```json
{
  "type": "translation_preview",
  "source_revision": 42,
  "source_stable_count": 9,
  "target_language": "Chinese",
  "text": "最新的词可能还会变化",
  "start_ms": 54800,
  "end_ms": 58300,
  "latency_ms": 486
}
```

Rules:

- Translate the current `partial`, not the full transcript.
- Use latest-only scheduling.
- Drop results whose `source_revision` is older than the current transcript
  revision.
- Preview failures are silent.
- `partial: null` clears the target preview locally. The service does not need
  to emit an empty preview.

### Stable Translation

`translation_stable` is keyed by stable source segment id:

```json
{
  "type": "translation_stable",
  "source_revision": 42,
  "segment_id": "seg_000009",
  "target_language": "Chinese",
  "text": "然后把翻译作为订阅者接入。",
  "start_ms": 48600,
  "end_ms": 54800,
  "latency_ms": 538
}
```

Rules:

- Emit for stable segments only.
- The source segment must never change after this event.
- Use a small FIFO queue.
- Timeout and queue overflow are explicit status events.

### Translation Status

```json
{
  "type": "translation_status",
  "source_revision": 42,
  "segment_id": "seg_000009",
  "status": "timeout",
  "error": "translation timed out after 5.0s"
}
```

Allowed statuses:

- `timeout`
- `failed`
- `skipped_backlog`
- `stale_revision`

Do not expose stack traces or model internals.

## Translation Scheduler

HY-MT is not treated as a streaming decoder. Every request is bounded
text-to-text generation.

Preview defaults:

| Setting | Value |
|---|---:|
| debounce | 800-1200ms |
| minimum text growth | 8-12 CJK chars or 20-30 Latin chars |
| max_new_tokens | 96-128 |
| queue policy | latest-only |
| stale result policy | drop |

Stable translation defaults:

| Setting | Value |
|---|---:|
| queue size | 4 |
| max_new_tokens | 256 |
| timeout | 5s |
| queue full policy | `translation_status: skipped_backlog` |

Stable translation has priority over preview work that has not started yet. Do
not try to preempt a HY-MT generation already running. Keep preview generation
bounded so it cannot block stable translations for long.

Run at most one HY-MT generation at a time in v1. ASR and HY-MT can still
contend at CUDA level, so WebSocket E2E must measure source update latency with
translation enabled.

## Translator

`HYMTTranslator` should be synchronous and small:

```python
class HYMTTranslator:
    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int = 256,
    ) -> str: ...
```

Model loading:

- load once at service startup when translation is enabled;
- default `local_files_only=True`;
- default `dtype=torch.bfloat16`;
- default `device_map="cuda:0"`;
- never download weights from request handling.

Generation:

```python
do_sample=False
repetition_penalty=1.05
```

Source language comes from:

1. stable source segment language;
2. start command `language`;
3. CJK-character heuristic;
4. empty string.

Prompt:

- If source or target language is Chinese, use HY-MT's Chinese prompt:

  ```text
  将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：

  {source_text}
  ```

- Otherwise use HY-MT's English prompt:

  ```text
  Translate the following segment into {target_language}, without additional explanation.

  {source_text}
  ```

Cleanup:

- strip leading/trailing whitespace;
- reject exact prompt echo;
- do not normalize punctuation beyond trimming.

## WebSocket Sending

Background translation needs its own output path because the receive loop can be
blocked on `websocket.receive()`:

```text
ASR/session task
  -> event_queue.put(transcript event)

translation worker
  -> event_queue.put(translation event)

sender task
  -> event_queue.get()
  -> send_json under send_lock
```

All WebSocket writes must use the same `asyncio.Lock`. Do not let two tasks call
`websocket.send_text` concurrently.

## Implementation Status

The source transcript event model is implemented. HY-MT translation is still not
implemented; do not add it until the transcript protocol remains stable under
service-level E2E tests.

Implemented source layer:

- `TranscriptStore` owns `revision`, `stable_count`, append-only stable
  segments, replace-only `partial`, `transcript_update`, and
  `transcript_final`.
- `RealtimeASRSession` emits only source transcript events. It does not emit
  legacy event types named `partial`, `committed`, or `final`.
- Long continuous speech advances the stable cursor from the prefix repeated by
  the previous and latest hypotheses; the latest tail remains partial.
- VAD endpoint, `flush`, and `finish` run ASR finish and stabilize the remaining
  aligned turn text.
- Unit tests cover cursor deltas, partial replacement, long speech, prefix
  rewrite, flush, finish, no-punctuation text, and protocol invariants.

Remaining work:

1. Add translation runtime.

   Subscribe to `transcript_update`. Implement latest-only preview translation
   and bounded stable translation.

2. Add HY-MT startup flags and ready metadata.

   Load once at startup from local cache and fail fast if unavailable.

3. Add fake-translator unit tests.

   Cover stale preview drop, stable translation queue, timeout, queue overflow,
   finish backlog policy, and WebSocket send serialization.

4. Add WebSocket E2E gates.

   Compare ASR-only and ASR+translation source latency, source quality,
   translation latency, GPU memory, and shutdown cleanup.

## Validation

Transcript tests:

- `revision` increases monotonically;
- `stable_count` increases monotonically;
- stable segments are never changed after append;
- `partial` is replaced as a whole;
- long continuous speech advances stable history without VAD endpoint;
- VAD endpoint finalizes safe current text;
- `flush` clears current `partial` into stable history;
- `finish` emits one final source snapshot;
- no-punctuation input does not create unbounded partial growth;
- old ASR rewrites do not mutate stable history.

Translation tests:

- source transcript events are sent before translation work;
- preview translation is latest-only;
- stale preview results are dropped;
- stable translation is keyed by stable source segment id;
- preview failure is silent;
- stable translation timeout emits `translation_status`;
- queue full emits `skipped_backlog`;
- finish drops old translation backlog and waits only for finish-created stable
  segments;
- all WebSocket sends are serialized through one lock.

HY-MT smoke:

```bash
HF_HUB_OFFLINE=1 uv run python tools/smoke_hymt_translation.py \
  --model tencent/HY-MT1.5-1.8B \
  --target English
```

Report load time, generation latency, output text, device, dtype, and peak GPU
memory if available.

WebSocket E2E:

- source ASR CER;
- first source update latency;
- source update-gap p95;
- stable cursor cadence during long speech;
- partial max duration;
- first translation preview latency;
- preview refresh interval;
- stable translation p50/p95;
- stable translation coverage;
- max GPU memory;
- lingering GPU processes after shutdown.

Private audio, transcripts, and generated translation goldens stay in
`local_data/` and `local_goldens/`.

## Rejected Alternatives

### Use `committed` Instead Of `stable`

Rejected. `committed` tends to mix audio-buffer commit, VAD turn close, and text
stability. The protocol uses `stable` for the append-only prefix and `partial`
for the replace-only tail.

### Segment Lifecycle Events

Rejected as the core protocol. `created / updated / finalized` exposes internal
state transitions that the frontend does not need. A monotonic stable cursor is
the smaller abstraction.

### Only Enable `live_stability_delay_ms`

Rejected as the whole strategy. A duration threshold is useful only after the
latest ASR hypothesis repeats a prefix from the previous hypothesis. Blindly
stabilizing by time can freeze the most rewrite-prone tail.

### Wait For VAD Endpoint

Rejected. VAD detects acoustic silence, not text stability. Users can speak for
minutes without a long enough pause.

### Rewrite Stable History

Rejected for realtime v1. Full-history correction requires text diffing,
translation invalidation, and frontend merge policy. Move the stable cursor
conservatively instead.

### Translate Every ASR Hypothesis

Rejected. HY-MT generation is not free, and ASR hypotheses rewrite. Translate
only the latest `partial` preview plus stable appends.

### Put Translation In `RealtimeASRSession`

Rejected. It couples text translation to audio/VAD/ASR state and makes source
ASR regression harder.

### Store Translation In Source Transcript Segments

Rejected for v1. Translation has target-language choices, preview state,
timeouts, retries, and failures. Keep translation as separate events until
persistence or export requires a richer document model.

## Open Questions

- What target language should the local translation service default to?
- Should the UI visually distinguish `partial` from stable history?
- Is one 4090 enough for Qwen3-ASR + HY-MT under long continuous speech, or do
  we need quantized HY-MT or a separate device?
