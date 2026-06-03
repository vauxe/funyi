# Realtime Translation Pipeline

Status: source transcript events, a synchronous HY-MT adapter, the optional
WebSocket translation runtime, and the subtitle replay model are wired.
Model, prompt, or decode-path changes still need the translation quality gate.

Purpose: describe how optional bilingual subtitles run above `/ws/asr` without
changing ASR behavior. The public WebSocket command and event contract stays in
`@docs/realtime_asr_service.md`; this document covers translation runtime
scheduling, finish semantics, and quality gates only.

## Boundary

Translation is a service-layer side track. It consumes emitted transcript events
and does not modify `Qwen3ASRModel`, `RealtimeConnectionSession`,
`TranscriptStore`, or the source segment schema.

v1 has two paths:

- stable: `stable_appends -> stable unit queue -> TranslationModelActor ->
  translation_stable`;
- preview: `pending stable unit + partial -> latest-only debounce -> TranslationModelActor ->
  translation_preview`.

Stable translations are durable. Preview translations are temporary, replaceable,
and never enter history or export.

Out of scope:

- token-level HY-MT streaming;
- cross-request prefix cache;
- ASR backend or stabilization changes.

## Flow

```text
audio frames
  -> RealtimeConnectionSession
  -> transcript_update / transcript_final
  -> update TranslationRuntime scheduler state without waiting for the model
  -> send source events immediately

TranslationRuntime
  -> TranslationModelActor
  -> HYMTTranslator.translate(...) / translate_batch(...)
  -> event_queue

sender task
  -> websocket.send_text(...)

client SubtitleDocument
  -> previous stable line / current draft line / SRT history
```

Only the sender task writes to the WebSocket.

The service gives the translation scheduler each source event before queueing
that source event, so stale preview work can be cancelled before clients see a
newer source revision. Current translation results are queued later and never
block audio ingest. When the aligned ASR CUDA graph path is prewarmed, ASR graph
replay and HY-MT generation may overlap; graph capture and optimization details
live in `@docs/performance_optimization.md`.

## Public API Surface

`@docs/realtime_asr_service.md` owns target selection, capability errors, and
the payloads for `ready.translation`, `translation_preview`,
`translation_stable`, and `translation_status`. The runtime receives a
normalized `target_language` and treats `None` as translation disabled.

Never expose stack traces or model internals through translation status text.

## Runtime Rules

For every `transcript_update`, the service loop is:

```text
update translation scheduler state
stage each stable_appends item into a stable translation unit
if non-empty partial text exists: update_preview(revision, partial)
else: cancel_preview(revision)
queue source event
```

Stable, while a target language is active:

- stable history is reliable: every source stable segment is covered by exactly
  one `translation_stable` before `transcript_final`, unless the translator
  fails;
- adjacent source stable segments may be grouped into one stable translation
  unit while a replaceable `partial` tail is still active; the event anchors to
  the final covered source segment and carries `source_segment_ids` /
  `source_segment_indices` for full coverage;
- stable units close when the current `partial` tail clears, or during
  `finish`; punctuation is not treated as proof that a translation unit is
  complete;
- stable generation runs through one model actor; when
  `--translation-stable-batch-size` is greater than `1`, adjacent queued stable
  jobs with the same source language and target language may share one
  `translate_batch` call;
- stable jobs are not dropped for backlog pressure and are not timed out by the
  service;
- translator failures emit `translation_status` for the affected stable
  translation unit.

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
- while stable source text is pending inside the current translation unit,
  `translation_preview` should be displayed against the composed source text
  `pending stable source text + partial.text`;
- `translation_stable` / stable `translation_status` should prefer
  `source_segment_ids` / `source_segment_indices` over the anchor fields and
  fold the covered stable source segments into one displayed translation unit;
- the anchor `source_segment_id` / `source_segment_index` is a compatibility
  lookup, not the full source coverage when lists are present;
- `translation_preview` annotates the current draft only when `source_revision`
  matches;
- the compact subtitle window renders `stable_lines[-1]` above `current` below
  and lets the client layout constrain the visible text;
- SRT/detail output uses stable history only, with translation as a second line
  in the same SRT entry when translation display is enabled.

Scheduling:

- audio ingest and source event sending never wait for translation;
- preview has priority over normal stable backlog because it is the lowest
  latency translation path;
- preview work that has not entered the model can be superseded by newer state;
- a preview model call that already entered HY-MT cannot be preempted and may
  delay later stable history or `finish`;
- stable backlog runs when no preview is ready and is drained during `finish`;
- stable batching preserves output event order and never batches across source
  language or target language;
- finish-created stable jobs have priority during `finish`;
- if a stable job is already running when preview arrives, do not cancel the
  model actor call; drop the preview if it is stale by completion time;
- stable and preview share one `TranslationModelActor`, so the same translator
  instance is never entered concurrently;
- preview timeouts only discard the result; they do not interrupt the model
  call already running on the actor;
- HY-MT calls share one actor thread, so translation model ownership is
  serialized even when ASR is running concurrently.

## Finish

Translation-enabled `finish`:

```text
run session.finish()
send finish-created transcript_update events
enter translation finish mode
cancel pending or logically running preview work
wait for any already running stable translation to publish once
translate finish-created stable units before queued stable jobs
send translation_stable or translation_status for those stable jobs
send transcript_final
close WebSocket
```

Do not try to cancel the model actor thread already inside
`HYMTTranslator.translate`.
Running stable jobs are not retranslated; they publish once before
`transcript_final`. Already-running preview model calls may continue on the
model actor, and stable finish work waits for the actor before publishing.
Preview results after finish are ignored. Source-only mode keeps the current
`session.finish()` behavior and simply has no translation runtime to drain.

## Translator

`HYMTTranslator.translate(text, *, target_language, source_language="",
max_new_tokens=512) -> str` remains synchronous. `translate_batch(...)` is an
optional stable-batching path with the same target/source language contract.
Runtime calls both through the single `TranslationModelActor` thread. Load the
model once at startup and never download weights from request handling.

Default runtime path when translation is enabled:

- model: `tencent/Hy-MT2-1.8B`;
- attention: `sdpa`;
- decode backend: `fixed_mask`;
- `trust_remote_code`: off; the pinned Transformers dependency has native
  `hunyuan_v1_dense` support;
- W8A16 on gate/up and fused RMSNorm: on, validated against the Hy-MT2 stock
  golden.

Prompt, sampling, tokenizer/model, or decode-path changes require the
translation quality gate. Protocol/runtime-only changes do not.

## Validation

Protocol/runtime changes use the focused fake-translator, service-ordering, and
subtitle replay tests, plus WebSocket E2E when `realtime_server.py` behavior or
ordering changes. Exact commands live in `@docs/validation_and_regression.md`.

Translation quality gate (`tools/gate_translation.py`), only for
model/prompt/generation/decode changes. Like the ASR CER gate, the truth is an
official-code golden generated by the UNMODIFIED stock model, not funyi gating
itself.

- Eval set: `local_data/opus_mt_eval.jsonl` (1200 cases, en<->zh and en<->ja from
  opus-100 test; regenerate with `tools/fetch_opus_eval.py`, pinned to an
  immutable opus-100 commit). zh<->ja is not in opus-100 and stays on the 42-case
  in-domain set `local_data/translation_perf_cases.jsonl`.
- Golden: `tools/gen_translation_golden.py` runs the stock `transformers` model
  (plain `generate()`, greedy, bf16/sdpa) with NONE of funyi's runtime -- no
  `fixed_mask` custom decode, static cache, `logits_to_keep`, dynamic-rope
  surgery, or W8A16 -- sharing only the prompt and tokenization. funyi's runtime
  is gated against this golden, so the gate proves funyi reproduces the ORIGINAL
  model, not merely that it is self-consistent.
- opus-100 references are loose subtitle alignments, so absolute chrF is
  unreliable; the gate uses only the paired delta vs the golden, which cancels
  reference noise.

What it gates, in `--quality-baseline-json` (baseline-regression) mode:

- no new gross errors versus the baseline: target language, empty output, length
  outliers, repetition loops, required structural markers;
- `must_preserve` items (protocol labels, fixed UI strings, numbers, units,
  source segment ids);
- per-direction mean chrF drop (chrF2 vs reference) above `--max-mean-chrf-drop`:
  the gate fails if ANY single direction's mean drop exceeds the threshold.
  Gating per direction, not on a pooled mean, keeps a regression confined to one
  direction (the dominant real failure mode) from being diluted by the others.
  Per-case chrF drops are recorded and counted (`_CASE_CHRF_DROP_FLAG`, 10 pts)
  for human review but never fail the gate alone: single-reference chrF is too
  noisy on short sentences. Use the aggregate gate only on the large set
  (>=~200 cases/direction); on the 42-case set per-direction means are too noisy.

Guards: the baseline's `reference_metric` must match (`chrf2`) or the gate fails
rather than silently comparing across metrics; if `--max-mean-chrf-drop` is set
but no case is comparable the gate fails (`no_comparable_chrf_cases`) instead of
passing vacuously; and a `run_config_diff` (dtype / attn / decode-backend /
generation differing from the baseline) is surfaced as a non-failing warning so a
delta is not mistaken for a model change when it is really a config change. chrF2
here is a self-contained char-n-gram F (whitespace-stripped); only the paired
delta is used, never the absolute value (opus refs make absolutes meaningless).

Workflow (generate the stock golden once, then gate the runtime profile against
it, then human-read the biggest movers):

    # eval set, from a pinned opus-100 commit; only needed when the local file
    # is missing or intentionally refreshed
    uv run --with pyarrow python tools/fetch_opus_eval.py \
      --output local_data/opus_mt_eval.jsonl
    # immutable Hy-MT2 commit; use the Hugging Face id when passing a revision
    MODEL=tencent/Hy-MT2-1.8B
    MODEL_REVISION=0123456789abcdef0123456789abcdef01234567
    # golden, from the unmodified stock model (regenerate when the eval set or
    # upstream model revision changes)
    uv run python tools/gen_translation_golden.py \
      --model "$MODEL" \
      --model-revision "$MODEL_REVISION" \
      --allow-download \
      --dataset local_data/opus_mt_eval.jsonl \
      --output local_goldens/translation/opus_mt2_official_golden.json
    # default HYMTTranslator profile gated against the stock golden
    uv run python tools/gate_translation.py \
      --model "$MODEL" \
      --model-revision "$MODEL_REVISION" \
      --allow-download \
      --dataset local_data/opus_mt_eval.jsonl \
      --quality-baseline-json local_goldens/translation/opus_mt2_official_golden.json \
      --max-mean-chrf-drop 0.5 \
      --worst-output local_goldens/translation/opus_mt2_funyi_vs_golden_worst.json

For a pre-downloaded local directory, first download the same revision to
`local_data/models/Hy-MT2-1.8B`, then run both model commands with
`--model local_data/models/Hy-MT2-1.8B` and omit `--model-revision`. Existing
local paths are treated as explicit contents; the revision flag is only applied
to Hugging Face model ids.

`--worst-output` writes the largest chrF-drop cases with source / reference /
golden / candidate side by side, because the metric cannot tell a benign rephrase
from a real nuance loss — a human reads those. Each gate row stores its `output`
text so the comparison is possible at all.

The service uses `--translation-max-new-tokens 256` as a latency/runaway cap.
The model quality gate above uses the translator default `512` unless that cap is
the change being tested.

Current Hy-MT2 reference point (1200 cases): default optimized translator
profile passes the stock golden gate with 0 new errors, one quality-gate length
warning, 8 notably changed cases, and mean chrF drop `0.0279` (threshold `0.5`).
The expected non-failing `run_config_diff` warning is separate. Read the
`--worst-output` dump per direction before trusting a new run -- the metric
cannot see adequacy.

Private audio, transcripts, and generated outputs stay in `local_data/`,
`local_goldens/`, or `/tmp`.
