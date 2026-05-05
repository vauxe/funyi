# Streaming Runtime

Use this when changing `streaming_transcribe`, live-caption behavior, or
streaming benchmark/gate logic.

For the single-user WebSocket/VAD service design around this runtime, see
`@docs/realtime_asr_service.md`.

## Default Streaming

Default streaming remains full-audio re-feed and upstream-compatible. It is
validated by `local_goldens/streaming_regression.json` and the commands in
`@docs/validation_and_regression.md`.

Do not use optimized flags as the local streaming golden source. If an
optimization changes text, validate final text through streaming CER instead of
hash equality.

## Bounded Live Mode

`init_streaming_state(max_window_sec=...)` enables bounded live-caption mode. It
changes streaming semantics intentionally:

- `audio_accum` is capped to the recent audio window.
- old stable prefix text moves to `committed_text`.
- `partial_text` remains mutable.
- public `state.text` is `committed_text + partial_text`.
- when `max_window_sec` is set and `max_prefix_tokens` is omitted, runtime uses
  `max_prefix_tokens=192`.

Display duration and model context are separate: UI captions can show 1-2
lines / 2-6s while the model keeps 30s or 45s of context.

Recommended starting point:

| Setting | Value |
|---|---|
| `step_ms` | 1000 or 2000 |
| `chunk_size_sec` | 2.0 (default) / 1.0 (low-latency preset below) |
| product default `max_window_sec` | 30.0 |
| strict-quality `max_window_sec` | 45.0 |
| low-latency single-user `max_window_sec` | 20.0 |
| sweep/probe upper bound | 60.0 |

## Live-Window Results

Measured CN sweeps on 167 60s windows, compared with the same-`step_ms=1000`
full-audio re-feed `opt_nograph` baseline:

| Path | CER mean | p50 wall | active-update p95 mean | CER abs-delta mean | CER abs-delta max | CER worse-delta max |
|---|---:|---:|---:|---:|---:|---:|
| full re-feed baseline | 9.41% | 8.399s | 0.5093s | - | - | - |
| `max_window_sec=20` | 9.47% | 6.396s | 0.3751s | 0.359% | 2.373% | 2.373% |
| `max_window_sec=30` | 9.40% | 6.272s | 0.3663s | 0.251% | 2.135% | 2.135% |
| `max_window_sec=45` | 9.38% | 6.758s | 0.3989s | 0.104% | 2.113% | 0.714% |

Use 30s as the product starting point. Use 45s when quality regression is more
important than minimum latency. Keep gates separate:

- `abs-delta`: behavior drift, including better-CER windows.
- `worse-delta`: quality regression only.

Live45 fails an offline-style `abs max <= 2%` gate because one window improves
by 2.113%, but it passes the regression gate with `worse max=0.714%`.

Measured against non-graph live30 on the same 167 windows:

| Path | CER mean | p50 wall | active-update p95 mean | CER abs-delta mean | CER abs-delta max |
|---|---:|---:|---:|---:|---:|
| `live30 + cuda_graph` | 9.43% | 6.617s | 0.2734s | 0.095% | 2.000% |

Use live30 + CUDA graph only when update tail latency matters more than
batch-style total wall: active-update p95 mean improved by about 25%, while
total p50 wall was slower than non-graph live30.

## Commands

Streaming profile:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/profile_streaming.py \
  --step-ms 1000 --max-window-sec 30 \
  --flashinfer --fused-rmsnorm --fused-linears
```

Live benchmark:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/benchmark_streaming.py \
  --cases short_default_15s --step-ms 1000 --repeats 15 \
  --max-window-sec 30 \
  --flashinfer --fused-rmsnorm --fused-linears
```

Live CER sweep:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_streaming_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths opt_nograph --window-sec 60 --num-windows 200 --step-ms 1000 \
  --max-window-sec 30 \
  --flashinfer --fused-rmsnorm --fused-linears --timed \
  --output /tmp/streaming_cer_live30_opt_nograph.json
```

Current stable streaming optimized baseline, without CUDA graph:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_streaming_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths opt_nograph --window-sec 60 --num-windows 200 --step-ms 2000 \
  --flashinfer --fused-rmsnorm --fused-linears --timed \
  --output /tmp/streaming_cer_opt_nograph_all_fused.json
```

Current measured CN baseline: mean CER 9.41%, p50 wall 6.505s/window, first
text at 2000ms, active-update p95 mean 0.3811s.

## Speculative Verification

`init_streaming_state(spec_decode=True)` replaces the per-step rollback decode
with a verifier prefill over `prompt + rollback_draft`. Accepted draft tokens
skip the per-token decode loop; the first rejected position uses the
verifier's argmax and the decode loop continues from there.

Not byte-identical under bf16. Prefill-path KV differs from decode-path KV by
bf16 ε, which propagates 28 layers and can flip argmax at low-margin positions
(homophones, punctuation). Quality-gated by streaming CER, not hash equality.

Measured local CN sweep, 167 60s windows, step_ms=1000, live30,
`flashinfer + fused_rmsnorm + fused_linears`, `spec_decode=True`. Three paths,
same audio, same machine, back-to-back. Save matching review artifacts under
`local_goldens/cer/` when rerunning with validation audio you are allowed to
use.

| Path | CER mean | wall mean | wall p90 | wall max | active-update p95 mean | delta vs base |
|---|---:|---:|---:|---:|---:|---:|
| base                | 9.397% | 8.709s | 11.07s | 14.39s | 463.2ms | - |
| `spec_decode`       | 9.405% | 7.904s | 9.82s  | 11.88s | 441.8ms | abs_mean 0.042%, abs_max 1.379% |
| `spec_decode` + `cuda_graph` | 9.400% | 5.953s | 6.53s | 7.15s | 254.8ms | abs_mean 0.062%, abs_max 1.379% |

Passes the CN optimized gate (`abs_mean <= 0.3%`, `abs_max <= 2%`). Aggregate
wall across the 167 windows dropped from 1454s (base) to 1320s (spec, -9%) to
994s (spec+graph, -32%). 146/167 spec+graph windows remain byte-equal vs base;
diverging windows are dominated by homophone and punctuation flips inherited
from the bf16 ε drift described above, not from the graph tail. Historical
`lm_head` W8A16 rows were removed. The current qkv/gate_up-only W8A16 golden
has mean CER 9.378%; vs spec+graph, `delta_abs_mean=0.086%` and
`delta_abs_max=1.379%`. Wall mean is 5.953s -> 3.793s and active-update p95
mean is 254.8ms -> 169.7ms. The forced-Chinese W8A16 golden has identical final
CER to W8A16 auto-language, wall mean 3.777s, and active-update p95 mean
166.5ms.

Use `spec_decode` + `cuda_graph` together when both total wall and update tail
latency matter. The speedup from stacking them compounds: spec cuts prefill+
decode count, graph cuts per-step CPU/launch overhead in the tail decode loop.

Implementation notes:

- Draft ids come directly from the token suffix rolled back by
  `_build_streaming_prefix_plan`. Prefix trimming may still commit old text, but
  that no longer disables speculative verification; the verifier treats the
  rolled-back token suffix as a draft under the trimmed prompt and accepts only
  the greedy-matching prefix.
- `ASRStreamingState.spec_decode_stats` tracks attempts, trimmed-prefix
  attempts, verified/accepted draft tokens, and no-draft steps for reuse-rate
  profiling.
- When `cuda_graph` is enabled the backend routes
  `infer_streaming_with_draft` to `CudaGraphDecoder.generate_with_draft`,
  which runs the verifier prefill with `DynamicCache`, crops to
  `prompt_len + accepted`, copies KV into the `StaticCache`, and replays the
  captured graph for the tail decode loop. The warmup+capture+decode tail is
  shared with the non-spec path (`_decode_after_prefill`), so the graph
  capture behavior is identical whether spec is on or off.

## Low-Latency Single-User Preset

For single-user live captioning where the GPU is not shared and perceptual
latency matters more than aggregate wall or byte-regression, use:

```python
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    cuda_graph=True, flashinfer=True,
    fused_rmsnorm=True, fused_linears=True,
    dtype=torch.bfloat16, device_map="cuda:0",
)
state = model.init_streaming_state(
    chunk_size_sec=1.0,   # update cadence
    unfixed_chunk_num=2,  # keep default; =1 destabilizes partial text
    unfixed_token_num=5,  # keep default rollback
    max_window_sec=20.0,  # shorter context = cheaper prefill
    max_prefix_tokens=64, # service-tuned rolling text prefix cap
    spec_decode=True,
)
# or, equivalently:
# state = model.init_streaming_state(**Qwen3ASRModel.low_latency_preset_kwargs())
```

`Qwen3ASRModel.low_latency_preset_kwargs()` returns the streaming kwargs
above. The loading-side flags (`cuda_graph`, `flashinfer`, `fused_*`) stay
explicit on `from_pretrained` -- the preset deliberately does not flip them
globally, since they are runtime load options, not streaming parameters.

Perceptual latency budget:

- first-caption: `unfixed_chunk_num * chunk_size_sec + inference + render`
  ~= 2.0s + 0.21s + UI ~= ~2.3s
- steady-state avg: `chunk_size_sec/2 + inference + render` ~= 0.75s
- steady-state worst: `chunk_size_sec + inference + render` ~= 1.3s

Measured local CN sweep, 167 60s windows, step_ms=1000:

| Path | CER mean | wall mean | wall p90 | active-update p95 mean | active-update p95 max | first-text mean |
|---|---:|---:|---:|---:|---:|---:|
| live30 chunk=2s (default) | 9.40% | 5.95s | 6.53s | 255ms | 316ms | 2000ms |
| **live20 chunk=1s (low-latency preset)** | **9.46%** | 10.00s | 10.65s | **212ms** | **258ms** | **1012ms** |

Tradeoffs vs the default live30 path:

- first-text -49%, active-update p95 mean -17%, p95 max -18%
- aggregate wall +68% (twice as many inference steps per minute), but single
  GPU utilization only rises to ~17% per 60s real-time on a 4090
- CER mean +0.06pp, `delta_abs_mean 0.39%`, `delta_abs_max 2.94%` (one window
  exceeds the 2% streaming gate); drift is two-sided (55 worse / 42 better /
  70 byte-equal), not a quality regression. Do not use this preset as a
  hash-regression or strict-CER-gate path.
- The local WebSocket service enables the transformers/CUDA path by default.
  W8A16 currently means qkv/gate_up only. A local 600s service gate passed with
  CER 7.13%, speed 10.82x, update-gap p95 158ms, max RSS 1987MB, and max
  whole-GPU delta 776MB from check start. Rerun multi-window service gates on
  your own validation audio before changing the preset.
- The service preset pins `max_prefix_tokens=64`. In a local 600s real-audio
  prefix-cap probe, 64 beat 128 and 192 on direct streaming wall (`37.48s` vs
  `39.40s` / `39.35s`) and CER (`6.87%` vs `7.09%` / `7.09%`). WebSocket/VAD
  E2E is mostly flat on throughput, but the same cap passed repeated 600s
  service gates with CER `7.20%` and update-gap p95 around `140-162ms`.

For transformers/CUDA live-caption presets, do not decrease `chunk_size_sec`
below 1.0 or `max_window_sec` below 20 without a fresh CER sweep -- neither
range has been characterized there and `unfixed_chunk_num=1` is known to
destabilize `state.partial_text`.

### Why this is not the code default

Code defaults in `qwen3_asr_runtime/model.py` stay at
`chunk_size_sec=2.0`, `max_window_sec=None`, `spec_decode=False`, and
`from_pretrained` leaves every optimized flag off. Changing these would
violate the AGENTS.md invariants:

- "Default offline path must remain upstream-compatible and byte-regressed
  by `local_goldens/offline_regression.json`."
- "Streaming default path remains full-audio re-feed unless `max_window_sec`
  is explicitly set for bounded live captions."

Switching the default to this preset would break the streaming hash
regression golden, force all callers onto bf16-non-deterministic spec
decoding, and push the bounded-live semantics (`committed_text` /
`partial_text` split, explicit 64-token service prefix cap) onto callers who don't need
them. Callers who want the preset opt in explicitly with the arguments
above, or via `Qwen3ASRModel.low_latency_preset_kwargs()` for the
streaming-side kwargs.

## CUDA Graph Note

`init_streaming_state()` resets the decode runtime so a new live session does
not inherit the previous session's CUDA graph, StaticCache, or decode buffers.
Without that reset, live30 graph sweeps OOMed on the second 60s window because
the previous session kept max-window graph/cache memory resident.

Known rejected live-cadence probes:

- `chunk_size_sec=3`: full 167-window sweep was slightly slower than 2s and
  delayed first text to 3000ms.
- `unfixed_token_num=3`: 20-window smoke was slower than the default 5-token
  rollback and had larger behavior drift.
- live30 graph with `max_new_tokens=64`: 20-window smoke was slower than the
  default 512-token cap.
- live30 graph with `max_prefix_tokens=64`: 20-window smoke was much slower
  and had worse quality; short prefixes made the model rewrite too much text.
  That live30 result does not apply to the current live20 service profile.
- live30 graph length bucketing: `bucket=32` worsened update p95, and
  `bucket=256` made generation much slower by increasing full-cache work.
- Naive incremental decoder KV reuse across growing audio: exact reuse requires
  the cached prefix embeddings and RoPE positions to be unchanged. In this
  prompt layout only the text before `<|audio_pad|>` satisfies that invariant;
  it is 9 tokens in the empty-context prompt. The audio prefix does not satisfy
  it: 16s->18s showed old/new audio encoder prefixes differ
  (`audio_features` max diff ~0.10), and reusing any audio KV prefix, even 4s,
  changed greedy generation at token 8 (`没有悬念` -> `还有悬念`).
- Frozen-block incremental audio/KV reuse is an approximate new streaming
  semantic, not an upstream-compatible optimization. Splitting full 18s mel
  features into 800-frame blocks kept this smoke's generated tokens identical,
  but freezing 16s blocks and appending an 18s tail changed upstream text at
  token 8. Even under the frozen-block semantic, bf16 full-prefill vs
  prefix-cache+tail continuation diverged at token 10. Do not add this as a
  runtime path without a dedicated CER gate and a clear quality tradeoff.
- reserved StaticCache / persistent graph across streaming steps: the
  streaming profiler showed that `_capture_graph` fires every step because
  `prompt_len` grows, each capture costing ~40ms (side-stream warmup + real
  capture). Pre-allocating the StaticCache and buffers at an upper-bound
  `max_len` (e.g. `audio_tokens_upper + text_prompt + max_prefix_tokens +
  max_new_tokens`) so the graph is captured once and replayed seems to work
  for the first 10-15 steps (~115ms/step steady state), but keeps verifier
  prefill intermediates out of the graph's allocator pool. Without per-step
  cache reallocation, memory grows ~300MB/step; at step ~20 allocator
  spills into unified memory and a single prefill balloons to 46s. Pre-
  existing per-step reallocation was doing double duty as an allocator
  reset. Re-attempting this needs explicit pool-handle management, not
  just a reserve flag.
