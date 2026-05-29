# Performance Optimization

Use when adding, reviewing, or removing an optimization.

## Active Stack

Library/offline optimizations are opt-in. `realtime_server.py` defaults to the
validated single-user service profile.

| kwarg | Flag | Effect |
|---|---|---|
| `cuda_graph=True` | `--cuda-graph` | captured single-token decode |
| `flashinfer=True` | `--flashinfer` | decode attention |
| `fused_rmsnorm=True` | `--fused-rmsnorm` | `F.rms_norm` |
| `fused_linears=True` | `--fused-linears` | fused qkv and gate/up |
| `quantized_linears=True` | `--quantized-linears` | W8A16 qkv/gate_up |

Rules:

- FlashInfer only uses a custom kernel for single-token text decode. Text
  prefill and audio-encoder attention still enter the same attention dispatcher,
  then fall back to SDPA; keep this route because direct SDPA/FA2 dispatch was
  slower on the live profile.
- W8A16 requires `fused_linears=True`.
- `lm_head` stays BF16.
- Keep HND StaticCache layout.

## Numbers

RTX 4090, bf16, 60s CN median wall: base `~5s`, graph `1.89s`, FlashInfer
`1.44s`, fused RMSNorm `1.25s`, fused linears `1.18s`, W8A16 qkv/gate_up
`0.894s`.

Offline remains decode-bound. Streaming remains repeated-prefill bound until it
gets stateful reuse.

W8A16 is offline-only. Streaming is prefill-bound (decode is ~14% of a live20
step), and the W8A16 Triton GEMM (fp32 `tl.dot`) makes multi-token prefill ~3x
slower. live20 per-update steady-state: W8A16 on `~162ms` vs off `~52ms`
(flashinfer); the 80-window live20 CER gate is equal (cer_mean `0.0961` off vs
`0.0965` on, `recheck_w8a16_{on,off}.json`). The streaming service defaults
W8A16 off; a fast tensor-core (not fp32 Triton) prefill GEMM is the only way
quantization helps the streaming path.

Roofline rule (validated on three paths): **decode-bound → W8A16 helps; prefill/
compute-bound → W8A16 hurts** (fp32 Triton GEMM). (1) ASR offline = decode-bound
→ W8A16 helps (opt-in). (2) ASR streaming = prefill-bound → W8A16 hurts, off. (3)
Translation HY-MT = decode-bound → **W8A16 on gate/up + cuBLAS prefill GEMM gives
~1.12x** (per-token decode `5.22ms`→`4.39ms`; with the fp32 Triton GEMM it was a
net loss `0.82x`, so the cuBLAS GEMM path is required) and is **quality-safe vs
the stock-model golden** (1200 opus cases, en<->zh/en<->ja, chrF2): funyi is
statistically indistinguishable from stock in every direction (paired deltas
within noise), 84% byte-identical, 0 new errors; W8A16's own on-vs-off effect is
mean drop `-0.11`. See `@docs/realtime_translation_design.md` for the golden/gate.
HY-MT q/o/down (out=2048) under-occupy
the GEMV and give no gain; only gate/up (out=6144) help. The forced aligner is a
single prefill forward (no decode) → prefill-bound → do **not** apply W8A16.
Its real win is **fused RMSNorm + linears** (same patches as the ASR path),
default-on via `--timestamp-fused`: **~1.4x** on the per-segment align forward
(interleaved A/B 34.7→25.0ms, paired-ratio IQR 1.37-1.44). Not bit-identical —
bf16 argmax flips shift `<=~1%` of timestamps by `<=0.16s` (1-2 segment-time
units) with **no word-count change** (60-segment validation: 57/60 byte-identical,
max drift 0.16s) — so it diverges from the exact `_assert_same` parity golden
(an *implementation*-parity gate) but is well inside perceptual timestamp
tolerance, the same trade the ASR fused stack already makes vs CER. FA2 and a
hand-written block-local encoder attention were measured **flat** (encoder cost
is window-independent: 6s vs 30s window both ~31-34ms), so don't pursue them.
Measure the aligner forward with an **interleaved A/B** (both models resident,
alternating per call) — a sequential all-A-then-all-B run is confounded by GPU
thermal throttling and falsely reported ~1.0x.

Known-language prompts are opt-in; auto language stays default.

## Workflow

1. Profile the real path with `tools/profile_transcribe.py`,
   `tools/profile_decode_named.py`, or `tools/profile_streaming.py`.
2. Continue only if the target is at least 3% of 60s `wall_mean`, or clearly
   improves live active-update p95.
3. Micro-benchmark the exact shape under `/tmp/`; normally require at least
   1.2x speedup.
4. Add an explicit `from_pretrained(..., flag=True)` kwarg and CLI flag.
5. Run CER gates from `@docs/validation_and_regression.md`; stop if
   `delta_abs_mean > 0.3%` or `delta_abs_max > 2%` unless accepted.
6. Keep only if end-to-end median improves at least 3%.
7. Record only publishable aggregate metrics; keep raw audio-derived outputs in
   `local_goldens/` or `/tmp`.

## Dead Ends

Do not reopen without new evidence:

- FlashInfer prefill, FlashInfer BF16 GEMM, tensor-core decode
- Direct SDPA or FlashAttention 2 varlen dispatch for the audio encoder in the
  FlashInfer live profile; older 60s live checks measured no stable win and a
  same-run regression (`~10.6s` direct SDPA, `~10.35s` FA2, `~9.4s` dispatcher
  fallback)
- Audio-feature block caching under the older sliding live window; hit rate was
  low, assembled features drifted from full recompute, and audio-tower-only
  speedup was only `~1.06x`
- `torch.compile`, HF `StaticCache`, auto `max_new_tokens`
- FlashInfer/Triton `silu_and_mul`, `fused_add_rmsnorm`, RMSNorm replacement
- naive W8A16 prefill GEMM, FP8 KV/cache, FP8 or fused `lm_head`
- folding final RMSNorm into `lm_head`
- fused W8A16 `gate_up + silu * up`
- W8A16 on `o_proj`/`down_proj` (out=2048 → Triton GEMV under-occupies, ~50% of
  SMs; no decode speedup) and W8A16 on `lm_head` (~4-5% but CER `delta_abs_mean`
  ~0.315pp, over the 0.3pp gate). Reopen only with a higher-occupancy INT8 GEMV
  kernel / finer-grained lm_head quant.
- Draft-model speculative decode with Qwen3-ASR-0.6B drafting for 1.7B: text CER
  agreement is high (~98%) but **token**-level longest-common-prefix is only
  4.5-16% (0.6B and 1.7B reach the same text via different token splits), so the
  accept rate is too low to help. Reopen only with a token-aligned draft.
- Shrinking the live20 `max_window_sec`: latency is flat (~46ms) because the
  prewarmed graph is fixed-size; <=10s also wrecks CER (>20%). No win.
- Streaming incremental-prefill / cross-step KV reuse: the mechanism is proven
  CORRECT (within-step byte-exact, cross-step CER-viable `+0.25pp`; mrope
  positions are sequential so `rope_deltas=0`; FlashInfer mishandles a
  partial-query-against-cache prefill, so the increment must use SDPA). But it
  delivered **no net latency win** (76 vs 71ms, slightly slower). Root cause,
  now measured: at a steady-state 20s window the prefill sequence is **90% audio
  embeddings, only ~10% text** (260 audio vs 28 text tokens of 288). Cross-step
  KV reuse can only reuse the *text* prefix (the audio slides 0.5s/step), so it
  saves at most ~4ms of the ~42ms prefill — swamped by the `inputs_embeds` rebuild
  + SDPA-increment overhead. The audio side is its own dead-end (feature caching
  drifts, ~1.06x). So the earlier "~1.4x combined" ceiling was optimistic; the
  reusable fraction is tiny. Do not reopen for single-user; only multi-user /
  much longer text contexts would change the 90/10 split.
- First-decode-token via graph replay instead of the eager warmup forward: the
  per-step warmup runs eager with a *sliced* attention mask (variable shape, not
  graphable); on graph-hit steps it can be replaced by a full-mask graph replay.
  Proven **byte-exact** (sha256-identical transcripts), but only **~2.7% median**
  (best ~4.9%), **below the 3% keep bar**. The profiler's per-section
  `cuda.synchronize()` inflated the eager-warmup estimate (22ms) that made it look
  like ~1.3x; un-profiled the warmup overlaps and is cheap. Reverted.
