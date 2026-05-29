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
  delivered **no net latency win** (76 vs 71ms, slightly slower): it still runs
  the encoder full each step, adds `inputs_embeds` overhead, and only shrinks the
  text-prefill — which is not the dominant cost once decode is large. Reopen only
  as a *combined* incremental-prefill + spec-decode (+ encoder-feature cache)
  build, ceiling ~1.4x (capped by the still-full encoder), and only if multi-user
  concurrency / longer windows make the prefill cost matter (single-user live20
  already has ~10x realtime headroom after W8A16-off).
