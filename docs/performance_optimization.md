# Performance Optimization

Use this when adding or reviewing an optimization.

## Optimized Decode Stack

For direct library/offline use, optimizations are opt-in through
`TransformersASRBackend.from_pretrained` and surfaced as CLI flags on
`tools/benchmark_offline.py` and `tools/sweep_cer_vs_srt.py`.
`realtime_server.py` is different: it defaults to the validated single-user
GPU service profile and exposes `--no-*` flags only for fallback and comparison.

| kwarg | flag | effect |
|---|---|---|
| `cuda_graph=True` | `--cuda-graph` | `CudaGraphDecoder`: StaticCache + captured decode step |
| `flashinfer=True` | `--flashinfer` | FlashInfer single-token decode attention |
| `fused_rmsnorm=True` | `--fused-rmsnorm` | Replace hand-rolled RMSNorm with `F.rms_norm` |
| `fused_linears=True` | `--fused-linears` | Fuse q/k/v and gate/up projections |
| `quantized_linears=True` | `--quantized-linears` | W8A16 GEMV/GEMM for fused qkv/gate_up |

Important notes:

- FlashInfer dispatch is decode-only; prefill/audio encoder fall back to SDPA.
- W8A16 linears require `fused_linears=True`. qkv/gate_up use int8 GEMV for
  `B=1, L=1` and int8 GEMM for L>1 without retaining BF16 fallback weights.
  `lm_head` stays BF16 because it directly controls greedy token selection.
- FlashInfer requires setting `_attn_implementation = "flashinfer"` on
  thinker/text/audio sub-configs.
- Keep the direct HND StaticCache path. Old NHD conversion measured slower on
  same-process 60s A/B with identical text.

## 4090 Numbers

bf16, SDPA baseline, median wall per 60s CN window:

| Path | wall |
|---|---:|
| runtime base | ~5s |
| `+ cuda_graph` | 1.89s |
| `+ flashinfer` | 1.44s |
| `+ fused_rmsnorm` | 1.25s |
| `+ fused_linears` | 1.18s |
| `+ W8A16 linears` | 0.894s |

## Current Bottleneck

For 60s CN windows, the optimized offline path is still decode-bound. Current
W8A16 quantizes qkv/gate_up only; `lm_head` remains BF16 to keep greedy token
selection stable.

## Known-Language Prompt

When the caller already knows the language, forcing it in the prompt is a small
opt-in latency win because the model no longer needs to generate the language
header. On the current W8A16 CN sweep, `--language Chinese` measured mean CER
9.64% and p50 wall 0.878s, vs auto-language mean CER 9.62% and p50 wall
0.894s. Treat this as an API-level opt-in for known-language workloads, not a
default: auto language detection must remain upstream-compatible.

## Optimization Workflow

1. Profile first:

   ```bash
   uv run python tools/profile_transcribe.py
   uv run python tools/profile_decode_named.py \
     --audio "$ASR_AUDIO" \
     --flashinfer --fused-rmsnorm --fused-linears
   uv run python tools/profile_streaming.py --flashinfer --fused-rmsnorm --fused-linears
   ```

   Stop unless the target region is at least 3% of 60s `wall_mean` or, for
   live streaming, clearly improves active-update tail latency.

2. Micro-benchmark the exact shape under `/tmp/`. Use CUDA events and compare
   `max_diff` / `mean_diff`. Stop unless the candidate is at least 1.2x faster.

3. Integrate behind a new explicit `from_pretrained(..., flag=True)` kwarg and
   matching CLI flag. Default behavior must stay byte-identical.

4. Run CN CER gate:

   ```bash
   uv run python tools/sweep_cer_vs_srt.py \
     --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
     --paths graph --window-sec 60 --num-windows 200 \
     --flashinfer --fused-rmsnorm --fused-linears --<your-flag> \
     --output /tmp/cer_candidate.json
   uv run python tools/merge_cer_sweeps.py \
     --input local_goldens/cer/cer_base.json=base \
             /tmp/cer_candidate.json=candidate \
     --output /tmp/cer_compare.json
   ```

   Stop if `delta_abs_mean > 0.3%` or `delta_abs_max > 2%`. Repeat on JA
   stripped data as language coverage.

5. Confirm end-to-end wall. Keep the change only if median improves at least 3%
   on short and medium windows.

6. Commit measured CER and wall numbers. If rejected, add a one-line dead-end
   note below.

## Known Dead Ends

Do not revive without new evidence:

- FlashInfer prefill: one-layer parity held, but real multi-layer Qwen3-ASR
  output diverged badly.
- `torch.compile` on thinker: no net win on captured decode.
- Auto-estimating `max_new_tokens`: no measurable CUDA-graph wall change.
- HF `StaticCache` via `cache_implementation="static"`: slower on long audio.
- FlashInfer/Triton `silu_and_mul`: micro-bench win did not survive integration.
- FlashInfer `fused_add_rmsnorm`: CER passed but full-window p50 gain was below
  the 3% keep threshold.
- FlashInfer RMSNorm replacing `F.rms_norm`: no 1.2x micro-bench win.
- `single_decode_with_kv_cache(..., use_tensor_cores=True)`: slower on
  short/mid KV, long-KV gains too small.
- Chunked EOS checking in `CudaGraphDecoder`: text identical but slower.
- FlashInfer BF16 GEMM: unusable on current 4090/SM89 stack due unsupported
  CUTLASS path and cuDNN `libcudart` conflict.
- Naive W8A16 prefill GEMM as a speed-only change: slower at the real 60s
  prefill shape `B=1, L=795, hidden=2048`. The kept W8A16 GEMM is for the
  lower-VRAM qkv/gate_up path, where not retaining BF16 weights matters.
- KV-cache FP8/quantization on FlashInfer decode: BF16 single-token attention
  was faster at the measured 60s shapes (`0.026-0.038ms`) than native FP8 KV
  (`0.117-0.130ms`) or dequant-to-BF16 attention (`0.100-0.134ms`), and updating
  one quantized KV slot cost about `0.213ms`.
- FP8 prefill `lm_head`: the isolated `_scaled_mm` kernel was faster, but the
  20-window smoke was byte-identical and did not improve end-to-end wall after
  excluding the cold first window (`0.968s -> 0.972s`).
- FP8 decode GEMV/GEMM experiments: slower than the current W8A16 Triton GEMV
  at decode shape.
- Fused `lm_head` top-1: matched argmax in a micro-bench but was slightly
  slower than separate logits plus `argmax`.
- Folding final RMSNorm weight into `lm_head`: small 20-window smoke speedup
  after excluding the cold first window (`0.961s -> 0.955s`), but only `10/20`
  windows were byte-identical and the gain is below the keep threshold.
- Fused W8A16 `gate_up + silu * up`: a micro-bench win did not survive real
  transcription; 20-window smoke was effectively unchanged after excluding the
  cold first window (`0.961s -> 0.961s`) and was not byte-identical.
