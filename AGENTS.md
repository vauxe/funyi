# Agent Notes

Keep this file short. It is the startup checklist only.

## Goal

Build a runtime Qwen3-ASR around the upstream `transformers` backend, without
depending on upstream package code at runtime. Preserve upstream behavior by
default for library/offline paths; the local service entrypoint may enable the
validated single-user profile by default.

## Invariants

- Runtime scope is transcription only.
- Public releases must not include private audio, transcripts, or
  audio-derived goldens.
- Keep local validation audio in `local_data/` and generated goldens/CER sweeps
  in `local_goldens/`; both are ignored by git.
- Default offline path must remain upstream-compatible and should be
  byte-regressed against a local runtime-default golden before release.
- Optimized paths are quality-gated by punctuation-stripped CER vs SRT generated
  from validation audio you are allowed to use.
- Streaming default path remains full-audio re-feed unless `max_window_sec` is
  explicitly set for bounded live captions.

## Current Baseline

- Offline all-fused stack: `cuda_graph + flashinfer + fused_rmsnorm +
  fused_linears`; measure current quality and latency with local CER sweeps.
- Offline W8A16 is opt-in on top of all-fused; it is not default.
- Known-language prompts are opt-in. Keep auto language as the default
  upstream-compatible path.
- Streaming live window: 30s is the product starting point; 45s is stricter
  quality mode. Keep `abs-delta` behavior drift separate from `worse-delta`
  quality regression.
- Streaming `live30 + cuda_graph` is stable after decode-runtime reset; use it
  for lower update p95, not for lower total p50 wall.
- Streaming `spec_decode=True` + `cuda_graph=True` is the current best live30
  path from internal validation. It is not byte-identical under bf16; do not use
  it as a hash-regression path.
- Low-latency single-user preset: `chunk_size_sec=1.0 + max_window_sec=20 +
  spec_decode + cuda_graph + all fused`. See @docs/streaming_runtime.md
  "Low-Latency Single-User Preset" for the full recipe.
- Local realtime service default: live20 + `spec_decode + cuda_graph +
  flashinfer + fused_rmsnorm + fused_linears + W8A16`, with VAD-ended committed
  segments and no server caption export. W8A16 currently means qkv/gate_up.

## Important Paths

- `qwen3_asr_runtime/model.py` — public offline/streaming wrapper
- `qwen3_asr_runtime/backends/transformers.py` — main backend and opt-in flags
- `qwen3_asr_runtime/hf_qwen3_asr/` — local HF model/config/processor implementation
- `qwen3_asr_runtime/decode_runtime.py` — CUDA graph decode loop
- `qwen3_asr_runtime/spec_decode.py` — speculative verification decode for streaming
- `qwen3_asr_runtime/flashinfer_attention.py` — FlashInfer decode attention
- `qwen3_asr_runtime/quant_linears.py` — W8A16 linears
- `qwen3_asr_runtime/fused_rmsnorm.py`, `fused_linears.py` — fused opt-ins
- `realtime_server.py` — local WebSocket ASR service entrypoint
- `qwen3_asr_runtime/realtime_session.py`, `vad.py`,
  `transcript_store.py` — service session state, VAD, and in-memory segments
- `tools/run_regression.py`, `tools/sweep_cer_vs_srt.py`,
  `tools/merge_cer_sweeps.py` — offline correctness and CER gates
- `tools/run_streaming_regression.py`, `tools/benchmark_streaming.py`,
  `tools/sweep_streaming_cer_vs_srt.py` — streaming gates and live-window sweeps
- `tools/ws_e2e_leak_check.py` — WebSocket service E2E and resource check

## Read On Demand

- `@docs/validation_and_regression.md` — exact commands for public smoke,
  local offline/streaming hash regression, and CER sweeps.
- `@docs/streaming_runtime.md` — streaming state model, bounded live windows,
  live20/live30/live45 results, and streaming quality gates.
- `@docs/performance_optimization.md` — optimized decode stack, 4090 numbers,
  dead ends, and the optimization workflow.
- `@docs/qwen3_asr_1_7b_architecture.md` — model architecture reference.

## Environment

- Dependency manager: `uv`, project mode non-package (`[tool.uv] package = false`).
- Default deps include the local service stack, Silero VAD, Torchaudio,
  FlashInfer, and Ninja so `realtime_server.py` starts with default parameters
  after `uv sync`.

## Caveats

- Do not use `run_regression.py` to validate optimized flags; use CER gates.
- Do not treat `prefill_chunk` as a correctness path.
- Prefill through FlashInfer is intentionally disabled due multi-layer divergence.
