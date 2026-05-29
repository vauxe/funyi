# Agent Notes

Startup checklist only. Keep this file short.

## Goal

Build a runtime Qwen3-ASR around the upstream `transformers` backend without
depending on upstream package code at runtime. Preserve upstream-compatible
library/offline defaults. The local service entrypoint may enable the validated
single-user profile by default.

## Invariants

- Runtime scope is transcription only.
- Public releases must not include private audio, transcripts, or audio-derived
  goldens.
- Keep validation audio in `local_data/` and generated outputs in
  `local_goldens/`; both are ignored by git.
- Default offline path must remain upstream-compatible and byte-regressed before
  release.
- Optimized paths are quality-gated by punctuation-stripped CER vs allowed SRT.
- Streaming default remains full-audio re-feed unless `max_window_sec` is set.

## Current Baseline

- Offline all-fused: `cuda_graph + flashinfer + fused_rmsnorm + fused_linears`.
- Offline W8A16 is opt-in on top of all-fused, not default.
- Known-language prompts are opt-in; auto language stays default.
- Generic bounded-live prefix-mode start point is live30; stricter quality mode
  is live45.
- Local service default is the live20 model streaming profile with `cuda_graph +
  flashinfer + fused_rmsnorm + fused_linears + W8A16` plus required forced-aligner
  timestamp patches.
- W8A16 means qkv/gate_up only.

## Important Paths

- `qwen3_asr_runtime/model.py`: public offline/streaming wrapper
- `qwen3_asr_runtime/backends/transformers.py`: backend and opt-in flags
- `qwen3_asr_runtime/hf_qwen3_asr/`: local HF model/config/processor
- `qwen3_asr_runtime/decode_runtime.py`: CUDA graph decode loop
- `qwen3_asr_runtime/spec_decode.py`: streaming speculative verification
- `qwen3_asr_runtime/flashinfer_attention.py`: FlashInfer decode attention
- `qwen3_asr_runtime/quant_linears.py`: W8A16 linears
- `qwen3_asr_runtime/fused_rmsnorm.py`, `fused_linears.py`: fused opt-ins
- `realtime_server.py`: local WebSocket ASR service
- `qwen3_asr_runtime/realtime_session.py`, `vad.py`, `transcript_store.py`:
  realtime session, VAD, and in-memory source transcript
- `qwen3_asr_runtime/realtime_timestamps.py`: forced-aligner timestamp runtime
- `tools/run_regression.py`, `tools/sweep_cer_vs_srt.py`,
  `tools/merge_cer_sweeps.py`: offline checks
- `tools/run_streaming_regression.py`, `tools/benchmark_streaming.py`,
  `tools/sweep_streaming_cer_vs_srt.py`: streaming checks
- `tools/ws_e2e_leak_check.py`: service E2E and resource check

## Read On Demand

- `@docs/validation_and_regression.md`: exact commands and gates
- `@docs/streaming_runtime.md`: streaming semantics and live presets
- `@docs/realtime_asr_service.md`: WebSocket protocol and session rules
- `@docs/performance_optimization.md`: optimized stack and rejected paths
- `@docs/qwen3_asr_1_7b_architecture.md`: model-shape reference

## Environment

- Dependency manager: `uv`, project mode non-package.
- Default deps include service stack, Silero VAD, Torchaudio, FlashInfer, Ninja.
- Desktop quality gate: `make desktop-check`; format with `make desktop-format`.

## Caveats

- Do not use `run_regression.py` for optimized flags; use CER gates.
- Do not treat `prefill_chunk` as a correctness path.
- FlashInfer prefill is intentionally disabled due multi-layer divergence.
