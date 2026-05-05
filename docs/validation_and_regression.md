# Validation And Regression

Use this when changing correctness-sensitive code, tool metrics, or regression
generation. Public releases do not include audio-derived goldens. Keep local
validation assets under `local_data/` and generated outputs under
`local_goldens/`; both directories are ignored by git.

The examples below use:

```bash
ASR_AUDIO=local_data/sample.wav
ASR_SRT=local_data/sample.srt
```

## Public Smoke

These checks do not require private audio or generated goldens:

```bash
uv run python -m compileall -q qwen3_asr_runtime realtime_server.py tools tests
uv run python -m unittest tests.test_streaming_spec_decode tests.test_realtime_asr
git diff --check
```

## Generate Local Goldens

Generate exact-regression goldens from audio you are allowed to use:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/generate_offline_regression_golden.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --audio "$ASR_AUDIO" \
  --reference-srt "$ASR_SRT" \
  --output local_goldens/offline_regression.json

TRANSFORMERS_VERBOSITY=error uv run python tools/generate_streaming_regression_golden.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --audio "$ASR_AUDIO" \
  --reference-srt "$ASR_SRT" \
  --output local_goldens/streaming_regression.json
```

## Offline Exact Regression

The default runtime transformers path should stay byte-identical to a local
runtime-default golden on the same stack.

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/run_regression.py \
  --golden local_goldens/offline_regression.json

TRANSFORMERS_VERBOSITY=error uv run python tools/run_regression.py \
  --golden local_goldens/offline_regression.json \
  --cases short_default_15s,short_context_15s,short_forced_language_15s
```

Do not pass optimization flags here. CUDA graph / FlashInfer / fused kernels can
drift byte output; validate optimized paths with CER sweeps.

## Streaming Exact Regression

The streaming golden locks the default full-audio re-feed state machine: chunk
buffering, rollback prefix, snapshots, and final text.

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/run_streaming_regression.py \
  --golden local_goldens/streaming_regression.json

TRANSFORMERS_VERBOSITY=error uv run python tools/run_streaming_regression.py \
  --golden local_goldens/streaming_regression.json \
  --cases short_default_15s --step-ms 2000

TRANSFORMERS_VERBOSITY=error uv run python tools/benchmark_streaming.py \
  --golden local_goldens/streaming_regression.json \
  --cases short_default_15s --step-ms 2000 --repeats 15 --check-final
```

## Realtime Service E2E

Use this when changing `realtime_server.py`, session/VAD behavior, WebSocket
contracts, or service dependencies. This is a service-level gate, separate from
model regression goldens.

Start the service:

```bash
uv run python realtime_server.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000
```

Then run a bounded real-audio WebSocket check:

```bash
uv run python tools/ws_e2e_leak_check.py \
  --url ws://127.0.0.1:8000/ws/asr \
  --audio "$ASR_AUDIO" \
  --reference-srt "$ASR_SRT" \
  --start-sec 0 \
  --max-audio-sec 600 \
  --language Chinese \
  --finish-timeout-sec 300 \
  --max-wall-sec 900 \
  --output-json /tmp/realtime-e2e-0000.json
```

For a longer local gate, repeat with additional `--start-sec` values that are
valid for your validation audio.

## CER Sweeps

Generate local CER sweeps against an SRT reference:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths base --window-sec 60 --num-windows 200 \
  --output local_goldens/cer/cer_base.json

TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths graph --window-sec 60 --num-windows 200 \
  --flashinfer --fused-rmsnorm --fused-linears \
  --output local_goldens/cer/cer_candidate.json

uv run python tools/merge_cer_sweeps.py \
  --input local_goldens/cer/cer_base.json=base \
          local_goldens/cer/cer_candidate.json=candidate \
  --output local_goldens/cer/cer_compare.json
```

For streaming CER:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_streaming_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths graph --window-sec 60 --num-windows 167 --step-ms 1000 \
  --chunk-size-sec 2.0 --max-window-sec 30 --timed --spec-decode \
  --flashinfer --fused-rmsnorm --fused-linears \
  --output local_goldens/cer/streaming_cer_candidate.json
```

Use `--strip-ruby` for SRT files that contain Japanese-style furigana
annotations.
