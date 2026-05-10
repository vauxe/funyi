# Validation And Regression

Use this when changing correctness-sensitive runtime code, tools, service
behavior, or documented metrics. Public releases do not include private audio,
transcripts, or audio-derived outputs.

Local assets:

```bash
ASR_AUDIO=local_data/sample.wav
ASR_SRT=local_data/sample.srt
```

Keep validation audio in `local_data/` and generated outputs in
`local_goldens/`; both are ignored by git.

## Public Smoke

No private audio required:

```bash
uv run python -m compileall -q qwen3_asr_runtime realtime_server.py tools tests
uv run python -m unittest discover tests
git diff --check
```

## Exact Regression

Generate local goldens from audio you are allowed to use:

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

Run default-path checks:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/run_regression.py \
  --golden local_goldens/offline_regression.json

TRANSFORMERS_VERBOSITY=error uv run python tools/run_streaming_regression.py \
  --golden local_goldens/streaming_regression.json

TRANSFORMERS_VERBOSITY=error uv run python tools/benchmark_streaming.py \
  --golden local_goldens/streaming_regression.json \
  --cases short_default_15s --step-ms 2000 --repeats 15 --check-final
```

Do not pass optimization flags to exact regression. CUDA graph, FlashInfer,
fused kernels, W8A16, and spec decode are CER-gated instead.

## CER Gates

Offline all-fused candidate:

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

Generic live30 streaming candidate:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_streaming_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths graph --window-sec 60 --num-windows 167 --step-ms 1000 \
  --chunk-size-sec 2.0 --max-window-sec 30 --timed --spec-decode \
  --flashinfer --fused-rmsnorm --fused-linears \
  --output local_goldens/cer/streaming_cer_candidate.json
```

Local realtime service profile:

```bash
TRANSFORMERS_VERBOSITY=error uv run python tools/sweep_streaming_cer_vs_srt.py \
  --audio "$ASR_AUDIO" --srt "$ASR_SRT" \
  --paths graph --window-sec 60 --num-windows 167 --step-ms 500 \
  --chunk-size-sec 0.5 --unfixed-chunk-num 4 \
  --max-window-sec 20 --max-prefix-tokens 64 \
  --timed --spec-decode --cuda-graph-len-bucket 64 \
  --flashinfer --fused-rmsnorm --fused-linears --quantized-linears \
  --output local_goldens/cer/streaming_cer_service_profile.json
```

Use `--strip-ruby` for SRT files with furigana annotations.

## WebSocket E2E

Use this when changing `realtime_server.py`, realtime session behavior,
WebSocket contracts, or service dependencies.

```bash
uv run python realtime_server.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000

uv run python tools/ws_e2e_leak_check.py \
  --url ws://127.0.0.1:8000/ws/asr \
  --audio "$ASR_AUDIO" \
  --reference-srt "$ASR_SRT" \
  --start-sec 0 \
  --max-audio-sec 600 \
  --chunk-sec 0.1 \
  --send-delay-sec 0.1 \
  --language Chinese \
  --finish-timeout-sec 300 \
  --max-wall-sec 900 \
  --output-json /tmp/realtime-e2e-0000.json
```

For a longer gate, repeat with additional valid `--start-sec` offsets. Keep
service timing, RSS, and whole-GPU telemetry out of committed goldens.
