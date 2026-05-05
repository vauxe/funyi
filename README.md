# Qwen3-ASR Runtime

A runtime transcription wrapper around the Qwen/Qwen3-ASR-1.7B transformers
checkpoint, plus a local realtime WebSocket ASR service. The library/offline path
keeps upstream-compatible defaults; the service entrypoint defaults to the
validated single-user optimized profile.

## What is included

- `qwen3_asr_runtime/`: standalone runtime package and optimized transformers backend
- `realtime_server.py`: single-user WebSocket ASR service
- `tools/`: local validation, benchmark, and regression commands
- `docs/`: runtime notes and validation workflow

Public releases do not include audio, transcripts, or generated goldens. Keep
local validation assets under `local_data/` and generated outputs under
`local_goldens/`; both directories are ignored by git.

## Install

```bash
uv sync --python 3.12
```

## Realtime service

After cloning the repository, start the local realtime ASR service with the GPU
`transformers` backend:

```bash
uv run python realtime_server.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000
```

The service defaults to the validated low-latency single-user GPU profile:
`cuda_graph`, `flashinfer`, `fused_rmsnorm`, `fused_linears`, and the
`w8a16` linears are enabled. Realtime sessions use the low-latency
streaming preset. Use `--no-cuda-graph`, `--no-flashinfer`,
`--no-fused-rmsnorm`, `--no-fused-linears`, or `--no-w8a16` only for debugging,
environment fallback, or quality comparison.

## Python API

Use the library API for direct offline transcription:

```python
from qwen3_asr_runtime import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype="bfloat16",
    device_map="cuda:0",
).eval()
result = model.transcribe(audio="path/to/audio.wav")[0]
print(result.language, result.text)
```

For streaming API details and bounded-live examples, see
`docs/streaming_runtime.md`.

## Validation

```bash
uv run python -m compileall -q qwen3_asr_runtime realtime_server.py tools tests
uv run python -m unittest tests.test_streaming_spec_decode tests.test_realtime_asr
git diff --check
```

For local audio regression, CER sweeps, and WebSocket E2E checks, see
`docs/validation_and_regression.md`.

## Docs

- `docs/realtime_asr_service.md`: WebSocket service protocol and boundaries
- `docs/streaming_runtime.md`: streaming state model and live-caption presets
- `docs/validation_and_regression.md`: local validation workflow
- `docs/performance_optimization.md`: optimized decode stack and gates
- `docs/qwen3_asr_1_7b_architecture.md`: model architecture reference
