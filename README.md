# Qwen3-ASR Runtime

Standalone runtime around `Qwen/Qwen3-ASR-1.7B`, plus a local single-user
WebSocket ASR service.

Library/offline defaults stay upstream-compatible. The service defaults to the
validated low-latency single-user GPU profile.

## Layout

- `qwen3_asr_runtime/`: runtime package and optimized transformers backend
- `realtime_server.py`: local WebSocket ASR service
- `desktop/`: lightweight Tauri desktop client for system-audio captions
- `tools/`: validation, benchmark, and regression commands
- `docs/`: design notes and validation workflow

Do not publish private audio, transcripts, or generated goldens. Keep local
validation audio in `local_data/` and generated outputs in `local_goldens/`.

## Install

```bash
uv sync --python 3.12
```

## Realtime Service

```bash
uv run python realtime_server.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000
```

Service defaults: live20, `spec_decode`, `cuda_graph`, FlashInfer,
`fused_rmsnorm`, `fused_linears`, and W8A16 qkv/gate_up. Use `--no-*` flags only
for debugging, fallback, or comparison.

## Python API

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

## Smoke

```bash
uv run python -m compileall -q qwen3_asr_runtime realtime_server.py tools tests
uv run python -m unittest tests.test_streaming_spec_decode tests.test_realtime_asr
git diff --check
```

## Docs

- `docs/validation_and_regression.md`: commands and gates
- `docs/realtime_asr_service.md`: WebSocket protocol and state rules
- `docs/streaming_runtime.md`: streaming defaults and live presets
- `docs/performance_optimization.md`: optimization stack and dead ends
- `docs/qwen3_asr_1_7b_architecture.md`: model-shape facts
- `docs/realtime_translation_design.md`: optional translation pipeline and client replay model
