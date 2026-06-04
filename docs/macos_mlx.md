# macOS MLX Backend

The MLX backend is the Apple Silicon macOS path. It runs the ASR model, the
forced aligner, and HY-MT translation on Metal through local model layers in
this repository. The desktop client does not choose models; it only connects to
the local WebSocket service.

On Apple Silicon, `--backend auto` resolves to MLX when `mlx` is installed. Use
`--backend mlx` to force the MLX ASR backend, or `--backend transformers` to
force the CUDA/torch path on a machine where that makes sense.

## Supported Models

Supported here means the repository has a dedicated MLX model layer for the
checkpoint family and the service can route it through the macOS backend. Public
release quality is still gated locally by the same goldens as the CUDA path.

| Role | Supported model ids | Switch with | Notes |
|---|---|---|---|
| ASR | `Qwen/Qwen3-ASR-0.6B` | `FUNYI_ASR_MODEL` or `--model` | Official safetensors checkpoint. |
| ASR | `Qwen/Qwen3-ASR-1.7B` | `FUNYI_ASR_MODEL` or `--model` | Official sharded safetensors checkpoint. This is the default ASR model. |
| ASR | `mlx-community/Qwen3-ASR-0.6B-4bit` | `FUNYI_ASR_MODEL` or `--model` | Lower-memory MLX 4-bit ASR checkpoint. |
| ASR | `mlx-community/Qwen3-ASR-1.7B-4bit` | `FUNYI_ASR_MODEL` or `--model` | Lower-memory MLX 4-bit ASR checkpoint with the 1.7B ASR architecture. |
| Forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B` | `FUNYI_TIMESTAMP_MODEL` or `--timestamp-model` | Required for realtime ASR timestamps. |
| Forced aligner | `mlx-community/Qwen3-ForcedAligner-0.6B-4bit` | `FUNYI_TIMESTAMP_MODEL` or `--timestamp-model` | Lower-memory MLX 4-bit timestamp model. Use only as the timestamp model. |
| Translation | `tencent/Hy-MT2-1.8B` | `FUNYI_TRANSLATION_MODEL` or `--translation-model` | Official HY-MT checkpoint. This is the default translation model. |
| Translation | `mlx-community/Hy-MT2-1.8B-4bit` | `FUNYI_TRANSLATION_MODEL` or `--translation-model` | Lower-memory MLX 4-bit HY-MT checkpoint. |

The Qwen ASR checkpoints share `qwen3_asr_runtime/mlx_qwen3_asr/` plus
`qwen3_asr_runtime/mlx_common/`. The forced aligner reuses that Qwen ASR model
layer but has a timestamp classifier head, so it is not an ASR model. HY-MT
translation uses `qwen3_asr_runtime/mlx_hunyuan/`; it shares only the common MLX
primitives.

Other MLX conversions with the same config shape are compatible candidates and
may load through the same environment variables. Treat them as validated support
only after they pass the local golden gates.

## Switching Models

Model switching is a backend restart. The desktop client reconnects to the same
WebSocket URL after the backend is restarted; there is no UI hot-swap for model
ids.

Switch models by setting the role-specific environment variables before
starting the service. For a lower-memory Apple Silicon run:

```bash
FUNYI_ASR_MODEL=mlx-community/Qwen3-ASR-0.6B-4bit \
FUNYI_TIMESTAMP_MODEL=mlx-community/Qwen3-ForcedAligner-0.6B-4bit \
FUNYI_TRANSLATION_MODEL=mlx-community/Hy-MT2-1.8B-4bit \
FUNYI_ALLOW_DOWNLOADS=1 \
./scripts/start_backend.sh
```

Use `mlx-community/Qwen3-ASR-1.7B-4bit` as `FUNYI_ASR_MODEL` for the 1.7B
4-bit ASR variant. To keep the default models, run:

```bash
FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh
```

Disable translation:

```bash
FUNYI_TRANSLATION_MODEL= ./scripts/start_backend.sh
```

`auto` chooses MLX on Apple Silicon. To force it explicitly:

```bash
./scripts/start_backend.sh \
  --backend mlx \
  --timestamp-backend mlx \
  --translation-backend mlx
```

Use local model directories instead of Hugging Face ids:

```bash
FUNYI_ASR_MODEL=/path/to/Qwen3-ASR-1.7B \
FUNYI_TIMESTAMP_MODEL=/path/to/Qwen3-ForcedAligner-0.6B \
FUNYI_TRANSLATION_MODEL=/path/to/Hy-MT2-1.8B \
./scripts/start_backend.sh
```

`FUNYI_ALLOW_DOWNLOADS=1` is only needed for a fresh cache or when a selected
Hugging Face model id is not already cached locally.
