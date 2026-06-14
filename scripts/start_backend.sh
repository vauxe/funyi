#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

asr_model="${FUNYI_ASR_MODEL-Qwen/Qwen3-ASR-1.7B}"
host="${FUNYI_HOST-127.0.0.1}"
port="${FUNYI_PORT-8000}"
translation_model="${FUNYI_TRANSLATION_MODEL-tencent/Hy-MT2-1.8B}"
timestamp_model="${FUNYI_TIMESTAMP_MODEL-Qwen/Qwen3-ForcedAligner-0.6B}"
allow_downloads="${FUNYI_ALLOW_DOWNLOADS-0}"
allow_cpu="${FUNYI_ALLOW_CPU-0}"
firered_vad_model_dir="${FUNYI_FIRERED_VAD_MODEL_DIR-local_data/models/firered-stream-vad-onnx}"

args=(
  python realtime_server.py
  --model "$asr_model"
  --host "$host"
  --port "$port"
  --firered-vad-model-dir "$firered_vad_model_dir"
)

if [[ -n "$translation_model" ]]; then
  args+=(--translation-model "$translation_model")
fi

if [[ -z "$timestamp_model" ]]; then
  echo "FUNYI_TIMESTAMP_MODEL is required for realtime ASR." >&2
  exit 64
fi
args+=(--timestamp-model "$timestamp_model")

case "$allow_downloads" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ -n "$translation_model" ]]; then
      args+=(--no-translation-local-files-only)
    fi
    args+=(--no-timestamp-local-files-only)
    ;;
esac

case "$allow_cpu" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--allow-cpu)
    if [[ -n "$translation_model" ]]; then
      echo "FUNYI_ALLOW_CPU set: CPU mode is slow and not realtime; HY-MT (1.8B) is heavy on CPU. Consider FUNYI_TRANSLATION_MODEL= and a smaller ASR model such as Qwen/Qwen3-ASR-0.6B." >&2
    fi
    ;;
esac

exec uv run --frozen "${args[@]}" "$@"
