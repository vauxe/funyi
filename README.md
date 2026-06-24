# Funyi

Local speech-to-text and live captions app built on `Qwen/Qwen3-ASR-1.7B`.
It runs a local realtime ASR WebSocket service and a Tauri desktop caption
client. Funyi is for local, single-user use.

## Demo

https://github.com/user-attachments/assets/cda710b8-5a05-4bd0-9e9f-5d2c9bc1de68

## Requirements

- Python 3.11+; Python 3.12 recommended.
- `uv`.
- Backend: Windows/Linux/WSL with NVIDIA CUDA, macOS 14+ on Apple Silicon, or
  opt-in CPU fallback with `FUNYI_ALLOW_CPU=1` (slow).
- Desktop: Node.js with Corepack `pnpm`, Rust/Cargo, and Windows/macOS native
  build tools.
- Model download access, or local model directories.

Native desktop builds also need Visual Studio Build Tools 2022 on Windows, or
Xcode Command Line Tools on macOS.

## Models

| Role | Default | Override |
|---|---|---|
| ASR | `Qwen/Qwen3-ASR-1.7B` | `FUNYI_ASR_MODEL` / `--model` |
| Timestamps | `Qwen/Qwen3-ForcedAligner-0.6B` | `FUNYI_TIMESTAMP_MODEL` / `--timestamp-model` |
| Translation | `tencent/Hy-MT2-1.8B` | `FUNYI_TRANSLATION_MODEL` / `--translation-model` |

Set `FUNYI_TRANSLATION_MODEL=` to disable translation. Restart the backend after
changing model variables. Apple Silicon MLX model ids are documented in
`docs/macos_mlx.md`.

## Quick Start

From the repository root:

```bash
uv sync --python 3.12 --frozen
```

Start the backend. Use the download variable on first run or when model caches
are empty.

Linux, WSL, or macOS:

```bash
FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh
./scripts/start_backend.sh
```

Windows PowerShell:

```powershell
$env:FUNYI_ALLOW_DOWNLOADS = "1"
.\scripts\start_backend.ps1
.\scripts\start_backend.ps1
```

Check the backend:

```bash
curl http://127.0.0.1:8000/healthz
```

Start the desktop client:

```bash
cd desktop
corepack pnpm install
corepack pnpm run dev
```

Connect to:

```text
ws://127.0.0.1:8000/ws/asr
```

## Common Runs

| Command | What it does |
|---|---|
| `FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh` | Start full backend and allow model downloads. |
| `./scripts/start_backend.sh` | Start full backend from cached/local models. |
| `FUNYI_TRANSLATION_MODEL= ./scripts/start_backend.sh` | Start ASR plus timestamps, without translation. |
| `FUNYI_PORT=8001 ./scripts/start_backend.sh` | Start on another port. |
| `./scripts/start_backend.sh --no-vad` | Disable VAD speech gating. |
| `FUNYI_ALLOW_CPU=1 ./scripts/start_backend.sh` | CPU fallback; slow, not realtime. |

In Windows PowerShell, use `.\scripts\start_backend.ps1` and set variables with
`$env:NAME = "value"` before running the script.

Windows `start_backend.ps1` sets `TORCHDYNAMO_DISABLE=1` and
`TORCH_COMPILE_DISABLE=1` by default when unset, so the default HY-MT translation
path does not require a separate TorchInductor/Triton setup. No
`TORCH_COMPILE` variable is needed.

## Documentation

- `desktop/README.md`: desktop client.
- `docs/realtime_asr_service.md`: WebSocket and file transcription API.
- `docs/macos_mlx.md`: Apple Silicon MLX backend.
- `docs/cpu_backend.md`: CPU fallback.
- `docs/validation_and_regression.md`: local quality gates.

## License

Project code is MIT. Some vendored files retain Apache-2.0 notices. See
`LICENSE`, `LICENSES/Apache-2.0.txt`, and `THIRD_PARTY_NOTICES.md`.
