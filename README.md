# Funyi

Funyi is a local speech-to-text and live captions app built on
`Qwen/Qwen3-ASR-1.7B`. It runs on your machine, starts a local realtime ASR
WebSocket service, and provides a lightweight Tauri desktop caption client.

Funyi is designed for local, single-user use. It is not a hosted service or a
public multi-user ASR server.

## Demo

https://github.com/user-attachments/assets/cda710b8-5a05-4bd0-9e9f-5d2c9bc1de68

## Requirements

- Python 3.10 or newer; Python 3.12 is recommended.
- `uv` for Python dependencies.
- For the realtime backend: Linux or WSL with an NVIDIA CUDA GPU, or macOS on
  Apple Silicon with MLX/Metal. The backend default is `auto`, so macOS Apple
  Silicon selects the MLX ASR, forced-aligner, and translation paths.
- Access to download the ASR, forced-aligner, and translation models, or local
  model directories.
- For the desktop client: Node.js with Corepack-enabled `pnpm`, Rust/Cargo, and
  Windows or macOS native build tools.

Native desktop builds also need:

- Windows: Visual Studio Build Tools 2022 with the `Desktop development with
  C++` workload and a Windows 10/11 SDK.
- macOS: Xcode Command Line Tools (`xcode-select --install`).

The desktop client currently runs on Windows and macOS.

## Supported Models

Models are selected when the backend starts. CUDA uses the local transformers
backend; Apple Silicon uses MLX by default. The ids below are validated
defaults/examples; local model directories with the same architecture and
compatible config may also load, but should be gated locally before release use.

| Role | Validated model ids |
|---|---|
| ASR | `Qwen/Qwen3-ASR-1.7B` (default), `Qwen/Qwen3-ASR-0.6B`, `mlx-community/Qwen3-ASR-1.7B-4bit`, `mlx-community/Qwen3-ASR-0.6B-4bit` |
| Timestamps | `Qwen/Qwen3-ForcedAligner-0.6B` (required), `mlx-community/Qwen3-ForcedAligner-0.6B-4bit` |
| Translation | `tencent/Hy-MT2-1.8B` (default), `mlx-community/Hy-MT2-1.8B-4bit`; disable with `FUNYI_TRANSLATION_MODEL=` |

The `mlx-community/*-4bit` ids are Apple Silicon MLX paths. See
`docs/macos_mlx.md` for switching commands.

## Quick Start

### 1. Start The Backend

Run these commands from the repository root. On Linux/WSL the service uses the
CUDA path; on macOS Apple Silicon it uses the MLX/Metal path.

Install Python dependencies:

```bash
uv sync --python 3.12 --frozen
```

Start the backend in one terminal. On a fresh checkout or empty model cache, use
the download target once. The first start can take a while because it downloads
and warms the ASR, timestamp, and translation models:

```bash
FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh
```

After the models are cached, use:

```bash
./scripts/start_backend.sh
```

Check the backend:

```bash
curl http://127.0.0.1:8000/healthz
```

### 2. Start The Desktop Client

Run the desktop client from a Windows or macOS checkout. On Windows, use a
Windows checkout so system-audio capture uses native WASAPI.

Install desktop dependencies once:

```bash
cd desktop
corepack pnpm install
```

Start the desktop client:

```bash
corepack pnpm run dev
```

In the desktop UI, connect to:

```text
ws://127.0.0.1:8000/ws/asr
```

Choose your audio source, optionally choose a speech language and translation
target, then start captions.

On macOS, system audio capture may require Screen & System Audio Recording
permission. Microphone capture requires macOS 15+ and Microphone permission.
For file transcription, choose `File` as the audio source and start; it uses the
same backend and translation target.

## Windows Desktop

Run the backend in Linux or WSL. Start the desktop client from a Windows
checkout using the desktop commands above, then connect it to the backend URL,
usually `ws://127.0.0.1:8000/ws/asr`.

## Common Runs

| Command | What it does |
|---|---|
| `FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh` | Start the full backend and allow forced-aligner/translation model downloads. |
| `./scripts/start_backend.sh` | Start the full backend from cached or local models. |
| `FUNYI_TRANSLATION_MODEL= ./scripts/start_backend.sh` | Start ASR plus the required forced aligner, without translation. |
| `FUNYI_PORT=8001 ./scripts/start_backend.sh` | Start the backend on another port. |
| `./scripts/start_backend.sh --no-vad` | Start without VAD speech gating; all received audio is passed to ASR. |

To use local model directories instead of Hugging Face model ids:

```bash
FUNYI_ASR_MODEL=/path/to/Qwen3-ASR-1.7B \
FUNYI_TIMESTAMP_MODEL=/path/to/Qwen3-ForcedAligner-0.6B \
FUNYI_TRANSLATION_MODEL=/path/to/Hy-MT2-1.8B \
./scripts/start_backend.sh
```

Realtime ASR requires the forced aligner. Translation is available when the
backend starts with a translation model and the client requests a target
language.

## Privacy

Audio sent to `ws://127.0.0.1:8000/ws/asr` is processed by the local Python
service.

## Documentation

- `desktop/README.md`: desktop client and OS audio capture.
- `docs/macos_mlx.md`: Apple Silicon MLX models and switching.
- `docs/realtime_asr_service.md`: WebSocket and file transcription API.
- `docs/validation_and_regression.md`: local gates and private-data rules.

## License

Funyi's original project code is licensed under the MIT License. See `LICENSE`.

The files under `qwen3_asr_runtime/hf_qwen3_asr/` are copied from Qwen3-ASR /
Hugging Face Transformers integration code and retain their upstream
Apache-2.0 notices. See `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`.

Third-party dependencies, model weights, and external datasets remain under
their own license terms.
