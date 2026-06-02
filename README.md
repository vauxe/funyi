# Funyi

Funyi is a local speech-to-text and live captions app built on
`Qwen/Qwen3-ASR-1.7B`. It runs on your machine, starts a local realtime ASR
WebSocket service, and provides a lightweight Tauri desktop caption client.

Funyi is designed for local, single-user use. It is not a hosted service or a
public multi-user ASR server.

## Requirements

- Python 3.10 or newer; Python 3.12 is recommended.
- `uv` for Python dependencies.
- Linux or WSL with an NVIDIA CUDA GPU for the realtime backend.
- Access to download the ASR, forced-aligner, and translation models, or local
  model directories.
- For the desktop client: Node.js with Corepack-enabled `pnpm`, Rust/Cargo, and
  Windows or macOS native build tools.

Windows desktop builds also need Visual Studio Build Tools 2022 with the
`Desktop development with C++` workload and a Windows 10/11 SDK.

The desktop client currently runs on Windows and macOS.

## Quick Start

### 1. Start The Backend

Run these commands from the repository root in Linux or WSL.

Install Python dependencies:

```bash
uv sync --python 3.12 --frozen
```

Start the backend in one terminal. On a fresh checkout or empty model cache, use
the download target once. The first start can take a while because it downloads
and warms the models:

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
| `corepack pnpm install` | Install desktop dependencies from `desktop/`. |
| `corepack pnpm run dev` | Start the desktop client from `desktop/`. |

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

## Troubleshooting

- If downloads fail, run
  `FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh` once or pass local model
  paths.
- If the backend fails to start, check CUDA availability and model paths.
- If the desktop cannot connect, confirm the URL is `ws://127.0.0.1:8000/ws/asr`
  and `curl http://127.0.0.1:8000/healthz` returns `{"status":"ok"}`.
- If `pnpm` is not found, run desktop commands as `corepack pnpm ...`.
- On Windows, run the desktop client from a Windows checkout when validating
  WASAPI loopback. WSL is not the right place to validate Windows system-audio
  capture.

## Documentation

- `desktop/README.md`: desktop client details and OS audio-capture notes.
- `docs/realtime_asr_service.md`: WebSocket protocol, timestamp behavior, and
  realtime service rules.
- `docs/realtime_translation_design.md`: translation behavior and target-language
  details.

## License

Funyi's original project code is licensed under the MIT License. See `LICENSE`.

The files under `qwen3_asr_runtime/hf_qwen3_asr/` are copied from Qwen3-ASR /
Hugging Face Transformers integration code and retain their upstream
Apache-2.0 notices. See `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`.

Third-party dependencies, model weights, and external datasets remain under
their own license terms.
