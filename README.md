# Funyi

Funyi is a local speech-to-text and live captions app built on
`Qwen/Qwen3-ASR-1.7B`.

It can:

- transcribe audio files from Python;
- run a local realtime ASR WebSocket service;
- show live captions in a lightweight Tauri desktop client;
- capture system playback audio on supported platforms;
- optionally add subtitle translation and forced-aligner timestamps.

Funyi is designed for local, single-user use. It is not a hosted service and it
is not a public multi-user ASR server.

## Requirements

- Python 3.10 or newer. Python 3.12 is recommended.
- `uv` for Python dependencies.
- An NVIDIA CUDA GPU for the default realtime profile.
- A local model path or access to download `Qwen/Qwen3-ASR-1.7B`.
- For the desktop client: Node.js with Corepack-enabled `pnpm`, Rust/Cargo,
  and macOS or Windows native build tools.
- On Windows, Tauri also needs Visual Studio Build Tools 2022 with the
  `Desktop development with C++` workload so `cl.exe` and `link.exe` are
  available, plus a Windows 10/11 SDK.

The desktop client currently runs on Windows and macOS. A Linux audio capture
module exists, but Linux desktop builds are disabled until the overlay window
layer has Linux support.

| Platform | System audio capture |
|---|---|
| Windows | WASAPI loopback from the default playback device |
| macOS | ScreenCaptureKit system audio capture, with Screen & System Audio Recording permission |

## Install

Install the Python runtime:

```bash
uv sync --python 3.12
```

Install the desktop dependencies only if you want to run or build the Tauri
client:

```bash
make desktop-install
```

If your shell does not expose a `pnpm` command, use `corepack pnpm ...`
directly instead of trying to install a separate global shim.

## Start Live Captions

Start the full local backend first:

```bash
make backend
```

This starts ASR with translation and forced-aligner timestamps enabled, using
the validated local service optimization stack. Common variants:

```bash
make backend-download
make backend-asr
FUNYI_PORT=8001 make backend
make backend BACKEND_ARGS="--live-stability-delay-ms 8000"
```

The realtime service defaults to `--live-stability-delay-ms 12000` so stable
history stays conservative. Use the replaceable `partial` line for low-latency
live subtitle display; stable text is split into subtitle-sized cues after it is
safe to commit.

Check that the service is alive:

```bash
curl http://127.0.0.1:8000/healthz
```

Then start the desktop client:

```bash
make desktop
```

Use the desktop UI to connect to:

```text
ws://127.0.0.1:8000/ws/asr
```

On Windows, run the desktop client from a Windows checkout when you want to
validate WASAPI loopback. WSL can compile the Linux Tauri build, but it is not
the right place to validate a native Windows window or Windows system-audio
capture.

## Enable Translation

`make backend` enables `tencent/HY-MT1.5-1.8B` by default. To use a different
model or local path:

```bash
FUNYI_TRANSLATION_MODEL=/path/to/HY-MT1.5-1.8B make backend
```

Then choose a target language in the desktop UI. Set
`FUNYI_TRANSLATION_MODEL=` to disable translation and leave the desktop target
language set to `Off`. Auxiliary models load from local files by default; use
`make backend-download` only when a download is expected.

## Enable Forced-Aligner Timestamps

Stable transcript segments already include sample-clock timing.
`make backend` enables forced-aligned timestamps with
`Qwen/Qwen3-ForcedAligner-0.6B` by default. To use a different model or local
path:

```bash
FUNYI_TIMESTAMP_MODEL=/path/to/Qwen3-ForcedAligner-0.6B make backend
```

Set `FUNYI_TIMESTAMP_MODEL=` to disable forced alignment. Auxiliary models load
from local files by default; use `make backend-download` only when a download is
expected.

## Transcribe An Audio File

Use the Python API for offline transcription:

```python
from qwen3_asr_runtime import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype="bfloat16",
    device_map="cuda:0",
).eval()

result = model.transcribe("local_data/sample.wav")[0]
print(result.language)
print(result.text)
```

To force a known language:

```python
result = model.transcribe("local_data/sample.wav", language="Chinese")[0]
```

If you pass a Hugging Face model id, the model loads through Transformers and
may download weights into the Hugging Face cache. Pass a local path when you
need fully offline startup.

Python API boundary:

- `Qwen3ASRModel.from_pretrained(path_or_id, backend="transformers", **kwargs)`
  is the supported loader. Other backend names are rejected.
- `transcribe(audio, context="", language=None)` accepts one audio input or a
  list of inputs and returns a list of `ASRTranscription(language, text,
  time_stamps=None)`.
- `return_time_stamps=True` is not supported by `transcribe`; use the realtime
  service with `--timestamp-model` or `Qwen3ForcedAlignerBackend` directly when
  forced-aligned timestamps are required.
- Streaming callers use `init_streaming_state(...)`,
  `streaming_transcribe(pcm16k, state)`, and
  `finish_streaming_transcribe(state)`. Library streaming defaults stay
  upstream-compatible; the local service applies the live20 low-latency preset.

## Build The Desktop App

After installing desktop dependencies:

```bash
cd desktop
corepack pnpm run build
```

Linux desktop bundles are not supported until the overlay window layer has Linux
support.

## Check Desktop Changes

From the repository root:

```bash
make desktop-check
make desktop-format
```

`desktop-check` runs the desktop lint, format, TypeScript, UI test, and Rust
test gates.

## Data And Privacy

The ASR service runs locally on your machine. Audio sent to
`ws://127.0.0.1:8000/ws/asr` is processed by the local Python service.

Keep private validation audio in `local_data/` and generated outputs in
`local_goldens/`. Both directories are ignored by git. Do not publish private
audio, transcripts, or audio-derived goldens.

## Troubleshooting

If the service fails during startup, first check:

- the model path or Hugging Face model id is correct;
- the CUDA device is available;
- optional timestamp or translation models are present locally when
  local-files-only mode is enabled;
- the desktop client is using `ws://127.0.0.1:8000/ws/asr`;
- if the shell says `pnpm` is not recognized, run desktop commands as
  `corepack pnpm ...`;
- on Windows, `where cl` and `where link` both succeed after installing
  Visual Studio Build Tools 2022 with `Desktop development with C++` and a
  Windows SDK.

If you only need to rebuild dependencies after a cleanup:

```bash
uv sync --python 3.12
cd desktop
corepack pnpm install
```

## More Documentation

- `desktop/README.md`: desktop client details and OS audio-capture notes.
- `docs/realtime_asr_service.md`: WebSocket protocol and service behavior.
- `docs/realtime_translation_design.md`: translation scheduling, finish
  semantics, and quality gates.
- `docs/streaming_runtime.md`: streaming runtime semantics.
- `docs/validation_and_regression.md`: developer validation commands.
