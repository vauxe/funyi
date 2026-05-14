# Funyi Desktop

Lightweight Tauri client for the local realtime ASR service.

The UI owns the `/ws/asr` WebSocket. The native layer only captures system audio
and emits `pcm_s16le` frames at 16 kHz. This keeps ASR, translation, and CUDA
runtime behavior in the Python service.

## Status

- Windows: default system output capture via WASAPI loopback.
- Linux: PipeWire/PulseAudio monitor source capture through `pactl` + `parec`.
- macOS: interface is present, but system audio capture is intentionally disabled
  until a ScreenCaptureKit or virtual-device adapter is implemented with the
  required user approval and release entitlements.

## Run

Start the ASR service first:

```bash
uv run python realtime_server.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000 \
  --translation-target-language English
```

Then run the desktop client from this directory:

```bash
pnpm install
pnpm run dev
```

For Windows plus WSL development, run the Tauri client in the Windows checkout
and point it at the WSL service URL, for example `ws://127.0.0.1:8000/ws/asr`
when the backend is exposed on localhost.

Linux/WSL Tauri builds need native WebView/DBus development packages. On Ubuntu
the missing-package class usually starts with:

```bash
sudo apt install pkg-config libdbus-1-dev libwebkit2gtk-4.1-dev
```

Linux system-audio capture uses monitor sources, not microphones. With PipeWire
or PulseAudio, install the PulseAudio CLI tools if needed and select a source
whose name ends with `.monitor`:

```bash
sudo apt install pulseaudio-utils
pactl list short sources
```

The Windows client path is the important path for system-audio capture because
WASAPI loopback captures the default Windows playback device.

## Contract

Native audio events:

```json
{
  "seq": 0,
  "sampleRate": 16000,
  "format": "pcm_s16le",
  "dataBase64": "..."
}
```

The frontend decodes each frame and forwards it unchanged to `/ws/asr`.
Backpressure is handled by dropping frames when the WebSocket buffer is too far
behind rather than growing memory without bound.
