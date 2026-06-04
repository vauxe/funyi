# Funyi Desktop

Lightweight Tauri client for the local ASR service.

The UI owns the `/ws/asr` WebSocket for live captions and the
`/api/transcriptions` HTTP endpoint for local file transcription. The native
layer only captures live audio — system output or microphone input — and emits
`pcm_s16le` frames at 16 kHz. This keeps ASR, translation, and CUDA runtime
behavior in the Python service.

The default window is a compact, always-on-top caption strip near the bottom of
the display. Detailed connection settings stay inline, and stable subtitle
history appears automatically when the window is tall enough.

A settings popover (the kebab button in the caption controls) exposes the caption
panel opacity, an optional background image, and a one-click "Copy transcript" to
the clipboard. The server URL, speech/translation languages, audio source, panel
opacity, and background image are remembered across launches (stored in the
webview's `localStorage`); the background image is downscaled before it is saved.

## Status

- Windows: default system output capture via WASAPI loopback.
- Windows: microphone input capture via active WASAPI recording devices.
- macOS: system audio capture through ScreenCaptureKit. The first capture start
  may require Screen & System Audio Recording permission in System Settings.
- macOS: microphone input capture through ScreenCaptureKit on macOS 15+. The
  first microphone start may require Microphone permission in System Settings.

## Run

From the repository root, start the ASR service first:

```bash
FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh
```

After the models are cached, use `./scripts/start_backend.sh`.

The backend can run on the same Apple Silicon macOS machine with MLX/Metal, or
on Linux/WSL with CUDA. On Windows, run this desktop client from a Windows
checkout and connect it to the WSL backend URL, usually
`ws://127.0.0.1:8000/ws/asr`.

On macOS, model switching happens by restarting the backend with different
environment variables. The desktop client still connects to the same URL after
the backend restarts. See `../docs/macos_mlx.md` for the supported MLX model
list and switching commands.

Then run the desktop client:

```powershell
cd desktop
corepack pnpm install
corepack pnpm run dev
```

If your shell does not expose a `pnpm` command, keep using `corepack pnpm ...`
rather than assuming a separate global `pnpm` install exists.

The client only accepts `ws://` loopback service URLs such as
`ws://127.0.0.1:8000/ws/asr`. File transcription derives the matching
`http://` loopback API URL from that value. Remote hosts, `wss://`, credentialed
URLs, and non-loopback HTTP are rejected.

Native Windows Tauri builds also require Visual Studio Build Tools 2022 with
the `Desktop development with C++` workload and a Windows 10/11 SDK. `cargo`
alone is not enough; `cl.exe` and `link.exe` must be available in the build
environment.

The Windows client path is the important path for system-audio capture because
WASAPI loopback captures the default Windows playback device. Windows
microphones are listed from active recording devices and captured through
shared-mode WASAPI input.

## Accessibility

Captions are exposed to screen readers through a dedicated polite live region
that announces only *stabilized* lines (never the per-partial current line, to
avoid flooding the reader); a `transcript_final` rebuild never re-announces, and
the log is capped. Caption source and translation are split into spans with
their own BCP-47 `lang` and `dir="auto"` for correct pronunciation and direction.
History rows are editable (`contenteditable`, `role="textbox"`).
