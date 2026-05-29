# Funyi Desktop

Lightweight Tauri client for the local realtime ASR service.

The UI owns the `/ws/asr` WebSocket. The native layer only captures audio —
system output or microphone input — and emits `pcm_s16le` frames at 16 kHz. This
keeps ASR, translation, and CUDA runtime behavior in the Python service.

The default window is a compact, always-on-top caption strip near the bottom of
the display. Detailed connection settings stay inline, and stable subtitle
history appears automatically when the window is tall enough.

## Status

- Windows: default system output capture via WASAPI loopback.
- macOS: system audio capture through ScreenCaptureKit. The first capture start
  may require Screen & System Audio Recording permission in System Settings.
- macOS: microphone input capture through ScreenCaptureKit on macOS 15+. The
  first microphone start may require Microphone permission in System Settings.
- Linux: desktop builds are disabled until the overlay window layer has Linux
  support; the native capture module has an experimental PipeWire/PulseAudio
  path only.

## Run

From the repository root, start the ASR service first:

```bash
make backend
```

Then run the desktop client:

```bash
make desktop-install
make desktop
```

If your shell does not expose a `pnpm` command, keep using `corepack pnpm ...`
rather than assuming a separate global `pnpm` install exists.

The client only connects to `ws://` loopback addresses (`127.0.0.0/8`,
`localhost`, `::1`); `wss://`, remote hosts, and credentialed URLs are rejected,
matching the Tauri CSP `connect-src ws://127.0.0.1:* ws://localhost:*`.

For Windows plus WSL development, run the Tauri client in the Windows checkout
and point it at the WSL service URL, for example `ws://127.0.0.1:8000/ws/asr`
when the backend is exposed on localhost.

Native Windows Tauri builds also require Visual Studio Build Tools 2022 with
the `Desktop development with C++` workload and a Windows 10/11 SDK. `cargo`
alone is not enough; `cl.exe` and `link.exe` must be available in the build
environment.

The Windows client path is the important path for system-audio capture because
WASAPI loopback captures the default Windows playback device.

## Development Checks

From the repository root:

```bash
make desktop-check
make desktop-format
```

From `desktop/`, the same gates are available as `corepack pnpm run check` and
`corepack pnpm run format`. To run only the fast UI suite (build + type-check +
`node --test`) without the Rust gates:

```bash
corepack pnpm run test       # build UI + UI tests and run them
corepack pnpm run typecheck  # tsc over ui/tsconfig.test.json
```

## Accessibility

Captions are exposed to screen readers through a dedicated polite live region
that announces only *stabilized* lines (never the per-partial current line, to
avoid flooding the reader); a `transcript_final` rebuild never re-announces, and
the log is capped. Caption source and translation are split into spans with
their own BCP-47 `lang` and `dir="auto"` for correct pronunciation and direction.
History rows are editable (`contenteditable`, `role="textbox"`).

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
