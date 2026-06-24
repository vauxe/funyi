# Funyi Desktop

Tauri client for the local Funyi ASR service. It captures system audio or
microphone input, sends 16 kHz `pcm_s16le` frames to `/ws/asr`, and uses
`/api/transcriptions` for local file transcription.

## Support

- Windows: system output via WASAPI loopback; microphone via WASAPI input.
- macOS: system audio via ScreenCaptureKit; microphone requires macOS 15+.

## Run

Start the backend from the repository root first. See `../README.md`.

Then run the desktop client:

```powershell
cd desktop
corepack pnpm install
corepack pnpm run dev
```

Connect to:

```text
ws://127.0.0.1:8000/ws/asr
```

Only `ws://` loopback service URLs are accepted. File transcription derives the
matching `http://` loopback API URL.

## Notes

- Windows builds require Visual Studio Build Tools 2022 with the
  `Desktop development with C++` workload and a Windows 10/11 SDK.
- macOS capture may require Screen & System Audio Recording or Microphone
  permission.
- User preferences are stored in the webview's `localStorage`.
