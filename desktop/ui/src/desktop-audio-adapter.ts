import { decodeBase64Pcm } from "./audio-format.js";
import type { AudioAdapter } from "./audio-adapter.js";
import type { AudioCaptureHost } from "./host-contract.js";

export function createDesktopAudioAdapter(host: AudioCaptureHost): AudioAdapter {
  return {
    decodePcm: decodeBase64Pcm,
    listSources: () => host.listAudioSources(),
    listenCaptureErrors: (handler) => host.listenAudioCaptureErrors(handler),
    listenFrames: (handler) => host.listenAudioFrames(handler),
    startCapture: (sourceId) => host.startAudioCapture(sourceId),
    stopCapture: () => host.stopAudioCapture(),
  };
}
