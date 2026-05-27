import { AsrClient } from "./asr-client.js";
import { createFunyiApp } from "./app-controller.js";
import { getAppElements } from "./app-dom.js";
import { createDesktopAudioAdapter } from "./desktop-audio-adapter.js";
import { desktopHost } from "./desktop-host.js";

const app = createFunyiApp({
  audio: createDesktopAudioAdapter(desktopHost),
  dom: getAppElements(),
  overlay: desktopHost,
  createClient: ({ url, ...callbacks }) => new AsrClient({ url, ...callbacks }),
});

void app.boot();
