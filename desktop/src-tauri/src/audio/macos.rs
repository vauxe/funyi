use tauri::AppHandle;

use super::{AudioSource, CaptureHandle};

const SOURCE_ID: &str = "macos_system_audio";

pub fn list_audio_sources() -> Vec<AudioSource> {
    vec![AudioSource {
        id: SOURCE_ID.to_string(),
        name: "System audio (macOS)".to_string(),
        kind: "system".to_string(),
        is_available: false,
        detail: "macOS system audio capture is intentionally behind this native adapter boundary. It requires a ScreenCaptureKit audio implementation with user approval and release entitlements, or an installed virtual audio device; microphone capture is not a substitute.".to_string(),
    }]
}

pub fn start_audio_capture(_app: AppHandle, _source_id: &str) -> Result<CaptureHandle, String> {
    Err("macOS system audio capture is not enabled in this build; implement ScreenCaptureKit or a virtual-device adapter behind the native audio boundary".to_string())
}
