use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread::JoinHandle;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

pub const AUDIO_FRAME_EVENT: &str = "audio-frame";
pub const AUDIO_CAPTURE_ERROR_EVENT: &str = "audio-capture-error";
pub const OUTPUT_SAMPLE_RATE: usize = 16_000;
pub const OUTPUT_FORMAT: &str = "pcm_s16le";
pub const OUTPUT_CHANNELS: usize = 1;
#[cfg(target_os = "windows")]
pub const OUTPUT_BITS: usize = 16;
pub const FRAME_MS: usize = 100;
pub const FRAME_BYTES: usize = OUTPUT_SAMPLE_RATE * FRAME_MS / 1000 * 2;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSource {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub is_available: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioFrame {
    pub seq: u64,
    pub sample_rate: u32,
    pub format: String,
    pub data_base64: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioCaptureError {
    pub message: String,
}

#[derive(Default)]
pub struct AudioCaptureState {
    handle: Mutex<Option<CaptureHandle>>,
}

pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    shutdown: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl CaptureHandle {
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    fn new(stop: Arc<AtomicBool>, join: JoinHandle<()>) -> Self {
        Self {
            stop,
            join: Some(join),
            shutdown: None,
        }
    }

    #[cfg(target_os = "linux")]
    fn new_with_shutdown(
        stop: Arc<AtomicBool>,
        join: JoinHandle<()>,
        shutdown: Box<dyn FnOnce() + Send + 'static>,
    ) -> Self {
        Self {
            stop,
            join: Some(join),
            shutdown: Some(shutdown),
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(shutdown) = self.shutdown.take() {
            shutdown();
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(shutdown) = self.shutdown.take() {
            shutdown();
        }
    }
}

pub fn list_audio_sources() -> Vec<AudioSource> {
    platform::list_audio_sources()
}

pub fn start_audio_capture(
    app: AppHandle,
    state: &AudioCaptureState,
    source_id: &str,
) -> Result<(), String> {
    let mut guard = state
        .handle
        .lock()
        .map_err(|_| "audio capture state lock failed".to_string())?;
    if guard.is_some() {
        return Err("audio capture is already running".to_string());
    }
    let handle = platform::start_audio_capture(app, source_id)?;
    *guard = Some(handle);
    Ok(())
}

pub fn stop_audio_capture(state: &AudioCaptureState) -> Result<(), String> {
    let handle = {
        let mut guard = state
            .handle
            .lock()
            .map_err(|_| "audio capture state lock failed".to_string())?;
        guard.take()
    };
    if let Some(handle) = handle {
        handle.stop();
    }
    Ok(())
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
pub fn make_handle(stop: Arc<AtomicBool>, join: JoinHandle<()>) -> CaptureHandle {
    CaptureHandle::new(stop, join)
}

#[cfg(target_os = "linux")]
pub fn make_handle_with_shutdown<F>(
    stop: Arc<AtomicBool>,
    join: JoinHandle<()>,
    shutdown: F,
) -> CaptureHandle
where
    F: FnOnce() + Send + 'static,
{
    CaptureHandle::new_with_shutdown(stop, join, Box::new(shutdown))
}

pub fn emit_audio_frame(app: &AppHandle, seq: u64, data: &[u8]) -> Result<(), tauri::Error> {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;

    app.emit(
        AUDIO_FRAME_EVENT,
        AudioFrame {
            seq,
            sample_rate: OUTPUT_SAMPLE_RATE as u32,
            format: OUTPUT_FORMAT.to_string(),
            data_base64: STANDARD.encode(data),
        },
    )
}

pub fn emit_audio_capture_error(app: &AppHandle, message: impl Into<String>) {
    let _ = app.emit(
        AUDIO_CAPTURE_ERROR_EVENT,
        AudioCaptureError {
            message: message.into(),
        },
    );
}

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "windows")]
use windows as platform;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
use linux as platform;

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
mod platform {
    use super::*;

    pub fn list_audio_sources() -> Vec<AudioSource> {
        vec![AudioSource {
            id: "system_default".to_string(),
            name: "System audio".to_string(),
            kind: "system".to_string(),
            is_available: false,
            detail: "Native system audio capture is not implemented for this platform.".to_string(),
        }]
    }

    pub fn start_audio_capture(_app: AppHandle, _source_id: &str) -> Result<CaptureHandle, String> {
        Err("native system audio capture is not implemented for this platform yet".to_string())
    }
}
