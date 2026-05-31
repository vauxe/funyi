use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

pub const AUDIO_FRAME_EVENT: &str = "audio-frame";
pub const AUDIO_CAPTURE_ERROR_EVENT: &str = "audio-capture-error";
pub const OUTPUT_SAMPLE_RATE: usize = 16_000;
pub const OUTPUT_FORMAT: &str = "pcm_s16le";
pub const OUTPUT_CHANNELS: usize = 1;
pub const OUTPUT_BITS: usize = 16;
pub const OUTPUT_BYTES_PER_SAMPLE: usize = OUTPUT_BITS / 8;
pub const FRAME_MS: usize = 100;
// Single source of truth for the mono-s16le byte layout. The frame slicer, the
// macOS PCM encoder, and the WASAPI block-align guard all depend on 2 bytes per
// sample, so derive it from the format constants instead of hardcoding `* 2`.
pub const FRAME_BYTES: usize =
    OUTPUT_SAMPLE_RATE * FRAME_MS / 1000 * OUTPUT_CHANNELS * OUTPUT_BYTES_PER_SAMPLE;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSource {
    pub id: String,
    pub name: String,
    pub kind: AudioSourceKind,
    pub is_available: bool,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(target_os = "windows", allow(dead_code))]
pub enum AudioSourceKind {
    System,
    Microphone,
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
}

impl CaptureHandle {
    fn new(stop: Arc<AtomicBool>, join: JoinHandle<()>) -> Self {
        Self {
            stop,
            join: Some(join),
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
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

pub(crate) fn spawn_capture_thread<F>(
    name: &str,
    app: AppHandle,
    run: F,
) -> Result<CaptureHandle, String>
where
    F: FnOnce(AppHandle, Arc<AtomicBool>) -> Result<(), String> + Send + 'static,
{
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let join = thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            if let Err(error) = run(app.clone(), thread_stop) {
                emit_audio_capture_error(&app, error);
            }
        })
        .map_err(|error| error.to_string())?;
    Ok(CaptureHandle::new(stop, join))
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

pub fn emit_pending_audio_frames(
    app: &AppHandle,
    pending: &mut VecDeque<u8>,
    seq: &mut u64,
) -> Result<(), tauri::Error> {
    while let Some(frame) = next_audio_frame(pending) {
        emit_audio_frame(app, *seq, &frame)?;
        *seq = (*seq).saturating_add(1);
    }
    Ok(())
}

fn next_audio_frame(pending: &mut VecDeque<u8>) -> Option<Vec<u8>> {
    if pending.len() < FRAME_BYTES {
        return None;
    }
    Some(pending.drain(..FRAME_BYTES).collect())
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

#[cfg(any(target_os = "macos", test))]
mod pcm;

#[cfg(target_os = "windows")]
use windows as platform;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_audio_frame_leaves_partial_tail_pending() {
        let mut pending = VecDeque::from(vec![7_u8; FRAME_BYTES + 2]);

        let frame = next_audio_frame(&mut pending);

        assert_eq!(frame.as_ref().map(Vec::len), Some(FRAME_BYTES));
        assert_eq!(pending.len(), 2);
        assert!(next_audio_frame(&mut pending).is_none());
    }
}
