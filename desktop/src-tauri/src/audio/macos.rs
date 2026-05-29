use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use screencapturekit::cm::ffi;
use screencapturekit::prelude::*;
use tauri::{AppHandle, Manager};

use super::pcm::{append_pcm_s16le, PcmCaptureBuffer};
use super::{
    emit_pending_audio_frames, spawn_capture_thread, AudioSource, AudioSourceKind, CaptureHandle,
    FRAME_BYTES, OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE,
};

const MICROPHONE_SOURCE_PREFIX: &str = "macos_microphone:";
const SYSTEM_SOURCE_ID: &str = "macos_system_audio";

pub fn list_audio_sources() -> Vec<AudioSource> {
    let mut sources = vec![AudioSource {
        id: SYSTEM_SOURCE_ID.to_string(),
        name: "Audio".to_string(),
        kind: AudioSourceKind::System,
        is_available: true,
        detail: "Captures macOS system audio with ScreenCaptureKit. The first start may require Screen & System Audio Recording permission in System Settings.".to_string(),
    }];

    let microphones = AudioInputDevice::list();
    if microphones.is_empty() {
        sources.push(AudioSource {
            id: "macos_microphone_unavailable".to_string(),
            name: "Microphone".to_string(),
            kind: AudioSourceKind::Microphone,
            is_available: false,
            detail: "No macOS microphone input device was reported by ScreenCaptureKit."
                .to_string(),
        });
    } else {
        sources.extend(microphones.into_iter().map(|device| AudioSource {
            id: format!("{MICROPHONE_SOURCE_PREFIX}{}", device.id),
            name: device.name,
            kind: AudioSourceKind::Microphone,
            is_available: true,
            detail: if device.is_default {
                "Captures default microphone input with ScreenCaptureKit. Requires macOS 15+ and Microphone permission in System Settings."
            } else {
                "Captures microphone input with ScreenCaptureKit. Requires macOS 15+ and Microphone permission in System Settings."
            }.to_string(),
        }));
    }

    sources
}

pub fn start_audio_capture(app: AppHandle, source_id: &str) -> Result<CaptureHandle, String> {
    let source = MacAudioSource::parse(source_id)?;

    let target_display = target_display_for_window(&app);
    spawn_capture_thread("funyi-screencapturekit-audio", app, move |app, stop| {
        capture_loop(app, stop, target_display, source)
    })
}

#[derive(Clone, Copy)]
struct TargetDisplay {
    center_x_px: f64,
    center_y_px: f64,
    height_px: u32,
    scale_factor: f64,
    width_px: u32,
}

#[derive(Clone)]
enum MacAudioSource {
    Microphone { device_id: String },
    SystemAudio,
}

struct CaptureState {
    pcm: PcmCaptureBuffer,
    seq: u64,
}

impl MacAudioSource {
    fn parse(source_id: &str) -> Result<Self, String> {
        if source_id == SYSTEM_SOURCE_ID {
            return Ok(Self::SystemAudio);
        }
        if let Some(device_id) = source_id.strip_prefix(MICROPHONE_SOURCE_PREFIX) {
            if device_id.is_empty() {
                return Err("missing macOS microphone device id".to_string());
            }
            return Ok(Self::Microphone {
                device_id: device_id.to_string(),
            });
        }
        Err(format!("unsupported macOS audio source: {source_id}"))
    }
}

fn capture_loop(
    app: AppHandle,
    stop: Arc<AtomicBool>,
    target_display: Option<TargetDisplay>,
    source: MacAudioSource,
) -> Result<(), String> {
    let content = SCShareableContent::get().map_err(|error| error.to_string())?;
    let display = select_display(content.displays(), target_display)?;
    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build();
    let base_config = SCStreamConfiguration::new()
        .with_width(display.width())
        .with_height(display.height())
        .with_sample_rate(OUTPUT_SAMPLE_RATE as i32)
        .with_channel_count(OUTPUT_CHANNELS as i32);
    let (config, output_type) = match &source {
        MacAudioSource::SystemAudio => (
            base_config
                .with_captures_audio(true)
                .with_excludes_current_process_audio(false),
            SCStreamOutputType::Audio,
        ),
        MacAudioSource::Microphone { device_id } => (
            base_config
                .with_captures_microphone(true)
                .with_microphone_capture_device_id(device_id),
            SCStreamOutputType::Microphone,
        ),
    };
    let state = Arc::new(Mutex::new(CaptureState {
        pcm: PcmCaptureBuffer::with_capacity(FRAME_BYTES * 4),
        seq: 0,
    }));
    let handler_app = app.clone();
    let handler_state = Arc::clone(&state);
    let selected_output_type = output_type;
    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(
        move |sample: CMSampleBuffer, output_type: SCStreamOutputType| {
            if output_type == selected_output_type {
                handle_audio_sample(&handler_app, &handler_state, sample);
            }
        },
        output_type,
    );

    stream.start_capture().map_err(|error| error.to_string())?;
    while !stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(50));
    }
    stream.stop_capture().map_err(|error| error.to_string())?;
    Ok(())
}

fn target_display_for_window(app: &AppHandle) -> Option<TargetDisplay> {
    let window = app.get_webview_window("main")?;
    let position = window.outer_position().ok()?;
    let size = window.outer_size().ok()?;
    let monitor = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten())?;
    let monitor_size = monitor.size();
    Some(TargetDisplay {
        center_x_px: position.x as f64 + size.width as f64 / 2.0,
        center_y_px: position.y as f64 + size.height as f64 / 2.0,
        height_px: monitor_size.height,
        scale_factor: monitor.scale_factor(),
        width_px: monitor_size.width,
    })
}

fn select_display(
    displays: Vec<SCDisplay>,
    target_display: Option<TargetDisplay>,
) -> Result<SCDisplay, String> {
    if let Some(target) = target_display {
        if let Some(display) = displays
            .iter()
            .find(|display| display_contains_target(display, target))
        {
            return Ok(display.clone());
        }
        if let Some(display) = displays
            .iter()
            .find(|display| display_size_matches_target(display, target))
        {
            return Ok(display.clone());
        }
    }

    displays
        .into_iter()
        .next()
        .ok_or_else(|| "ScreenCaptureKit did not report any capturable displays".to_string())
}

fn display_contains_target(display: &SCDisplay, target: TargetDisplay) -> bool {
    let frame = display.frame();
    let center_x_pt = target.center_x_px / target.scale_factor;
    let center_y_pt = target.center_y_px / target.scale_factor;
    point_in_rect(center_x_pt, center_y_pt, frame)
        || point_in_rect(target.center_x_px, target.center_y_px, frame)
}

fn display_size_matches_target(display: &SCDisplay, target: TargetDisplay) -> bool {
    near_u32(display.width(), target.width_px, 2) && near_u32(display.height(), target.height_px, 2)
}

fn point_in_rect(x: f64, y: f64, rect: screencapturekit::cg::CGRect) -> bool {
    x >= rect.min_x() && x < rect.max_x() && y >= rect.min_y() && y < rect.max_y()
}

fn near_u32(left: u32, right: u32, tolerance: u32) -> bool {
    left.abs_diff(right) <= tolerance
}

fn handle_audio_sample(app: &AppHandle, state: &Arc<Mutex<CaptureState>>, sample: CMSampleBuffer) {
    if !sample.is_valid() {
        return;
    }
    let Some(audio_buffers) = sample.audio_buffer_list() else {
        return;
    };
    let Some(audio_buffer) = audio_buffers.get(0) else {
        return;
    };
    let sample_count = sample.num_samples().max(0) as usize;
    let input_rate = audio_sample_rate(&sample).unwrap_or(OUTPUT_SAMPLE_RATE);
    let data = audio_buffer.data();
    let Ok(mut state) = state.lock() else {
        return;
    };
    append_pcm_s16le(&mut state.pcm, data, sample_count, input_rate);
    let CaptureState { pcm, seq } = &mut *state;
    let _ = emit_pending_audio_frames(app, &mut pcm.pending, seq);
}

fn audio_sample_rate(sample: &CMSampleBuffer) -> Option<usize> {
    let description = unsafe { ffi::cm_sample_buffer_get_format_description(sample.as_ptr()) };
    if description.is_null() {
        return None;
    }
    let rate = unsafe { ffi::cm_format_description_get_audio_sample_rate(description) };
    if rate.is_finite() && rate > 0.0 {
        Some(rate.round() as usize)
    } else {
        None
    }
}
