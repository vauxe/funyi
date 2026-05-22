use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use screencapturekit::cm::ffi;
use screencapturekit::prelude::*;
use tauri::{AppHandle, Manager};

use super::{
    emit_audio_capture_error, emit_audio_frame, make_handle, AudioSource, CaptureHandle,
    FRAME_BYTES, OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE,
};

const MICROPHONE_SOURCE_PREFIX: &str = "macos_microphone:";
const SYSTEM_SOURCE_ID: &str = "macos_system_audio";

pub fn list_audio_sources() -> Vec<AudioSource> {
    let mut sources = vec![AudioSource {
        id: SYSTEM_SOURCE_ID.to_string(),
        name: "System audio (ScreenCaptureKit)".to_string(),
        kind: "system".to_string(),
        is_available: true,
        detail: "Captures macOS system audio with ScreenCaptureKit. The first start may require Screen & System Audio Recording permission in System Settings.".to_string(),
    }];

    let microphones = AudioInputDevice::list();
    if microphones.is_empty() {
        sources.push(AudioSource {
            id: "macos_microphone_unavailable".to_string(),
            name: "Microphone".to_string(),
            kind: "microphone".to_string(),
            is_available: false,
            detail: "No macOS microphone input device was reported by ScreenCaptureKit."
                .to_string(),
        });
    } else {
        sources.extend(microphones.into_iter().map(|device| AudioSource {
            id: format!("{MICROPHONE_SOURCE_PREFIX}{}", device.id),
            name: if device.is_default {
                format!("{} (default microphone)", device.name)
            } else {
                format!("{} (microphone)", device.name)
            },
            kind: "microphone".to_string(),
            is_available: true,
            detail: "Captures microphone input with ScreenCaptureKit. Requires macOS 15+ and Microphone permission in System Settings.".to_string(),
        }));
    }

    sources
}

pub fn start_audio_capture(app: AppHandle, source_id: &str) -> Result<CaptureHandle, String> {
    let source = MacAudioSource::parse(source_id)?;

    let target_display = target_display_for_window(&app);
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let join = thread::Builder::new()
        .name("funyi-screencapturekit-audio".to_string())
        .spawn(move || {
            if let Err(error) = capture_loop(app.clone(), thread_stop, target_display, source) {
                emit_audio_capture_error(&app, error);
            }
        })
        .map_err(|error| error.to_string())?;
    Ok(make_handle(stop, join))
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

struct CaptureBuffer {
    pending: VecDeque<u8>,
    resampler: PcmResampler,
    seq: u64,
}

struct PcmResampler {
    input_index: u64,
    input_rate: usize,
    next_output_at: f64,
    previous_sample: Option<(f64, f32)>,
}

impl PcmResampler {
    fn new() -> Self {
        Self {
            input_index: 0,
            input_rate: OUTPUT_SAMPLE_RATE,
            next_output_at: 0.0,
            previous_sample: None,
        }
    }

    fn reset(&mut self, input_rate: usize) {
        self.input_index = 0;
        self.input_rate = input_rate.max(1);
        self.next_output_at = 0.0;
        self.previous_sample = None;
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
    let state = Arc::new(Mutex::new(CaptureBuffer {
        pending: VecDeque::with_capacity(FRAME_BYTES * 4),
        resampler: PcmResampler::new(),
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
    if displays.is_empty() {
        return Err("ScreenCaptureKit did not report any capturable displays".to_string());
    }

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

    Ok(displays.into_iter().next().unwrap())
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

fn handle_audio_sample(app: &AppHandle, state: &Arc<Mutex<CaptureBuffer>>, sample: CMSampleBuffer) {
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
    append_pcm_s16le(&mut state, data, sample_count, input_rate);
    drain_frames(app, &mut state);
}

fn append_pcm_s16le(
    state: &mut CaptureBuffer,
    data: &[u8],
    sample_count: usize,
    input_rate: usize,
) {
    if input_rate == OUTPUT_SAMPLE_RATE && sample_count > 0 && data.len() == sample_count * 2 {
        state.pending.extend(data.iter().copied());
        state.resampler.input_index = state
            .resampler
            .input_index
            .saturating_add(sample_count as u64);
        state.resampler.next_output_at = state.resampler.input_index as f64;
        state.resampler.previous_sample = None;
        return;
    }

    let samples = extract_mono_samples(data, sample_count);
    if samples.is_empty() {
        return;
    }

    let input_rate = input_rate.max(1);
    if state.resampler.input_rate != input_rate {
        state.resampler.reset(input_rate);
    }

    if input_rate == OUTPUT_SAMPLE_RATE {
        for sample in samples {
            push_i16_sample(&mut state.pending, sample);
        }
        state.resampler.input_index = state
            .resampler
            .input_index
            .saturating_add(sample_count as u64);
        state.resampler.next_output_at = state.resampler.input_index as f64;
        state.resampler.previous_sample = None;
        return;
    }

    let step = input_rate as f64 / OUTPUT_SAMPLE_RATE as f64;
    for sample in samples {
        let current_at = state.resampler.input_index as f64;
        while state.resampler.next_output_at <= current_at {
            let output_sample =
                if let Some((previous_at, previous_sample)) = state.resampler.previous_sample {
                    let span = (current_at - previous_at).max(f64::EPSILON);
                    let t = ((state.resampler.next_output_at - previous_at) / span).clamp(0.0, 1.0);
                    previous_sample + (sample - previous_sample) * t as f32
                } else {
                    sample
                };
            push_i16_sample(&mut state.pending, output_sample);
            state.resampler.next_output_at += step;
        }
        state.resampler.previous_sample = Some((current_at, sample));
        state.resampler.input_index = state.resampler.input_index.saturating_add(1);
    }
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

fn extract_mono_samples(data: &[u8], sample_count: usize) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    if sample_count > 0 {
        let f32_len = sample_count.saturating_mul(4);
        let i16_len = sample_count.saturating_mul(2);
        if data.len() == f32_len {
            return data.chunks_exact(4).map(f32_sample).collect();
        }
        if data.len() == i16_len {
            return data.chunks_exact(2).map(i16_sample).collect();
        }
        if f32_len > 0 && data.len() % f32_len == 0 {
            let channels = data.len() / f32_len;
            return average_interleaved_f32(data, sample_count, channels);
        }
        if i16_len > 0 && data.len() % i16_len == 0 {
            let channels = data.len() / i16_len;
            return average_interleaved_i16(data, sample_count, channels);
        }
    }

    data.chunks_exact(2).map(i16_sample).collect()
}

fn average_interleaved_f32(data: &[u8], sample_count: usize, channels: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .map(f32_sample)
        .collect::<Vec<_>>()
        .chunks_exact(channels)
        .take(sample_count)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn average_interleaved_i16(data: &[u8], sample_count: usize, channels: usize) -> Vec<f32> {
    data.chunks_exact(2)
        .map(i16_sample)
        .collect::<Vec<_>>()
        .chunks_exact(channels)
        .take(sample_count)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn f32_sample(chunk: &[u8]) -> f32 {
    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0)
}

fn i16_sample(chunk: &[u8]) -> f32 {
    i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0
}

fn push_i16_sample(output: &mut VecDeque<u8>, sample: f32) {
    let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    output.extend(scaled.to_le_bytes());
}

fn drain_frames(app: &AppHandle, state: &mut CaptureBuffer) {
    while state.pending.len() >= FRAME_BYTES {
        let mut frame = vec![0_u8; FRAME_BYTES];
        for byte in frame.iter_mut() {
            *byte = state.pending.pop_front().unwrap_or(0);
        }
        if emit_audio_frame(app, state.seq, &frame).is_err() {
            break;
        }
        state.seq = state.seq.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_96khz_float_mic_to_16khz_pcm() {
        let mut state = capture_buffer();
        let input = f32_bytes((0..96).map(|index| index as f32 / 96.0));

        append_pcm_s16le(&mut state, &input, 96, 96_000);

        assert_eq!(state.pending.len(), 16 * 2);
    }

    #[test]
    fn preserves_16khz_i16_pcm() {
        let mut state = capture_buffer();
        let mut input = Vec::new();
        for value in [-32768_i16, 0, 32767] {
            input.extend(value.to_le_bytes());
        }

        append_pcm_s16le(&mut state, &input, 3, 16_000);

        assert_eq!(state.pending.into_iter().collect::<Vec<_>>(), input);
    }

    #[test]
    fn averages_interleaved_stereo_float_samples() {
        let mut input = Vec::new();
        for sample in [0.0_f32, 1.0, -1.0, 1.0] {
            input.extend(sample.to_le_bytes());
        }

        assert_eq!(extract_mono_samples(&input, 2), vec![0.5, 0.0]);
    }

    fn capture_buffer() -> CaptureBuffer {
        CaptureBuffer {
            pending: VecDeque::new(),
            resampler: PcmResampler::new(),
            seq: 0,
        }
    }

    fn f32_bytes(samples: impl IntoIterator<Item = f32>) -> Vec<u8> {
        let mut bytes = Vec::new();
        for sample in samples {
            bytes.extend(sample.to_le_bytes());
        }
        bytes
    }
}
