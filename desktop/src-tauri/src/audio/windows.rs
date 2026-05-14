use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;

use tauri::AppHandle;
use wasapi::{initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat};

use super::{
    emit_audio_capture_error, emit_audio_frame, make_handle, AudioSource, CaptureHandle,
    FRAME_BYTES, OUTPUT_BITS, OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE,
};

const SOURCE_ID: &str = "system_default";

pub fn list_audio_sources() -> Vec<AudioSource> {
    vec![AudioSource {
        id: SOURCE_ID.to_string(),
        name: "Default system output (WASAPI loopback)".to_string(),
        kind: "system".to_string(),
        is_available: true,
        detail: "Captures audio currently playing on the default Windows playback device."
            .to_string(),
    }]
}

pub fn start_audio_capture(app: AppHandle, source_id: &str) -> Result<CaptureHandle, String> {
    if source_id != SOURCE_ID {
        return Err(format!("unsupported audio source: {source_id}"));
    }

    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let join = thread::Builder::new()
        .name("funyi-wasapi-loopback".to_string())
        .spawn(move || {
            if let Err(error) = capture_loop(app.clone(), thread_stop) {
                emit_audio_capture_error(&app, error.to_string());
            }
        })
        .map_err(|error| error.to_string())?;
    Ok(make_handle(stop, join))
}

fn capture_loop(app: AppHandle, stop: Arc<AtomicBool>) -> Result<(), Box<dyn std::error::Error>> {
    initialize_mta().ok()?;

    let enumerator = DeviceEnumerator::new()?;
    let device = enumerator.get_default_device(&Direction::Render)?;
    let mut audio_client = device.get_iaudioclient()?;
    let desired_format = WaveFormat::new(
        OUTPUT_BITS,
        OUTPUT_BITS,
        &SampleType::Int,
        OUTPUT_SAMPLE_RATE,
        OUTPUT_CHANNELS,
        None,
    );
    let block_align = desired_format.get_blockalign() as usize;
    let (_default_period, min_period) = audio_client.get_device_period()?;
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: min_period,
    };

    audio_client.initialize_client(&desired_format, &Direction::Capture, &mode)?;
    let event = audio_client.set_get_eventhandle()?;
    let buffer_frame_count = audio_client.get_buffer_size()?;
    let capture_client = audio_client.get_audiocaptureclient()?;
    let mut sample_queue =
        VecDeque::with_capacity(100 * block_align * (1024 + 2 * buffer_frame_count as usize));
    let mut seq = 0_u64;

    audio_client.start_stream()?;
    while !stop.load(Ordering::SeqCst) {
        capture_client.read_from_device_to_deque(&mut sample_queue)?;
        while sample_queue.len() >= FRAME_BYTES {
            let mut chunk = vec![0_u8; FRAME_BYTES];
            for byte in chunk.iter_mut() {
                *byte = sample_queue.pop_front().unwrap_or(0);
            }
            emit_audio_frame(&app, seq, &chunk)?;
            seq = seq.saturating_add(1);
        }
        let _ = event.wait_for_event(100);
    }
    audio_client.stop_stream()?;
    Ok(())
}
