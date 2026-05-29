use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tauri::AppHandle;
use wasapi::{initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat};

use super::{
    emit_pending_audio_frames, spawn_capture_thread, AudioSource, AudioSourceKind, CaptureHandle,
    OUTPUT_BITS, OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE,
};

const SOURCE_ID: &str = "system_default";

pub fn list_audio_sources() -> Vec<AudioSource> {
    vec![AudioSource {
        id: SOURCE_ID.to_string(),
        name: "Audio".to_string(),
        kind: AudioSourceKind::System,
        is_available: true,
        detail: "Captures audio currently playing on the default Windows playback device with WASAPI loopback."
            .to_string(),
    }]
}

pub fn start_audio_capture(app: AppHandle, source_id: &str) -> Result<CaptureHandle, String> {
    if source_id != SOURCE_ID {
        return Err(format!("unsupported audio source: {source_id}"));
    }

    spawn_capture_thread("funyi-wasapi-loopback", app, |app, stop| {
        capture_loop(app, stop).map_err(|error| error.to_string())
    })
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
    // Shared-mode `autoconvert` makes WASAPI deliver bytes in `desired_format`
    // (16 kHz / mono / s16), which is what downstream frame slicing assumes. If a
    // driver ever fails to honor that, captured audio would be mislabeled; a runtime
    // assertion of the negotiated mix format would harden this further.
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
        emit_pending_audio_frames(&app, &mut sample_queue, &mut seq)?;
        let _ = event.wait_for_event(100);
    }
    audio_client.stop_stream()?;
    Ok(())
}
