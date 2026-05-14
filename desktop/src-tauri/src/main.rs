mod audio;

use audio::AudioCaptureState;

#[tauri::command]
fn list_audio_sources() -> Vec<audio::AudioSource> {
    audio::list_audio_sources()
}

#[tauri::command]
fn start_audio_capture(
    app: tauri::AppHandle,
    state: tauri::State<'_, AudioCaptureState>,
    source_id: String,
) -> Result<(), String> {
    audio::start_audio_capture(app, &state, &source_id)
}

#[tauri::command]
fn stop_audio_capture(state: tauri::State<'_, AudioCaptureState>) -> Result<(), String> {
    audio::stop_audio_capture(&state)
}

fn main() {
    tauri::Builder::default()
        .manage(AudioCaptureState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_sources,
            start_audio_capture,
            stop_audio_capture
        ])
        .run(tauri::generate_context!())
        .expect("error while running Funyi desktop client");
}
