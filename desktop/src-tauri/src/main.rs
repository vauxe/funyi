mod audio;
mod overlay;
mod overlay_window;

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
compile_error!("Funyi desktop supports only Windows and macOS.");

use tauri::{AppHandle, Manager};

use audio::AudioCaptureState;
use overlay::ResizeDirection;
use overlay_window::{OverlayDragState, OverlayResizeState};

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

#[tauri::command]
fn start_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    overlay_window::start_overlay_drag(app, state)
}

#[tauri::command]
fn update_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    overlay_window::update_overlay_drag(app, state)
}

#[tauri::command]
fn end_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    overlay_window::end_overlay_drag(app, state)
}

#[tauri::command]
fn start_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
    direction: ResizeDirection,
) -> Result<(), String> {
    overlay_window::start_overlay_resize(app, state, direction)
}

#[tauri::command]
fn update_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
) -> Result<(), String> {
    overlay_window::update_overlay_resize(app, state)
}

#[tauri::command]
fn end_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
) -> Result<(), String> {
    overlay_window::end_overlay_resize(app, state)
}

#[tauri::command]
fn minimize_overlay(app: AppHandle) -> Result<(), String> {
    overlay_window::minimize_overlay(app)
}

#[tauri::command]
fn close_overlay(app: AppHandle, state: tauri::State<'_, AudioCaptureState>) -> Result<(), String> {
    let _ = audio::stop_audio_capture(&state);
    overlay_window::close_overlay(app)
}

fn main() {
    let builder = tauri::Builder::default()
        .manage(AudioCaptureState::default())
        .manage(OverlayDragState::default())
        .manage(OverlayResizeState::default());

    builder
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = overlay_window::setup_window(&window);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_audio_sources,
            start_audio_capture,
            stop_audio_capture,
            start_overlay_drag,
            update_overlay_drag,
            end_overlay_drag,
            start_overlay_resize,
            update_overlay_resize,
            end_overlay_resize,
            minimize_overlay,
            close_overlay
        ])
        .run(tauri::generate_context!())
        .expect("error while running Funyi desktop client");
}
