mod audio;
mod overlay;

use std::sync::Mutex;

use tauri::{AppHandle, LogicalSize, Manager, PhysicalPosition, WebviewWindow};

use audio::AudioCaptureState;
use overlay::{Frame, OverlayMode, Point, WorkBounds};

#[derive(Clone, Copy)]
struct OverlayDrag {
    pointer_start_x: f64,
    pointer_start_y: f64,
    window_start_x: i32,
    window_start_y: i32,
}

#[derive(Default)]
struct OverlayDragState(Mutex<Option<OverlayDrag>>);

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
fn set_overlay_mode(app: AppHandle, mode: String) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    resize_overlay_window(&window, OverlayMode::parse(&mode)?)
}

#[tauri::command]
fn start_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    *state.0.lock().map_err(|error| error.to_string())? = Some(OverlayDrag {
        pointer_start_x: cursor.x,
        pointer_start_y: cursor.y,
        window_start_x: position.x,
        window_start_y: position.y,
    });
    Ok(())
}

#[tauri::command]
fn update_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    let drag = *state.0.lock().map_err(|error| error.to_string())?;
    let Some(drag) = drag else {
        return Ok(());
    };
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let x = drag.window_start_x + (cursor.x - drag.pointer_start_x).round() as i32;
    let y = drag.window_start_y + (cursor.y - drag.pointer_start_y).round() as i32;
    set_window_position(&window, x, y)
}

#[tauri::command]
fn end_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    *state.0.lock().map_err(|error| error.to_string())? = None;
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let monitor = app
        .monitor_from_point(cursor.x, cursor.y)
        .map_err(|error| error.to_string())?;
    let monitors = window
        .available_monitors()
        .map_err(|error| error.to_string())?;
    let bounds = desktop_bounds(&monitors).or_else(|| monitor.as_ref().map(work_bounds));
    snap_window_to_edge(
        &window,
        bounds,
        Point {
            x: cursor.x.round() as i32,
            y: cursor.y.round() as i32,
        },
    )
}

#[tauri::command]
fn close_overlay(app: AppHandle, state: tauri::State<'_, AudioCaptureState>) -> Result<(), String> {
    let _ = audio::stop_audio_capture(&state);
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    window.close().map_err(|error| error.to_string())
}

fn main() {
    tauri::Builder::default()
        .manage(AudioCaptureState::default())
        .manage(OverlayDragState::default())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = set_overlay_window_layout(&window, OverlayMode::Compact);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_audio_sources,
            start_audio_capture,
            stop_audio_capture,
            set_overlay_mode,
            start_overlay_drag,
            update_overlay_drag,
            end_overlay_drag,
            close_overlay
        ])
        .run(tauri::generate_context!())
        .expect("error while running Funyi desktop client");
}

fn set_overlay_window_layout(window: &WebviewWindow, mode: OverlayMode) -> Result<(), String> {
    let (width, height) = mode.logical_size();
    window
        .set_size(LogicalSize::new(width, height))
        .map_err(|error| error.to_string())?;
    position_window_near_bottom(window, width, height)
}

fn resize_overlay_window(window: &WebviewWindow, mode: OverlayMode) -> Result<(), String> {
    let (width, height) = mode.logical_size();
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let previous_size = window.outer_size().map_err(|error| error.to_string())?;
    let monitor = window_monitor(window)?;
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    let plan = overlay::resize_plan(
        Frame {
            x: position.x,
            y: position.y,
            width: previous_size.width as i32,
            height: previous_size.height as i32,
        },
        (width * scale).round() as i32,
        (height * scale).round() as i32,
        monitor.as_ref().map(work_bounds),
    );

    if plan.move_before_resize {
        set_window_position(window, plan.frame.x, plan.frame.y)?;
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|error| error.to_string())?;
    } else {
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|error| error.to_string())?;
        set_window_position(window, plan.frame.x, plan.frame.y)?;
    }
    Ok(())
}

fn position_window_near_bottom(
    window: &WebviewWindow,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let monitor = window
        .current_monitor()
        .map_err(|error| error.to_string())?
        .or(window
            .primary_monitor()
            .map_err(|error| error.to_string())?);
    let Some(monitor) = monitor else {
        return Ok(());
    };
    let scale = monitor.scale_factor();
    let width_px = (width * scale).round() as i32;
    let height_px = (height * scale).round() as i32;
    let margin_px = (44.0 * scale).round() as i32;
    let (x, y) = overlay::initial_position(work_bounds(&monitor), width_px, height_px, margin_px);
    set_window_position(window, x, y)
}

fn snap_window_to_edge(
    window: &WebviewWindow,
    bounds: Option<WorkBounds>,
    cursor: Point,
) -> Result<(), String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let size = window.outer_size().map_err(|error| error.to_string())?;
    let frame = overlay::snapped_frame_near_point(
        Frame {
            x: position.x,
            y: position.y,
            width: size.width as i32,
            height: size.height as i32,
        },
        bounds,
        Some(cursor),
    );
    set_window_position(window, frame.x, frame.y)
}

fn work_bounds(monitor: &tauri::Monitor) -> WorkBounds {
    let work_area = monitor.work_area();
    let left = work_area.position.x;
    let top = work_area.position.y;
    WorkBounds {
        left,
        top,
        right: left + work_area.size.width as i32,
        bottom: top + work_area.size.height as i32,
    }
}

fn desktop_bounds(monitors: &[tauri::Monitor]) -> Option<WorkBounds> {
    overlay::combined_bounds(monitors.iter().map(work_bounds))
}

fn set_window_position(window: &WebviewWindow, x: i32, y: i32) -> Result<(), String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    if x == position.x && y == position.y {
        return Ok(());
    }
    window
        .set_position(PhysicalPosition::new(x, y))
        .map_err(|error| error.to_string())
}

fn window_monitor(window: &WebviewWindow) -> Result<Option<tauri::Monitor>, String> {
    Ok(window
        .current_monitor()
        .map_err(|error| error.to_string())?
        .or(window
            .primary_monitor()
            .map_err(|error| error.to_string())?))
}
