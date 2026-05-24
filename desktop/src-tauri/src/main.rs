mod audio;
mod overlay;

use std::sync::Mutex;

use tauri::{AppHandle, LogicalSize, Manager, PhysicalPosition, WebviewWindow};
#[cfg(windows)]
use windows::Win32::Graphics::Gdi::{CreateRectRgn, DeleteObject, SetWindowRgn};

use audio::AudioCaptureState;
use overlay::{Frame, OverlayLayout, OverlayMode, Point, ResizeDirection, WorkBounds};

#[derive(Clone, Copy)]
struct OverlayDrag {
    pointer_start_x: f64,
    pointer_start_y: f64,
    window_start_x: i32,
    window_start_y: i32,
}

#[derive(Default)]
struct OverlayDragState(Mutex<Option<OverlayDrag>>);

#[derive(Clone, Copy)]
struct OverlayResize {
    pointer_start_x: f64,
    pointer_start_y: f64,
    visible_start: Frame,
    direction: ResizeDirection,
    mode: OverlayMode,
    scale: f64,
}

#[derive(Default)]
struct OverlayResizeState(Mutex<Option<OverlayResize>>);

#[derive(Default)]
struct OverlayModeState(Mutex<OverlayMode>);

#[derive(Default)]
struct OverlayLayoutState(Mutex<OverlayLayout>);

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
fn set_overlay_mode(
    app: AppHandle,
    state: tauri::State<'_, OverlayModeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
    mode: String,
) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let mode = OverlayMode::parse(&mode)?;
    let layout = *layout_state.0.lock().map_err(|error| error.to_string())?;
    resize_overlay_window(&window, mode, layout)?;
    *state.0.lock().map_err(|error| error.to_string())? = mode;
    Ok(())
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
    mode_state: tauri::State<'_, OverlayModeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
) -> Result<(), String> {
    *state.0.lock().map_err(|error| error.to_string())? = None;
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let monitor = app
        .monitor_from_point(cursor.x, cursor.y)
        .map_err(|error| error.to_string())?;
    let bounds = match monitor.as_ref().map(work_bounds) {
        Some(bounds) => Some(bounds),
        None => {
            let monitors = window
                .available_monitors()
                .map_err(|error| error.to_string())?;
            desktop_bounds(&monitors)
        }
    };
    let mode = *mode_state.0.lock().map_err(|error| error.to_string())?;
    let layout = *layout_state.0.lock().map_err(|error| error.to_string())?;
    snap_window_to_edge(
        &window,
        mode,
        layout,
        bounds,
        monitor.as_ref().map(|monitor| monitor.scale_factor()),
        Point {
            x: cursor.x.round() as i32,
            y: cursor.y.round() as i32,
        },
    )?;
    #[cfg(windows)]
    set_overlay_window_region(&window, mode, layout)?;
    Ok(())
}

#[tauri::command]
fn start_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
    mode_state: tauri::State<'_, OverlayModeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
    direction: String,
) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let mode = *mode_state.0.lock().map_err(|error| error.to_string())?;
    let direction = ResizeDirection::parse(&direction)?;
    let layout = *layout_state.0.lock().map_err(|error| error.to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let size = window.outer_size().map_err(|error| error.to_string())?;
    let monitor = window_monitor(&window)?;
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    let full_frame = Frame {
        x: position.x,
        y: position.y,
        width: size.width as i32,
        height: size.height as i32,
    };
    #[cfg(windows)]
    let visible_start = visible_overlay_frame(&window, mode, layout, full_frame, Some(scale))?;
    #[cfg(not(windows))]
    let visible_start = full_frame;

    *state.0.lock().map_err(|error| error.to_string())? = Some(OverlayResize {
        pointer_start_x: cursor.x,
        pointer_start_y: cursor.y,
        visible_start,
        direction,
        mode,
        scale,
    });
    Ok(())
}

#[tauri::command]
fn update_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
) -> Result<(), String> {
    let resize = *state.0.lock().map_err(|error| error.to_string())?;
    let Some(resize) = resize else {
        return Ok(());
    };
    apply_overlay_resize(&app, resize, &layout_state)
}

#[tauri::command]
fn end_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
) -> Result<(), String> {
    let resize = {
        let mut state = state.0.lock().map_err(|error| error.to_string())?;
        state.take()
    };
    if let Some(resize) = resize {
        apply_overlay_resize(&app, resize, &layout_state)?;
    }
    Ok(())
}

#[tauri::command]
fn minimize_overlay(app: AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    window.minimize().map_err(|error| error.to_string())
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
        .manage(OverlayResizeState::default())
        .manage(OverlayModeState::default())
        .manage(OverlayLayoutState::default())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = set_overlay_window_layout(
                    &window,
                    OverlayMode::Compact,
                    OverlayLayout::default(),
                );
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            if !matches!(
                event,
                tauri::WindowEvent::Moved(_)
                    | tauri::WindowEvent::Resized(_)
                    | tauri::WindowEvent::ScaleFactorChanged { .. }
            ) {
                return;
            }
            let app = window.app_handle();
            let Some(webview) = app.get_webview_window("main") else {
                return;
            };
            let mode_state = app.state::<OverlayModeState>();
            let Ok(mode) = mode_state.0.lock().map(|mode| *mode) else {
                return;
            };
            let layout_state = app.state::<OverlayLayoutState>();
            let layout = if matches!(
                event,
                tauri::WindowEvent::Resized(_) | tauri::WindowEvent::ScaleFactorChanged { .. }
            ) {
                sync_overlay_layout_from_window(&webview, mode, &layout_state).ok()
            } else {
                layout_state.0.lock().map(|layout| *layout).ok()
            };
            #[cfg(windows)]
            if let Some(layout) = layout {
                let _ = set_overlay_window_region(&webview, mode, layout);
            }
            #[cfg(not(windows))]
            let _ = layout;
        })
        .invoke_handler(tauri::generate_handler![
            list_audio_sources,
            start_audio_capture,
            stop_audio_capture,
            set_overlay_mode,
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

fn set_overlay_window_layout(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    #[cfg(windows)]
    {
        let (width, _) = OverlayMode::History.logical_size();
        let height = layout.height(OverlayMode::History);
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|error| error.to_string())?;
        position_window_near_bottom(window, width, height)?;
        return set_overlay_window_region(window, mode, layout);
    }

    #[cfg(not(windows))]
    {
        let (width, height) = mode.logical_size();
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|error| error.to_string())?;
        position_window_near_bottom(window, width, height)
    }
}

fn resize_overlay_window(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    #[cfg(windows)]
    {
        return set_overlay_window_region(window, mode, layout);
    }

    #[cfg(not(windows))]
    {
        let (width, _) = current_logical_size(window)?;
        let height = layout.height(mode);
        resize_overlay_window_to_logical(window, width, height)
    }
}

#[cfg(not(windows))]
fn resize_overlay_window_to_logical(
    window: &WebviewWindow,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let previous_size = window.outer_size().map_err(|error| error.to_string())?;
    let monitor = window_monitor(window)?;
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    let (target_width, target_height) = (
        (width * scale).round() as i32,
        (height * scale).round() as i32,
    );
    let plan = overlay::resize_plan(
        Frame {
            x: position.x,
            y: position.y,
            width: previous_size.width as i32,
            height: previous_size.height as i32,
        },
        target_width,
        target_height,
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

#[cfg(windows)]
fn set_overlay_window_region(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let outer = window.outer_size().map_err(|error| error.to_string())?;
    let monitor = window_monitor(window)?;
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    let width = (outer.width as i32).max(1);
    let height = (outer.height as i32).max(1);
    let compact_height =
        ((layout.height(OverlayMode::Compact) * scale).round() as i32).clamp(1, height);
    let top = match mode {
        OverlayMode::Compact => height - compact_height,
        OverlayMode::History => 0,
    };

    unsafe {
        let hwnd = window.hwnd().map_err(|error| error.to_string())?;
        let region = CreateRectRgn(0, top, width, height);
        if region.is_invalid() {
            return Err(windows::core::Error::from_win32().to_string());
        }
        if SetWindowRgn(hwnd, Some(region), true) == 0 {
            let _ = DeleteObject(region.into());
            return Err(windows::core::Error::from_win32().to_string());
        }
    }
    Ok(())
}

fn current_logical_size(window: &WebviewWindow) -> Result<(f64, f64), String> {
    let outer = window.outer_size().map_err(|error| error.to_string())?;
    let monitor = window_monitor(window)?;
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    Ok((outer.width as f64 / scale, outer.height as f64 / scale))
}

fn sync_overlay_layout_from_window(
    window: &WebviewWindow,
    mode: OverlayMode,
    state: &OverlayLayoutState,
) -> Result<OverlayLayout, String> {
    let (_, height) = current_logical_size(window)?;
    let mut layout = state.0.lock().map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        layout.set_height(OverlayMode::History, height);
    }
    #[cfg(not(windows))]
    {
        layout.set_height(mode, height);
    }
    let _ = mode;
    Ok(*layout)
}

fn apply_overlay_resize(
    app: &AppHandle,
    resize: OverlayResize,
    layout_state: &OverlayLayoutState,
) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let dx = (cursor.x - resize.pointer_start_x).round() as i32;
    let dy = (cursor.y - resize.pointer_start_y).round() as i32;
    let frame =
        overlay::resized_frame(resize.visible_start, resize.direction, dx, dy, resize.scale);
    apply_resized_visible_frame(&window, resize.mode, frame, layout_state, resize.scale)
}

fn apply_resized_visible_frame(
    window: &WebviewWindow,
    mode: OverlayMode,
    visible_frame: Frame,
    layout_state: &OverlayLayoutState,
    scale: f64,
) -> Result<(), String> {
    let width = overlay::logical_width(visible_frame.width, scale);
    let layout = {
        let mut layout = layout_state.0.lock().map_err(|error| error.to_string())?;
        layout.set_height(mode, visible_frame.height as f64 / scale);
        *layout
    };

    #[cfg(windows)]
    {
        let full_height = layout.height(OverlayMode::History);
        let full_height_px = (full_height * scale).round() as i32;
        let full_y = match mode {
            OverlayMode::Compact => visible_frame.y + visible_frame.height - full_height_px,
            OverlayMode::History => visible_frame.y,
        };
        set_window_position(window, visible_frame.x, full_y)?;
        window
            .set_size(LogicalSize::new(width, full_height))
            .map_err(|error| error.to_string())?;
        return set_overlay_window_region(window, mode, layout);
    }

    #[cfg(not(windows))]
    {
        set_window_position(window, visible_frame.x, visible_frame.y)?;
        window
            .set_size(LogicalSize::new(width, layout.height(mode)))
            .map_err(|error| error.to_string())
    }
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
    mode: OverlayMode,
    layout: OverlayLayout,
    bounds: Option<WorkBounds>,
    target_scale: Option<f64>,
    cursor: Point,
) -> Result<(), String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let size = window.outer_size().map_err(|error| error.to_string())?;
    let full_frame = Frame {
        x: position.x,
        y: position.y,
        width: size.width as i32,
        height: size.height as i32,
    };
    #[cfg(windows)]
    let visible_frame = visible_overlay_frame(window, mode, layout, full_frame, target_scale)?;
    #[cfg(not(windows))]
    let visible_frame = full_frame;
    let frame = overlay::snapped_frame_near_point(visible_frame, bounds, Some(cursor));
    #[cfg(windows)]
    let (x, y) = overlay::full_position_from_visible_frame(mode, full_frame, frame);
    #[cfg(not(windows))]
    let (x, y) = (frame.x, frame.y);
    set_window_position(window, x, y)
}

#[cfg(windows)]
fn visible_overlay_frame(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
    full_frame: Frame,
    target_scale: Option<f64>,
) -> Result<Frame, String> {
    if mode == OverlayMode::History {
        return Ok(full_frame);
    }
    let scale = match target_scale {
        Some(scale) => scale,
        None => {
            let monitor = window_monitor(window)?;
            monitor
                .as_ref()
                .map_or(1.0, |monitor| monitor.scale_factor())
        }
    };
    let height = ((layout.height(OverlayMode::Compact) * scale).round() as i32)
        .clamp(1, full_frame.height.max(1));
    Ok(overlay::compact_visible_frame(full_frame, height))
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
