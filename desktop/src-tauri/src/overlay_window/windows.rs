use tauri::{AppHandle, LogicalSize, WebviewWindow, WindowEvent};
use windows::Win32::Graphics::Gdi::{CreateRectRgn, DeleteObject, SetWindowRgn};

use crate::overlay::{self, Frame, OverlayLayout, OverlayMode, Point, WorkBounds};

use super::{OverlayDragState, OverlayLayoutState, OverlayModeState};

pub(super) fn is_relevant_window_event(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::Moved(_) | WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. }
    )
}

pub(super) fn sync_after_window_event(
    window: &WebviewWindow,
    event: &WindowEvent,
    mode: OverlayMode,
    layout_state: &OverlayLayoutState,
) {
    let layout = if matches!(
        event,
        WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. }
    ) {
        super::sync_overlay_layout_from_window(window, mode, layout_state).ok()
    } else {
        layout_state.0.lock().map(|layout| *layout).ok()
    };
    if let Some(layout) = layout {
        let _ = set_overlay_window_region(window, mode, layout);
    }
}

pub(super) fn resized_layout_mode(_mode: OverlayMode) -> OverlayMode {
    OverlayMode::History
}

pub(super) fn set_overlay_window_layout(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let (width, _) = OverlayMode::History.logical_size();
    let height = layout.height(OverlayMode::History);
    window
        .set_size(LogicalSize::new(width, height))
        .map_err(|error| error.to_string())?;
    super::position_window_near_bottom(window, width, height)?;
    set_overlay_window_region(window, mode, layout)
}

pub(super) fn resize_overlay_window(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    set_overlay_window_region(window, mode, layout)
}

pub(super) fn start_overlay_drag(
    app: AppHandle,
    window: &WebviewWindow,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    *state.0.lock().map_err(|error| error.to_string())? = Some(super::OverlayDrag {
        pointer_start_x: cursor.x,
        pointer_start_y: cursor.y,
        window_start_x: position.x,
        window_start_y: position.y,
    });
    Ok(())
}

pub(super) fn update_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    let drag = *state.0.lock().map_err(|error| error.to_string())?;
    let Some(drag) = drag else {
        return Ok(());
    };
    let window = super::main_window(&app)?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let x = drag.window_start_x + (cursor.x - drag.pointer_start_x).round() as i32;
    let y = drag.window_start_y + (cursor.y - drag.pointer_start_y).round() as i32;
    super::set_window_position(&window, x, y)
}

pub(super) fn end_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
    mode_state: tauri::State<'_, OverlayModeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
) -> Result<(), String> {
    *state.0.lock().map_err(|error| error.to_string())? = None;
    let window = super::main_window(&app)?;
    let mode = *mode_state.0.lock().map_err(|error| error.to_string())?;
    let layout = *layout_state.0.lock().map_err(|error| error.to_string())?;
    finish_manual_drag(&app, &window, mode, layout)
}

pub(super) fn finish_native_overlay_drag(_app: AppHandle) -> Result<(), String> {
    Ok(())
}

pub(super) fn visible_resize_start(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
    full_frame: Frame,
    scale: f64,
) -> Result<Frame, String> {
    visible_overlay_frame(window, mode, layout, full_frame, Some(scale))
}

pub(super) fn apply_resized_visible_frame(
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
    let full_height = layout.height(OverlayMode::History);
    let full_height_px = (full_height * scale).round() as i32;
    let full_y = match mode {
        OverlayMode::Compact => visible_frame.y + visible_frame.height - full_height_px,
        OverlayMode::History => visible_frame.y,
    };
    super::set_window_position(window, visible_frame.x, full_y)?;
    window
        .set_size(LogicalSize::new(width, full_height))
        .map_err(|error| error.to_string())?;
    set_overlay_window_region(window, mode, layout)
}

fn finish_manual_drag(
    app: &tauri::AppHandle,
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let monitor = app
        .monitor_from_point(cursor.x, cursor.y)
        .map_err(|error| error.to_string())?;
    let work_bounds = super::available_work_bounds(window)?;
    let bounds = super::combined_work_bounds(&work_bounds);
    snap_window_to_edge(
        window,
        mode,
        layout,
        bounds,
        monitor.as_ref().map(|monitor| monitor.scale_factor()),
        Point {
            x: cursor.x.round() as i32,
            y: cursor.y.round() as i32,
        },
    )?;
    set_overlay_window_region(window, mode, layout)
}

fn set_overlay_window_region(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let outer = window.outer_size().map_err(|error| error.to_string())?;
    let monitor = super::window_monitor(window)?;
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

fn snap_window_to_edge(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
    bounds: Option<WorkBounds>,
    target_scale: Option<f64>,
    cursor: Point,
) -> Result<(), String> {
    let full_frame = super::current_window_frame(window)?;
    let visible_frame = visible_overlay_frame(window, mode, layout, full_frame, target_scale)?;
    let frame = overlay::snapped_frame_near_point(visible_frame, bounds, Some(cursor));
    let (x, y) = overlay::full_position_from_visible_frame(mode, full_frame, frame);
    super::set_window_position(window, x, y)
}

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
            let monitor = super::window_monitor(window)?;
            monitor
                .as_ref()
                .map_or(1.0, |monitor| monitor.scale_factor())
        }
    };
    let height = ((layout.height(OverlayMode::Compact) * scale).round() as i32)
        .clamp(1, full_frame.height.max(1));
    Ok(overlay::compact_visible_frame(full_frame, height))
}
