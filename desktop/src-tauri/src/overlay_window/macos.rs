use tauri::{AppHandle, LogicalSize, WebviewWindow, WindowEvent};

use crate::overlay::{self, Frame, OverlayLayout, OverlayMode, Point};

use super::{OverlayDragState, OverlayLayoutState, OverlayModeState};

pub(super) fn is_relevant_window_event(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. }
    )
}

pub(super) fn sync_after_window_event(
    window: &WebviewWindow,
    _event: &WindowEvent,
    mode: OverlayMode,
    layout_state: &OverlayLayoutState,
) {
    let _ = super::sync_overlay_layout_from_window(window, mode, layout_state);
}

pub(super) fn resized_layout_mode(mode: OverlayMode) -> OverlayMode {
    mode
}

pub(super) fn set_overlay_window_layout(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let _ = layout;
    let (width, height) = mode.logical_size();
    window
        .set_size(LogicalSize::new(width, height))
        .map_err(|error| error.to_string())?;
    super::position_window_near_bottom(window, width, height)
}

pub(super) fn resize_overlay_window(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let (width, _) = super::current_logical_size(window)?;
    let height = layout.height(mode);
    resize_overlay_window_to_logical(window, width, height)
}

pub(super) fn start_overlay_drag(
    _app: AppHandle,
    window: &WebviewWindow,
    _state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    window.start_dragging().map_err(|error| error.to_string())
}

pub(super) fn update_overlay_drag(
    _app: AppHandle,
    _state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    Ok(())
}

pub(super) fn end_overlay_drag(
    _app: AppHandle,
    _state: tauri::State<'_, OverlayDragState>,
    _mode_state: tauri::State<'_, OverlayModeState>,
    _layout_state: tauri::State<'_, OverlayLayoutState>,
) -> Result<(), String> {
    Ok(())
}

pub(super) fn finish_native_overlay_drag(app: AppHandle) -> Result<(), String> {
    let window = super::main_window(&app)?;
    let work_areas = super::available_work_bounds(&window)?;
    let cursor = app.cursor_position().ok().map(|position| Point {
        x: position.x.round() as i32,
        y: position.y.round() as i32,
    });
    let frame = overlay::frame_in_single_work_area_near_point(
        super::current_window_frame(&window)?,
        &work_areas,
        cursor,
    );
    super::set_window_position(&window, frame.x, frame.y)
}

pub(super) fn visible_resize_start(
    _window: &WebviewWindow,
    _mode: OverlayMode,
    _layout: OverlayLayout,
    full_frame: Frame,
    _scale: f64,
) -> Result<Frame, String> {
    Ok(full_frame)
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

    super::set_window_position(window, visible_frame.x, visible_frame.y)?;
    window
        .set_size(LogicalSize::new(width, layout.height(mode)))
        .map_err(|error| error.to_string())
}

fn resize_overlay_window_to_logical(
    window: &WebviewWindow,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let monitor = super::window_monitor(window)?;
    let scale = monitor
        .as_ref()
        .map_or(1.0, |monitor| monitor.scale_factor());
    let (target_width, target_height) = (
        (width * scale).round() as i32,
        (height * scale).round() as i32,
    );
    let plan = overlay::resize_plan(
        super::current_window_frame(window)?,
        target_width,
        target_height,
        monitor.as_ref().map(super::work_bounds),
    );

    if plan.move_before_resize {
        super::set_window_position(window, plan.frame.x, plan.frame.y)?;
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|error| error.to_string())?;
    } else {
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|error| error.to_string())?;
        super::set_window_position(window, plan.frame.x, plan.frame.y)?;
    }
    Ok(())
}
