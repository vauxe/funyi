use std::sync::Mutex;

use tauri::{
    AppHandle, LogicalSize, Manager, PhysicalPosition, WebviewWindow, Window, WindowEvent,
};

use crate::overlay::{self, Frame, OverlayLayout, OverlayMode, Point, ResizeDirection, WorkBounds};

#[cfg(target_os = "macos")]
#[path = "overlay_window/macos.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "overlay_window/windows.rs"]
mod platform;

#[derive(Clone, Copy)]
#[cfg(target_os = "windows")]
struct OverlayDrag {
    pointer_start_x: f64,
    pointer_start_y: f64,
    window_start_x: i32,
    window_start_y: i32,
}

#[derive(Default)]
pub struct OverlayDragState {
    #[cfg(target_os = "windows")]
    drag: Mutex<Option<OverlayDrag>>,
}

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
pub struct OverlayResizeState(Mutex<Option<OverlayResize>>);

#[derive(Default)]
pub struct OverlayModeState(Mutex<OverlayMode>);

#[derive(Default)]
pub struct OverlayLayoutState(Mutex<OverlayLayout>);

pub fn setup_window(window: &WebviewWindow) -> Result<(), String> {
    set_overlay_window_layout(window, OverlayMode::Compact)
}

pub fn handle_window_event(window: &Window, event: &WindowEvent) {
    if window.label() != "main" {
        return;
    }
    if !is_relevant_window_event(event) {
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

    let _ = sync_overlay_layout_from_window(&webview, mode, &layout_state);
}

pub fn set_overlay_mode(
    app: AppHandle,
    state: tauri::State<'_, OverlayModeState>,
    layout_state: tauri::State<'_, OverlayLayoutState>,
    mode: OverlayMode,
) -> Result<(), String> {
    let window = main_window(&app)?;
    let layout = *layout_state.0.lock().map_err(|error| error.to_string())?;
    resize_overlay_window(&window, mode, layout)?;
    *state.0.lock().map_err(|error| error.to_string())? = mode;
    Ok(())
}

pub fn start_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    let window = main_window(&app)?;
    platform::start_overlay_drag(app, &window, state)
}

pub fn update_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    platform::update_overlay_drag(app, state)
}

pub fn end_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    platform::end_overlay_drag(app, state)
}

pub fn start_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
    mode_state: tauri::State<'_, OverlayModeState>,
    direction: ResizeDirection,
) -> Result<(), String> {
    let window = main_window(&app)?;
    let mode = *mode_state.0.lock().map_err(|error| error.to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let monitor = window_monitor(&window)?;
    let scale = monitor_scale(monitor.as_ref());

    *state.0.lock().map_err(|error| error.to_string())? = Some(OverlayResize {
        pointer_start_x: cursor.x,
        pointer_start_y: cursor.y,
        visible_start: current_window_frame(&window)?,
        direction,
        mode,
        scale,
    });
    Ok(())
}

pub fn update_overlay_resize(
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

pub fn end_overlay_resize(
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

pub fn minimize_overlay(app: AppHandle) -> Result<(), String> {
    main_window(&app)?
        .minimize()
        .map_err(|error| error.to_string())
}

pub fn close_overlay(app: AppHandle) -> Result<(), String> {
    main_window(&app)?
        .close()
        .map_err(|error| error.to_string())
}

fn current_logical_size(window: &WebviewWindow) -> Result<(f64, f64), String> {
    let outer = window.outer_size().map_err(|error| error.to_string())?;
    let scale = current_monitor_scale(window)?;
    Ok((outer.width as f64 / scale, outer.height as f64 / scale))
}

fn sync_overlay_layout_from_window(
    window: &WebviewWindow,
    mode: OverlayMode,
    state: &OverlayLayoutState,
) -> Result<OverlayLayout, String> {
    let (_, height) = current_logical_size(window)?;
    let mut layout = state.0.lock().map_err(|error| error.to_string())?;
    layout.set_height(mode, height);
    Ok(*layout)
}

fn apply_overlay_resize(
    app: &AppHandle,
    resize: OverlayResize,
    layout_state: &OverlayLayoutState,
) -> Result<(), String> {
    let window = main_window(app)?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let dx = (cursor.x - resize.pointer_start_x).round() as i32;
    let dy = (cursor.y - resize.pointer_start_y).round() as i32;
    let frame =
        overlay::resized_frame(resize.visible_start, resize.direction, dx, dy, resize.scale);
    apply_resized_visible_frame(&window, resize.mode, frame, layout_state, resize.scale)
}

fn is_relevant_window_event(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. }
    )
}

fn set_overlay_window_layout(window: &WebviewWindow, mode: OverlayMode) -> Result<(), String> {
    let (width, height) = mode.logical_size();
    set_window_logical_size(window, width, height)?;
    position_window_near_bottom(window, width, height)
}

fn resize_overlay_window(
    window: &WebviewWindow,
    mode: OverlayMode,
    layout: OverlayLayout,
) -> Result<(), String> {
    let (width, _) = current_logical_size(window)?;
    resize_overlay_window_to_logical(window, width, layout.height(mode))
}

fn resize_overlay_window_to_logical(
    window: &WebviewWindow,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let monitor = window_monitor(window)?;
    let scale = monitor_scale(monitor.as_ref());
    let (target_width, target_height) = (
        (width * scale).round() as i32,
        (height * scale).round() as i32,
    );
    let plan = overlay::resize_plan(
        current_window_frame(window)?,
        target_width,
        target_height,
        monitor.as_ref().map(work_bounds),
    );

    if plan.move_before_resize {
        set_window_position(window, plan.frame.x, plan.frame.y)?;
        set_window_logical_size(window, width, height)?;
    } else {
        set_window_logical_size(window, width, height)?;
        set_window_position(window, plan.frame.x, plan.frame.y)?;
    }
    Ok(())
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

    set_window_position(window, visible_frame.x, visible_frame.y)?;
    set_window_logical_size(window, width, layout.height(mode))
}

fn position_window_near_bottom(
    window: &WebviewWindow,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let monitor = window_monitor(window)?;
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

fn current_monitor_scale(window: &WebviewWindow) -> Result<f64, String> {
    Ok(monitor_scale(window_monitor(window)?.as_ref()))
}

fn monitor_scale(monitor: Option<&tauri::Monitor>) -> f64 {
    monitor.map_or(1.0, |monitor| monitor.scale_factor())
}

fn main_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    app.get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())
}

fn available_work_bounds(window: &WebviewWindow) -> Result<Vec<WorkBounds>, String> {
    Ok(window
        .available_monitors()
        .map_err(|error| error.to_string())?
        .iter()
        .map(work_bounds)
        .collect())
}

fn cursor_point(app: &AppHandle) -> Option<Point> {
    app.cursor_position().ok().map(|position| Point {
        x: position.x.round() as i32,
        y: position.y.round() as i32,
    })
}

fn finish_overlay_drag<F>(app: AppHandle, resolve_frame: F) -> Result<(), String>
where
    F: FnOnce(Frame, &[WorkBounds], Option<Point>) -> Frame,
{
    let window = main_window(&app)?;
    let frame = current_window_frame(&window)?;
    let work_areas = available_work_bounds(&window)?;
    let cursor = cursor_point(&app);
    let frame = resolve_frame(frame, &work_areas, cursor);
    set_window_position(&window, frame.x, frame.y)
}

#[cfg(target_os = "windows")]
fn combined_work_bounds(bounds: &[WorkBounds]) -> Option<WorkBounds> {
    let (first, rest) = bounds.split_first()?;
    Some(
        rest.iter()
            .copied()
            .fold(*first, |combined, item| WorkBounds {
                left: combined.left.min(item.left),
                top: combined.top.min(item.top),
                right: combined.right.max(item.right),
                bottom: combined.bottom.max(item.bottom),
            }),
    )
}

fn current_window_frame(window: &WebviewWindow) -> Result<Frame, String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let size = window.outer_size().map_err(|error| error.to_string())?;
    Ok(Frame {
        x: position.x,
        y: position.y,
        width: size.width as i32,
        height: size.height as i32,
    })
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

fn set_window_logical_size(window: &WebviewWindow, width: f64, height: f64) -> Result<(), String> {
    window
        .set_size(LogicalSize::new(width, height))
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
