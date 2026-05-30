use std::sync::Mutex;

use tauri::{AppHandle, LogicalSize, Manager, PhysicalPosition, WebviewWindow};

#[cfg(target_os = "macos")]
use crate::overlay::Point;
use crate::overlay::{self, Frame, MonitorArea, ResizeDirection, WorkBounds};
#[cfg(any(target_os = "windows", test))]
use serde::Serialize;

#[cfg(any(target_os = "windows", test))]
pub const OVERLAY_DRAG_FINISHED_EVENT: &str = "overlay-drag-finished";
#[cfg(any(target_os = "windows", test))]
const OVERLAY_DRAG_DEFAULT_ERROR: &str = "overlay drag release failed";

#[cfg(any(target_os = "windows", test))]
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverlayDragFinished {
    drag_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[cfg(any(target_os = "windows", test))]
impl OverlayDragFinished {
    pub(super) fn new(drag_id: u32, error: Option<String>) -> Self {
        Self {
            drag_id,
            error: error.map(|error| {
                if error.is_empty() {
                    OVERLAY_DRAG_DEFAULT_ERROR.to_string()
                } else {
                    error
                }
            }),
        }
    }
}

#[cfg(target_os = "macos")]
#[path = "overlay_window/macos.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "overlay_window/windows.rs"]
mod platform;

#[derive(Default)]
pub struct OverlayDragState {
    #[cfg(target_os = "windows")]
    drag: Mutex<OverlayDragLifecycle>,
}

#[cfg(target_os = "windows")]
#[derive(Default)]
struct OverlayDragLifecycle {
    phase: OverlayDragPhase,
    next_id: u32,
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OverlayDragPhase {
    #[default]
    Idle,
    Active(u32),
    Finishing(u32),
}

#[cfg(target_os = "windows")]
impl OverlayDragState {
    fn begin_drag(&self) -> Result<u32, String> {
        let mut drag = self.drag.lock().map_err(|error| error.to_string())?;
        if !matches!(drag.phase, OverlayDragPhase::Idle) {
            return Err("overlay drag is already active".to_string());
        }
        let id = drag.next_id;
        drag.next_id = drag.next_id.wrapping_add(1);
        drag.phase = OverlayDragPhase::Active(id);
        Ok(id)
    }

    fn begin_finish_drag(&self, id: u32) -> Result<bool, String> {
        let mut drag = self.drag.lock().map_err(|error| error.to_string())?;
        if let OverlayDragPhase::Active(active_id) = drag.phase {
            if active_id != id {
                return Ok(false);
            }
            drag.phase = OverlayDragPhase::Finishing(id);
            return Ok(true);
        }
        Ok(false)
    }

    fn complete_finish_drag(&self, id: u32) -> Result<bool, String> {
        let mut drag = self.drag.lock().map_err(|error| error.to_string())?;
        if drag.phase == OverlayDragPhase::Finishing(id) {
            drag.phase = OverlayDragPhase::Idle;
            return Ok(true);
        }
        Ok(false)
    }

    fn finish_drag(&self, id: u32) -> Result<bool, String> {
        let mut drag = self.drag.lock().map_err(|error| error.to_string())?;
        if let OverlayDragPhase::Active(active_id) = drag.phase {
            if active_id != id {
                return Ok(false);
            }
            drag.phase = OverlayDragPhase::Idle;
            return Ok(true);
        }
        Ok(false)
    }
}

#[derive(Clone, Copy)]
struct OverlayResize {
    pointer_start_x: f64,
    pointer_start_y: f64,
    visible_start: Frame,
    direction: ResizeDirection,
    scale: f64,
}

#[derive(Default)]
pub struct OverlayResizeState(Mutex<Option<OverlayResize>>);

pub fn setup_window(window: &WebviewWindow) -> Result<(), String> {
    let (width, height) = overlay::collapsed_logical_size();
    set_window_logical_size(window, width, height)?;
    position_window_near_bottom(window, width, height)
}

pub async fn start_overlay_drag(
    app: AppHandle,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<Option<u32>, String> {
    let window = main_window(&app)?;
    platform::start_overlay_drag(app, &window, state).await
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
    direction: ResizeDirection,
) -> Result<(), String> {
    let window = main_window(&app)?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let monitor = window_monitor(&window)?;
    let scale = monitor_scale(monitor.as_ref());

    *state.0.lock().map_err(|error| error.to_string())? = Some(OverlayResize {
        pointer_start_x: cursor.x,
        pointer_start_y: cursor.y,
        visible_start: current_window_frame(&window)?,
        direction,
        scale,
    });
    Ok(())
}

pub fn update_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
) -> Result<(), String> {
    let resize = *state.0.lock().map_err(|error| error.to_string())?;
    let Some(resize) = resize else {
        return Ok(());
    };
    apply_overlay_resize(&app, resize)
}

pub fn end_overlay_resize(
    app: AppHandle,
    state: tauri::State<'_, OverlayResizeState>,
) -> Result<(), String> {
    let resize = {
        let mut state = state.0.lock().map_err(|error| error.to_string())?;
        state.take()
    };
    if let Some(resize) = resize {
        apply_overlay_resize(&app, resize)?;
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

fn apply_overlay_resize(app: &AppHandle, resize: OverlayResize) -> Result<(), String> {
    let window = main_window(app)?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let dx = (cursor.x - resize.pointer_start_x).round() as i32;
    let dy = (cursor.y - resize.pointer_start_y).round() as i32;
    let frame =
        overlay::resized_frame(resize.visible_start, resize.direction, dx, dy, resize.scale);
    apply_resized_visible_frame(&window, frame, resize.scale)
}

fn apply_resized_visible_frame(
    window: &WebviewWindow,
    visible_frame: Frame,
    scale: f64,
) -> Result<(), String> {
    let width = overlay::logical_width(visible_frame.width, scale);
    let height = overlay::logical_height(visible_frame.height, scale);

    set_window_position(window, visible_frame.x, visible_frame.y)?;
    set_window_logical_size(window, width, height)
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

fn monitor_bounds(monitor: &tauri::Monitor) -> WorkBounds {
    let position = monitor.position();
    let size = monitor.size();
    WorkBounds {
        left: position.x,
        top: position.y,
        right: position.x + size.width as i32,
        bottom: position.y + size.height as i32,
    }
}

fn monitor_area(monitor: &tauri::Monitor) -> MonitorArea {
    MonitorArea {
        bounds: monitor_bounds(monitor),
        work_area: work_bounds(monitor),
    }
}

fn monitor_scale(monitor: Option<&tauri::Monitor>) -> f64 {
    monitor.map_or(1.0, |monitor| monitor.scale_factor())
}

fn main_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    app.get_webview_window("main")
        .ok_or_else(|| "main window was not found".to_string())
}

fn available_monitor_areas(window: &WebviewWindow) -> Result<Vec<MonitorArea>, String> {
    Ok(window
        .available_monitors()
        .map_err(|error| error.to_string())?
        .iter()
        .map(monitor_area)
        .collect())
}

#[cfg(target_os = "macos")]
fn cursor_point(app: &AppHandle) -> Option<Point> {
    app.cursor_position().ok().map(|position| Point {
        x: position.x.round() as i32,
        y: position.y.round() as i32,
    })
}

#[cfg(target_os = "macos")]
fn finish_overlay_drag<F>(app: AppHandle, resolve_frame: F) -> Result<(), String>
where
    F: FnOnce(Frame, &[MonitorArea], Option<Point>) -> Frame,
{
    let cursor = cursor_point(&app);
    finish_overlay_drag_at(app, resolve_frame, cursor)
}

#[cfg(target_os = "macos")]
fn finish_overlay_drag_at<F>(
    app: AppHandle,
    resolve_frame: F,
    cursor: Option<Point>,
) -> Result<(), String>
where
    F: FnOnce(Frame, &[MonitorArea], Option<Point>) -> Frame,
{
    let window = main_window(&app)?;
    let frame = current_window_frame(&window)?;
    let monitor_areas = available_monitor_areas(&window)?;
    let frame = resolve_frame(frame, &monitor_areas, cursor);
    set_window_position(&window, frame.x, frame.y)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn overlay_drag_finished_payload_uses_frontend_contract_shape() {
        assert_eq!(
            serde_json::to_value(OverlayDragFinished {
                drag_id: 42,
                error: None,
            })
            .expect("serialize payload"),
            json!({ "dragId": 42 })
        );
        assert_eq!(
            serde_json::to_value(OverlayDragFinished {
                drag_id: 42,
                error: Some("rebound failed".to_string()),
            })
            .expect("serialize error payload"),
            json!({ "dragId": 42, "error": "rebound failed" })
        );
        assert_eq!(
            serde_json::to_value(OverlayDragFinished::new(42, Some(String::new())))
                .expect("serialize normalized empty error payload"),
            json!({ "dragId": 42, "error": "overlay drag release failed" })
        );
    }
}
