use tauri::{AppHandle, WebviewWindow};

use crate::overlay;

use super::OverlayDragState;

pub(super) async fn start_overlay_drag(
    _app: AppHandle,
    window: &WebviewWindow,
    _state: tauri::State<'_, OverlayDragState>,
) -> Result<Option<u32>, String> {
    window
        .start_dragging()
        .map_err(|error| error.to_string())
        .map(|_| None)
}

pub(super) fn update_overlay_drag(
    _app: AppHandle,
    _state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    Ok(())
}

pub(super) fn end_overlay_drag(
    app: AppHandle,
    _state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    super::finish_overlay_drag(app, overlay::frame_in_single_monitor_work_area_near_point)
}
