use tauri::{AppHandle, WebviewWindow};

use crate::overlay;

use super::OverlayDragState;

pub(super) fn start_overlay_drag(
    app: AppHandle,
    window: &WebviewWindow,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<(), String> {
    let position = window.outer_position().map_err(|error| error.to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    *state.drag.lock().map_err(|error| error.to_string())? = Some(super::OverlayDrag {
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
    let drag = *state.drag.lock().map_err(|error| error.to_string())?;
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
) -> Result<(), String> {
    *state.drag.lock().map_err(|error| error.to_string())? = None;
    super::finish_overlay_drag(app, |frame, work_areas, cursor| {
        overlay::snapped_frame_near_point(frame, super::combined_work_bounds(work_areas), cursor)
    })
}
