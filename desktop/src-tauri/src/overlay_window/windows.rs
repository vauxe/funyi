use tauri::{AppHandle, WebviewWindow};

use crate::overlay::{self, Frame, Point, WorkBounds};

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
    super::finish_overlay_drag(app, resolve_overlay_drag_frame)
}

fn resolve_overlay_drag_frame(
    frame: Frame,
    work_areas: &[WorkBounds],
    cursor: Option<Point>,
) -> Frame {
    overlay::frame_in_single_work_area_near_point(frame, work_areas, cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT_SCREEN: WorkBounds = WorkBounds {
        left: 0,
        top: 0,
        right: 1920,
        bottom: 1080,
    };
    const RIGHT_SCREEN: WorkBounds = WorkBounds {
        left: 1920,
        top: 0,
        right: 3840,
        bottom: 1080,
    };

    #[test]
    fn release_assigns_cross_monitor_frame_to_cursor_work_area() {
        let frame = Frame {
            x: 1500,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(
            resolve_overlay_drag_frame(
                frame,
                &[LEFT_SCREEN, RIGHT_SCREEN],
                Some(Point { x: 2500, y: 500 })
            ),
            Frame { x: 1920, ..frame }
        );
        assert_eq!(
            resolve_overlay_drag_frame(
                frame,
                &[LEFT_SCREEN, RIGHT_SCREEN],
                Some(Point { x: 1500, y: 500 })
            ),
            Frame { x: 960, ..frame }
        );
    }
}
