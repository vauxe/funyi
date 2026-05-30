use std::{thread, time::Duration};

use tauri::{AppHandle, Emitter, Manager, WebviewWindow};
use windows_sys::Win32::Foundation::{HWND, POINT};
use windows_sys::Win32::UI::HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_LBUTTON, VK_RBUTTON};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetCursorPos, GetSystemMetrics, SM_CXPADDEDBORDER, SM_CXSIZEFRAME, SM_CYSIZEFRAME,
    SM_SWAPBUTTON, SYSTEM_METRICS_INDEX,
};

use crate::overlay::{self, Frame, MonitorArea, Point, WorkBounds};

use super::OverlayDragState;

const NATIVE_DRAG_RELEASE_POLL_MS: u64 = 16;
const NATIVE_DRAG_RELEASE_STABLE_POLLS: usize = 3;
const NATIVE_DRAG_RELEASE_STABLE_TIMEOUT_MS: u64 = 500;
const MIN_SNAP_EDGE_TOLERANCE_PX: i32 = 2;
const NATIVE_SNAP_CURSOR_EDGE_TOLERANCE_PX: i32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeDragReleaseSnapshot {
    frame: Frame,
    is_maximized: bool,
    snap_edge_tolerance_px: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapAnchor {
    Left,
    Right,
    Top,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeSnapTrigger {
    Top,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

pub(super) async fn start_overlay_drag(
    app: AppHandle,
    window: &WebviewWindow,
    state: tauri::State<'_, OverlayDragState>,
) -> Result<Option<u32>, String> {
    let drag_id = state.begin_drag()?;
    if let Err(error) = window.start_dragging() {
        let _ = state.finish_drag(drag_id);
        return Err(error.to_string());
    }
    finish_overlay_drag_after_native_release(app, drag_id);
    Ok(Some(drag_id))
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
) -> Result<(), String> {
    Ok(())
}

fn finish_overlay_drag_after_native_release(app: AppHandle, drag_id: u32) {
    tauri::async_runtime::spawn_blocking(move || {
        let release_cursor = wait_for_primary_button_release();
        let finish_result = {
            let state = app.state::<OverlayDragState>();
            finish_native_drag_release(&state, drag_id, || {
                apply_native_drag_release(&app, release_cursor)
            })
        };
        match finish_result {
            Ok(true) => emit_overlay_drag_finished(&app, drag_id, None),
            Ok(false) => {}
            Err(error) => emit_overlay_drag_finished(&app, drag_id, Some(error)),
        }
    });
}

fn emit_overlay_drag_finished(app: &AppHandle, drag_id: u32, error: Option<String>) {
    let _ = app.emit(
        super::OVERLAY_DRAG_FINISHED_EVENT,
        super::OverlayDragFinished::new(drag_id, error),
    );
}

fn finish_native_drag_release<FRelease>(
    state: &OverlayDragState,
    drag_id: u32,
    release: FRelease,
) -> Result<bool, String>
where
    FRelease: FnOnce() -> Result<(), String>,
{
    if !state.begin_finish_drag(drag_id)? {
        return Ok(false);
    }
    let release_result = release();
    let completed = state.complete_finish_drag(drag_id)?;
    if !completed {
        return Ok(false);
    }
    release_result?;
    Ok(true)
}

fn apply_native_drag_release(app: &AppHandle, release_cursor: Option<Point>) -> Result<(), String> {
    let window = super::main_window(app)?;
    let Some(snapshot) = stable_native_drag_release_snapshot(&window)? else {
        return Ok(());
    };
    let monitor_areas = super::available_monitor_areas(&window)?;
    let current_snapshot = native_drag_release_snapshot(&window)?;
    if let Some(frame) = verified_native_drag_release_frame(
        snapshot,
        current_snapshot,
        &monitor_areas,
        release_cursor,
    ) {
        super::set_window_position(&window, frame.x, frame.y)?;
    }
    Ok(())
}

fn stable_native_drag_release_snapshot(
    window: &WebviewWindow,
) -> Result<Option<NativeDragReleaseSnapshot>, String> {
    read_stable_native_drag_release_snapshot(
        || native_drag_release_snapshot(window),
        NATIVE_DRAG_RELEASE_STABLE_POLLS,
        stable_snapshot_max_reads(),
        thread::sleep,
    )
}

fn native_drag_release_snapshot(
    window: &WebviewWindow,
) -> Result<NativeDragReleaseSnapshot, String> {
    Ok(NativeDragReleaseSnapshot {
        frame: super::current_window_frame(window)?,
        is_maximized: window.is_maximized().map_err(|error| error.to_string())?,
        snap_edge_tolerance_px: snap_edge_tolerance_for_window(window)?,
    })
}

fn read_stable_native_drag_release_snapshot<FRead, FSleep>(
    mut read_snapshot: FRead,
    stable_polls: usize,
    max_reads: usize,
    mut sleep: FSleep,
) -> Result<Option<NativeDragReleaseSnapshot>, String>
where
    FRead: FnMut() -> Result<NativeDragReleaseSnapshot, String>,
    FSleep: FnMut(Duration),
{
    let mut last = read_snapshot()?;
    let mut stable_count = 1;
    for _ in 1..max_reads {
        if stable_count >= stable_polls {
            return Ok(Some(last));
        }
        sleep(Duration::from_millis(NATIVE_DRAG_RELEASE_POLL_MS));
        let current = read_snapshot()?;
        if current == last {
            stable_count += 1;
        } else {
            last = current;
            stable_count = 1;
        }
    }
    Ok((stable_count >= stable_polls).then_some(last))
}

fn stable_snapshot_max_reads() -> usize {
    (NATIVE_DRAG_RELEASE_STABLE_TIMEOUT_MS / NATIVE_DRAG_RELEASE_POLL_MS) as usize + 1
}

#[cfg(test)]
fn native_drag_release_should_rebound(
    current_frame: Frame,
    is_maximized: bool,
    monitor_areas: &[MonitorArea],
    snap_edge_tolerance_px: i32,
    release_cursor: Option<Point>,
) -> bool {
    resolved_native_drag_release_frame(
        NativeDragReleaseSnapshot {
            frame: current_frame,
            is_maximized,
            snap_edge_tolerance_px,
        },
        monitor_areas,
        release_cursor,
    )
    .is_some()
}

fn resolved_native_drag_release_frame(
    snapshot: NativeDragReleaseSnapshot,
    monitor_areas: &[MonitorArea],
    release_cursor: Option<Point>,
) -> Option<Frame> {
    if snapshot.is_maximized
        || native_drag_release_matches_native_snap(snapshot, monitor_areas, release_cursor)
    {
        return None;
    }

    let frame = overlay::frame_in_single_monitor_work_area_near_point(
        snapshot.frame,
        monitor_areas,
        release_cursor,
    );
    (frame != snapshot.frame).then_some(frame)
}

fn verified_native_drag_release_frame(
    snapshot: NativeDragReleaseSnapshot,
    current_snapshot: NativeDragReleaseSnapshot,
    monitor_areas: &[MonitorArea],
    release_cursor: Option<Point>,
) -> Option<Frame> {
    (snapshot == current_snapshot)
        .then(|| resolved_native_drag_release_frame(snapshot, monitor_areas, release_cursor))
        .flatten()
}

fn native_drag_release_matches_native_snap(
    snapshot: NativeDragReleaseSnapshot,
    monitor_areas: &[MonitorArea],
    release_cursor: Option<Point>,
) -> bool {
    let Some(release_cursor) = release_cursor else {
        return false;
    };
    monitor_areas.iter().any(|monitor| {
        native_snap_trigger_for_cursor(release_cursor, monitor.bounds).is_some_and(|trigger| {
            frame_matches_native_snap_result(
                snapshot.frame,
                monitor.work_area,
                trigger,
                snapshot.snap_edge_tolerance_px,
            )
        })
    })
}

fn frame_matches_native_snap_result(
    frame: Frame,
    area: WorkBounds,
    trigger: NativeSnapTrigger,
    snap_edge_tolerance_px: i32,
) -> bool {
    let anchors = match trigger {
        NativeSnapTrigger::Top => &[
            SnapAnchor::Top,
            SnapAnchor::TopLeft,
            SnapAnchor::TopRight,
            SnapAnchor::BottomLeft,
            SnapAnchor::BottomRight,
        ][..],
        NativeSnapTrigger::Left => &[SnapAnchor::Left],
        NativeSnapTrigger::Right => &[SnapAnchor::Right],
        NativeSnapTrigger::TopLeft => &[SnapAnchor::TopLeft],
        NativeSnapTrigger::TopRight => &[SnapAnchor::TopRight],
        NativeSnapTrigger::BottomLeft => &[SnapAnchor::BottomLeft],
        NativeSnapTrigger::BottomRight => &[SnapAnchor::BottomRight],
    };

    (trigger == NativeSnapTrigger::Top
        && frame_matches_top_snap_result(frame, area, snap_edge_tolerance_px))
        || frame_matches_snap_anchors(frame, area, anchors, snap_edge_tolerance_px)
}

fn frame_matches_top_snap_result(
    frame: Frame,
    area: WorkBounds,
    snap_edge_tolerance_px: i32,
) -> bool {
    frame_spans_work_area_height(frame, area, snap_edge_tolerance_px)
        && frame_horizontally_within_bounds(frame, area, snap_edge_tolerance_px)
}

fn frame_matches_snap_anchors(
    frame: Frame,
    area: WorkBounds,
    anchors: &[SnapAnchor],
    snap_edge_tolerance_px: i32,
) -> bool {
    anchors.iter().copied().any(|anchor| {
        frame_matches_bounds(
            frame,
            snap_layout_bounds(area, anchor),
            snap_edge_tolerance_px,
        )
    })
}

fn snap_layout_bounds(area: WorkBounds, anchor: SnapAnchor) -> WorkBounds {
    let middle_x = area.left + (area.right - area.left) / 2;
    let middle_y = area.top + (area.bottom - area.top) / 2;
    match anchor {
        SnapAnchor::Left => WorkBounds {
            right: middle_x,
            ..area
        },
        SnapAnchor::Right => WorkBounds {
            left: middle_x,
            ..area
        },
        SnapAnchor::Top => WorkBounds {
            bottom: middle_y,
            ..area
        },
        SnapAnchor::TopLeft => WorkBounds {
            right: middle_x,
            bottom: middle_y,
            ..area
        },
        SnapAnchor::TopRight => WorkBounds {
            left: middle_x,
            bottom: middle_y,
            ..area
        },
        SnapAnchor::BottomLeft => WorkBounds {
            right: middle_x,
            top: middle_y,
            ..area
        },
        SnapAnchor::BottomRight => WorkBounds {
            left: middle_x,
            top: middle_y,
            ..area
        },
    }
}

fn native_snap_trigger_for_cursor(cursor: Point, bounds: WorkBounds) -> Option<NativeSnapTrigger> {
    let tolerance = NATIVE_SNAP_CURSOR_EDGE_TOLERANCE_PX;
    let near_left = edge_aligned(cursor.x, bounds.left, tolerance);
    let near_right = edge_aligned(cursor.x, bounds.right - 1, tolerance);
    let near_top = edge_aligned(cursor.y, bounds.top, tolerance);
    let near_bottom = edge_aligned(cursor.y, bounds.bottom - 1, tolerance);
    let horizontally_in_bounds =
        cursor.x >= bounds.left - tolerance && cursor.x <= bounds.right - 1 + tolerance;
    let vertically_in_bounds =
        cursor.y >= bounds.top - tolerance && cursor.y <= bounds.bottom - 1 + tolerance;

    if near_top && near_left {
        Some(NativeSnapTrigger::TopLeft)
    } else if near_top && near_right {
        Some(NativeSnapTrigger::TopRight)
    } else if near_bottom && near_left {
        Some(NativeSnapTrigger::BottomLeft)
    } else if near_bottom && near_right {
        Some(NativeSnapTrigger::BottomRight)
    } else if near_top && horizontally_in_bounds {
        Some(NativeSnapTrigger::Top)
    } else if near_left && vertically_in_bounds {
        Some(NativeSnapTrigger::Left)
    } else if near_right && vertically_in_bounds {
        Some(NativeSnapTrigger::Right)
    } else {
        None
    }
}

fn frame_spans_work_area_height(
    frame: Frame,
    area: WorkBounds,
    snap_edge_tolerance_px: i32,
) -> bool {
    edge_aligned(frame.y, area.top, snap_edge_tolerance_px)
        && edge_aligned(frame.y + frame.height, area.bottom, snap_edge_tolerance_px)
}

fn frame_horizontally_within_bounds(
    frame: Frame,
    area: WorkBounds,
    snap_edge_tolerance_px: i32,
) -> bool {
    frame.x >= area.left - snap_edge_tolerance_px
        && frame.x + frame.width <= area.right + snap_edge_tolerance_px
}

fn frame_matches_bounds(frame: Frame, bounds: WorkBounds, snap_edge_tolerance_px: i32) -> bool {
    edge_aligned(frame.x, bounds.left, snap_edge_tolerance_px)
        && edge_aligned(frame.y, bounds.top, snap_edge_tolerance_px)
        && edge_aligned(frame.x + frame.width, bounds.right, snap_edge_tolerance_px)
        && edge_aligned(
            frame.y + frame.height,
            bounds.bottom,
            snap_edge_tolerance_px,
        )
}

fn edge_aligned(value: i32, target: i32, snap_edge_tolerance_px: i32) -> bool {
    (value - target).abs() <= snap_edge_tolerance_px
}

fn snap_edge_tolerance_for_window(window: &WebviewWindow) -> Result<i32, String> {
    let hwnd: HWND = window.hwnd().map_err(|error| error.to_string())?.0;
    Ok(snap_edge_tolerance_for_hwnd(hwnd))
}

fn snap_edge_tolerance_for_hwnd(hwnd: HWND) -> i32 {
    let dpi = unsafe { GetDpiForWindow(hwnd) };
    if dpi == 0 {
        return snap_edge_tolerance_from_metrics(|metric| unsafe { GetSystemMetrics(metric) });
    }
    snap_edge_tolerance_from_metrics(|metric| unsafe { GetSystemMetricsForDpi(metric, dpi) })
}

fn snap_edge_tolerance_from_metrics<F>(get_metric: F) -> i32
where
    F: Fn(SYSTEM_METRICS_INDEX) -> i32,
{
    let padded_border = get_metric(SM_CXPADDEDBORDER);
    let x_frame = get_metric(SM_CXSIZEFRAME) + padded_border;
    let y_frame = get_metric(SM_CYSIZEFRAME) + padded_border;
    x_frame.max(y_frame).max(MIN_SNAP_EDGE_TOLERANCE_PX)
}

fn wait_for_primary_button_release() -> Option<Point> {
    wait_for_primary_button_release_with(
        primary_mouse_button_is_down,
        windows_cursor_point,
        thread::sleep,
    )
}

fn wait_for_primary_button_release_with<FIsDown, FReadCursor, FSleep>(
    mut is_down: FIsDown,
    mut read_cursor: FReadCursor,
    mut sleep: FSleep,
) -> Option<Point>
where
    FIsDown: FnMut() -> bool,
    FReadCursor: FnMut() -> Option<Point>,
    FSleep: FnMut(Duration),
{
    let mut release_cursor = None;
    while is_down() {
        release_cursor = read_cursor().or(release_cursor);
        sleep(Duration::from_millis(NATIVE_DRAG_RELEASE_POLL_MS));
    }
    read_cursor().or(release_cursor)
}

fn windows_cursor_point() -> Option<Point> {
    let mut point = POINT { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut point) } == 0 {
        return None;
    }
    Some(Point {
        x: point.x,
        y: point.y,
    })
}

fn primary_mouse_button_is_down() -> bool {
    primary_mouse_button_is_down_with(buttons_are_swapped(), |key| unsafe {
        GetAsyncKeyState(i32::from(key)) < 0
    })
}

fn primary_mouse_button_is_down_with<F>(buttons_swapped: bool, is_down: F) -> bool
where
    F: FnOnce(u16) -> bool,
{
    is_down(primary_mouse_button_virtual_key(buttons_swapped))
}

fn buttons_are_swapped() -> bool {
    unsafe { GetSystemMetrics(SM_SWAPBUTTON) != 0 }
}

fn primary_mouse_button_virtual_key(buttons_swapped: bool) -> u16 {
    if buttons_swapped {
        VK_RBUTTON
    } else {
        VK_LBUTTON
    }
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
    const START_FRAME: Frame = Frame {
        x: 100,
        y: 100,
        width: 960,
        height: 180,
    };
    const LEFT_MONITOR: MonitorArea = MonitorArea {
        bounds: LEFT_SCREEN,
        work_area: LEFT_SCREEN,
    };

    #[test]
    fn drag_state_allows_only_one_finisher_for_current_drag() {
        let state = OverlayDragState::default();
        let first = state.begin_drag().expect("begin first drag");

        assert!(state
            .begin_finish_drag(first)
            .expect("begin finishing first drag"));
        assert!(!state
            .begin_finish_drag(first)
            .expect("skip already finishing drag"));
        assert!(state
            .complete_finish_drag(first)
            .expect("complete first drag"));
        assert!(!state
            .complete_finish_drag(first)
            .expect("skip already finished drag"));

        let second = state.begin_drag().expect("begin second drag");
        assert!(!state
            .begin_finish_drag(first)
            .expect("skip already finished first drag"));
        assert!(state
            .begin_finish_drag(second)
            .expect("begin finishing current drag"));
        assert!(state
            .complete_finish_drag(second)
            .expect("complete current drag"));
    }

    #[test]
    fn drag_state_rejects_new_drag_while_current_drag_is_active_or_finishing() {
        let state = OverlayDragState::default();
        let drag_id = state.begin_drag().expect("begin drag");

        assert!(state.begin_drag().is_err());
        assert!(state
            .begin_finish_drag(drag_id)
            .expect("begin finishing drag"));
        assert!(state.begin_drag().is_err());
        assert!(state.complete_finish_drag(drag_id).expect("complete drag"));

        let next = state.begin_drag().expect("begin next drag");
        assert_ne!(next, drag_id);
    }

    #[test]
    fn active_drag_finish_runs_only_for_current_drag() {
        let state = OverlayDragState::default();
        let current = state.begin_drag().expect("begin current drag");
        let mut ran = false;

        assert!(!finish_native_drag_release(&state, current + 1, || {
            ran = true;
            Ok(())
        })
        .expect("skip wrong finish"));
        assert!(!ran);
        assert!(finish_native_drag_release(&state, current, || {
            ran = true;
            Ok(())
        })
        .expect("finish current"));
        assert!(ran);
    }

    #[test]
    fn native_drag_release_skips_rebound_for_maximized_snap() {
        let out_of_bounds_frame = Frame {
            x: -120,
            y: 100,
            width: 800,
            height: 320,
        };

        assert!(!native_drag_release_should_rebound(
            START_FRAME,
            true,
            &[LEFT_MONITOR],
            16,
            None,
        ));
        assert!(native_drag_release_should_rebound(
            out_of_bounds_frame,
            false,
            &[LEFT_MONITOR],
            16,
            None,
        ));
    }

    #[test]
    fn native_drag_release_does_not_rebound_contained_frame() {
        assert_eq!(
            resolved_native_drag_release_frame(
                NativeDragReleaseSnapshot {
                    frame: START_FRAME,
                    is_maximized: false,
                    snap_edge_tolerance_px: 16,
                },
                &[LEFT_MONITOR],
                Some(Point { x: 500, y: 200 }),
            ),
            None
        );
    }

    #[test]
    fn native_drag_release_resolves_rebound_from_the_stable_snapshot() {
        let out_of_bounds_snapshot = NativeDragReleaseSnapshot {
            frame: Frame {
                x: -120,
                y: 100,
                width: 800,
                height: 320,
            },
            is_maximized: false,
            snap_edge_tolerance_px: 16,
        };
        let maximized_snapshot = NativeDragReleaseSnapshot {
            frame: Frame {
                x: -8,
                y: -8,
                width: 1936,
                height: 1096,
            },
            is_maximized: true,
            snap_edge_tolerance_px: 16,
        };

        assert_eq!(
            resolved_native_drag_release_frame(
                out_of_bounds_snapshot,
                &[LEFT_MONITOR],
                Some(Point { x: 100, y: 100 }),
            ),
            Some(Frame {
                x: 0,
                y: 100,
                ..out_of_bounds_snapshot.frame
            })
        );
        assert_eq!(
            resolved_native_drag_release_frame(
                maximized_snapshot,
                &[LEFT_MONITOR],
                Some(Point { x: 500, y: 0 }),
            ),
            None
        );
    }

    #[test]
    fn native_drag_release_skips_rebound_when_current_snapshot_changed_before_set_position() {
        let out_of_bounds_snapshot = NativeDragReleaseSnapshot {
            frame: Frame {
                x: -120,
                y: 100,
                width: 800,
                height: 320,
            },
            is_maximized: false,
            snap_edge_tolerance_px: 16,
        };
        let moved_snapshot = NativeDragReleaseSnapshot {
            frame: Frame {
                x: -80,
                ..out_of_bounds_snapshot.frame
            },
            ..out_of_bounds_snapshot
        };

        assert_eq!(
            verified_native_drag_release_frame(
                out_of_bounds_snapshot,
                moved_snapshot,
                &[LEFT_MONITOR],
                Some(Point { x: 100, y: 100 }),
            ),
            None
        );
    }

    #[test]
    fn native_drag_release_skips_rebound_when_frame_matches_snap_layout() {
        let snapped_frame = Frame {
            x: 0,
            y: 0,
            width: 960,
            height: 1080,
        };

        assert!(!native_drag_release_should_rebound(
            snapped_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 500 }),
        ));
    }

    #[test]
    fn native_drag_release_reads_cursor_after_button_up() {
        let edge = Point { x: 0, y: 500 };
        let inside = Point { x: 240, y: 500 };

        assert_eq!(
            released_cursor_from_samples([inside, edge]),
            Some(edge),
            "inside to edge before release must keep native snap"
        );
        assert_eq!(
            released_cursor_from_samples([edge, inside]),
            Some(inside),
            "edge to inside before release must allow rebound"
        );
    }

    #[test]
    fn native_drag_release_skips_rebound_for_snap_geometry_when_cursor_is_on_edge() {
        let snapped_frame = Frame {
            x: -8,
            y: -8,
            width: 976,
            height: 1096,
        };

        assert!(!native_drag_release_should_rebound(
            snapped_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 500 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_for_snap_geometry_when_cursor_moved_off_edge() {
        let snapped_frame = Frame {
            x: -8,
            y: -8,
            width: 976,
            height: 1096,
        };

        assert!(native_drag_release_should_rebound(
            snapped_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 8, y: 500 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_for_top_left_geometry_when_cursor_is_on_plain_left_edge() {
        let top_left_snap_frame = Frame {
            x: -8,
            y: -8,
            width: 976,
            height: 556,
        };

        assert!(native_drag_release_should_rebound(
            top_left_snap_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 500 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_for_top_right_geometry_when_cursor_is_on_plain_right_edge() {
        let top_right_snap_frame = Frame {
            x: 952,
            y: -8,
            width: 976,
            height: 556,
        };

        assert!(native_drag_release_should_rebound(
            top_right_snap_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 1919, y: 500 }),
        ));
    }

    #[test]
    fn native_drag_release_uses_monitor_bounds_for_snap_anchor_and_work_area_for_layout() {
        const TASKBAR_BOTTOM_MONITOR: MonitorArea = MonitorArea {
            bounds: WorkBounds {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1080,
            },
            work_area: WorkBounds {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1040,
            },
        };
        let bottom_left_snap_frame = Frame {
            x: 0,
            y: 520,
            width: 960,
            height: 520,
        };

        assert!(!native_drag_release_should_rebound(
            bottom_left_snap_frame,
            false,
            &[TASKBAR_BOTTOM_MONITOR],
            16,
            Some(Point { x: 0, y: 1079 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_for_bottom_edge_without_corner_snap() {
        let bottom_aligned_frame = Frame {
            x: 0,
            y: 600,
            width: 960,
            height: 540,
        };

        assert!(native_drag_release_should_rebound(
            bottom_aligned_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 500, y: 1079 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_when_edge_aligned_frame_is_not_snap_sized() {
        let edge_aligned_frame = Frame {
            x: -120,
            y: 0,
            width: START_FRAME.width,
            height: START_FRAME.height,
        };

        assert!(native_drag_release_should_rebound(
            edge_aligned_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 0 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_for_full_height_frame_that_is_not_snap_sized() {
        let full_height_out_of_bounds_frame = Frame {
            x: -120,
            y: 0,
            width: 960,
            height: 1080,
        };

        assert!(native_drag_release_should_rebound(
            full_height_out_of_bounds_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 500 }),
        ));
    }

    #[test]
    fn native_drag_release_skips_rebound_for_snap_with_windows_frame_border_offsets() {
        let snapped_frame = Frame {
            x: -8,
            y: -8,
            width: 976,
            height: 1096,
        };
        let maximized_frame = Frame {
            x: -8,
            y: -8,
            width: 1936,
            height: 1096,
        };

        assert!(!native_drag_release_should_rebound(
            snapped_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 500 }),
        ));
        assert!(!native_drag_release_should_rebound(
            maximized_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 500, y: 0 }),
        ));
    }

    #[test]
    fn native_drag_release_skips_rebound_for_top_snap_bar_column_layout() {
        let snapped_column_frame = Frame {
            x: 632,
            y: -8,
            width: 656,
            height: 1096,
        };

        assert!(!native_drag_release_should_rebound(
            snapped_column_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 960, y: 0 }),
        ));
    }

    #[test]
    fn native_drag_release_skips_rebound_for_top_snap_bar_partial_layouts() {
        let top_half_frame = Frame {
            x: -8,
            y: -8,
            width: 1936,
            height: 556,
        };
        let top_left_frame = Frame {
            x: -8,
            y: -8,
            width: 976,
            height: 556,
        };
        let top_right_frame = Frame {
            x: 952,
            y: -8,
            width: 976,
            height: 556,
        };

        for frame in [top_half_frame, top_left_frame, top_right_frame] {
            assert!(!native_drag_release_should_rebound(
                frame,
                false,
                &[LEFT_MONITOR],
                16,
                Some(Point { x: 960, y: 0 }),
            ));
        }
    }

    #[test]
    fn native_drag_release_skips_rebound_for_top_snap_bar_bottom_layouts() {
        let bottom_left_frame = Frame {
            x: -8,
            y: 532,
            width: 976,
            height: 556,
        };
        let bottom_right_frame = Frame {
            x: 952,
            y: 532,
            width: 976,
            height: 556,
        };

        for frame in [bottom_left_frame, bottom_right_frame] {
            assert!(!native_drag_release_should_rebound(
                frame,
                false,
                &[LEFT_MONITOR],
                16,
                Some(Point { x: 960, y: 0 }),
            ));
        }
    }

    #[test]
    fn native_drag_release_rebounds_when_border_offset_exceeds_snap_tolerance() {
        let restored_out_of_bounds_frame = Frame {
            x: -40,
            y: -40,
            width: 1000,
            height: 1120,
        };

        assert!(native_drag_release_should_rebound(
            restored_out_of_bounds_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 0, y: 0 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_when_out_of_bounds_frame_is_not_snap_aligned() {
        let restored_out_of_bounds_frame = Frame {
            x: -120,
            y: 100,
            width: 800,
            height: 320,
        };

        assert!(native_drag_release_should_rebound(
            restored_out_of_bounds_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 100, y: 100 }),
        ));
    }

    #[test]
    fn native_drag_release_rebounds_when_cursor_did_not_trigger_snap_edge() {
        let snap_shaped_out_of_bounds_frame = Frame {
            x: -8,
            y: -8,
            width: 976,
            height: 1096,
        };

        assert!(native_drag_release_should_rebound(
            snap_shaped_out_of_bounds_frame,
            false,
            &[LEFT_MONITOR],
            16,
            Some(Point { x: 100, y: 500 }),
        ));
    }

    fn released_cursor_from_samples(cursors: [Point; 2]) -> Option<Point> {
        let mut button_checks = 0;
        let mut cursor_reads = 0;
        let button_states = [true, false];

        wait_for_primary_button_release_with(
            || {
                let is_down = button_states[button_checks.min(button_states.len() - 1)];
                button_checks += 1;
                is_down
            },
            || {
                let cursor = cursors[cursor_reads.min(cursors.len() - 1)];
                cursor_reads += 1;
                Some(cursor)
            },
            |_| {},
        )
    }

    #[test]
    fn native_drag_release_waits_for_a_stable_window_snapshot() {
        let transient = NativeDragReleaseSnapshot {
            frame: START_FRAME,
            is_maximized: false,
            snap_edge_tolerance_px: 16,
        };
        let settled = NativeDragReleaseSnapshot {
            frame: Frame {
                x: 0,
                y: 0,
                width: 960,
                height: 1080,
            },
            is_maximized: false,
            snap_edge_tolerance_px: 16,
        };
        let snapshots = [transient, settled, settled, settled];
        let mut reads = 0;

        let snapshot = read_stable_native_drag_release_snapshot(
            || {
                let snapshot = snapshots[reads.min(snapshots.len() - 1)];
                reads += 1;
                Ok(snapshot)
            },
            3,
            8,
            |_| {},
        )
        .expect("stable snapshot");

        assert_eq!(snapshot, Some(settled));
        assert_eq!(reads, 4);
    }

    #[test]
    fn native_drag_release_skips_rebound_when_snapshot_never_stabilizes() {
        let first = NativeDragReleaseSnapshot {
            frame: START_FRAME,
            is_maximized: false,
            snap_edge_tolerance_px: 16,
        };
        let last = NativeDragReleaseSnapshot {
            frame: Frame {
                x: 20,
                y: 20,
                width: 800,
                height: 300,
            },
            is_maximized: false,
            snap_edge_tolerance_px: 16,
        };
        let snapshots = [first, last];
        let mut reads = 0;

        let snapshot = read_stable_native_drag_release_snapshot(
            || {
                let snapshot = snapshots[reads.min(snapshots.len() - 1)];
                reads += 1;
                Ok(snapshot)
            },
            3,
            2,
            |_| {},
        )
        .expect("snapshot reads");

        assert_eq!(snapshot, None);
        assert_eq!(reads, 2);
    }

    #[test]
    fn native_drag_release_reports_release_error_and_returns_to_idle() {
        let state = OverlayDragState::default();
        let drag_id = state.begin_drag().expect("begin drag");

        assert_eq!(
            finish_native_drag_release(&state, drag_id, || {
                Err("set position failed".to_string())
            })
            .expect_err("surface rebound error"),
            "set position failed"
        );
        assert!(state.begin_drag().is_ok());
    }

    #[test]
    fn primary_mouse_button_poll_uses_windows_swap_setting() {
        let mut checked_key = None;
        assert!(primary_mouse_button_is_down_with(false, |key| {
            checked_key = Some(key);
            true
        }));
        assert_eq!(checked_key, Some(VK_LBUTTON));

        let mut checked_key = None;
        assert!(!primary_mouse_button_is_down_with(true, |key| {
            checked_key = Some(key);
            false
        }));
        assert_eq!(checked_key, Some(VK_RBUTTON));
    }
}
