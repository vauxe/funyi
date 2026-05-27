use serde::Deserialize;

const COLLAPSED_WINDOW_WIDTH: f64 = 960.0;
const COLLAPSED_WINDOW_HEIGHT: f64 = 180.0;
const HISTORY_WINDOW_WIDTH: f64 = 960.0;
const HISTORY_WINDOW_HEIGHT: f64 = 430.0;
const MIN_OVERLAY_WIDTH: f64 = 520.0;
const MIN_OVERLAY_HEIGHT: f64 = 128.0;
#[cfg(any(target_os = "windows", test))]
const SNAP_EDGE_MARGIN_PX: i32 = 42;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OverlayMode {
    #[default]
    Compact,
    History,
}

impl OverlayMode {
    pub fn logical_size(self) -> (f64, f64) {
        match self {
            Self::Compact => (COLLAPSED_WINDOW_WIDTH, COLLAPSED_WINDOW_HEIGHT),
            Self::History => (HISTORY_WINDOW_WIDTH, HISTORY_WINDOW_HEIGHT),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayLayout {
    compact_height: f64,
    history_height: f64,
}

impl Default for OverlayLayout {
    fn default() -> Self {
        Self {
            compact_height: OverlayMode::Compact.logical_size().1,
            history_height: OverlayMode::History.logical_size().1,
        }
    }
}

impl OverlayLayout {
    pub fn height(self, mode: OverlayMode) -> f64 {
        match mode {
            OverlayMode::Compact => self.compact_height,
            OverlayMode::History => self.history_height,
        }
    }

    pub fn set_height(&mut self, mode: OverlayMode, height: f64) {
        match mode {
            OverlayMode::Compact => {
                self.compact_height = clamp_height(height);
                self.history_height = self.history_height.max(self.compact_height);
            }
            OverlayMode::History => {
                self.history_height = height.max(self.compact_height);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum ResizeDirection {
    East,
    North,
    NorthEast,
    NorthWest,
    South,
    SouthEast,
    SouthWest,
    West,
}

impl ResizeDirection {
    fn has_east(self) -> bool {
        matches!(self, Self::East | Self::NorthEast | Self::SouthEast)
    }

    fn has_north(self) -> bool {
        matches!(self, Self::North | Self::NorthEast | Self::NorthWest)
    }

    fn has_south(self) -> bool {
        matches!(self, Self::South | Self::SouthEast | Self::SouthWest)
    }

    fn has_west(self) -> bool {
        matches!(self, Self::West | Self::NorthWest | Self::SouthWest)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Frame {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkBounds {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl WorkBounds {
    fn contains_frame(self, frame: Frame) -> bool {
        frame.x >= self.left
            && frame.y >= self.top
            && frame.x + frame.width <= self.right
            && frame.y + frame.height <= self.bottom
    }

    fn contains_point(self, point: Point) -> bool {
        point.x >= self.left && point.x < self.right && point.y >= self.top && point.y < self.bottom
    }
}

pub fn resized_frame(
    start: Frame,
    direction: ResizeDirection,
    dx: i32,
    dy: i32,
    scale: f64,
) -> Frame {
    let min_width = (MIN_OVERLAY_WIDTH * scale).round() as i32;
    let min_height = (MIN_OVERLAY_HEIGHT * scale).round() as i32;
    let mut left = start.x;
    let mut top = start.y;
    let mut right = start.x + start.width;
    let mut bottom = start.y + start.height;

    if direction.has_west() {
        left += dx;
    }
    if direction.has_east() {
        right += dx;
    }
    if direction.has_north() {
        top += dy;
    }
    if direction.has_south() {
        bottom += dy;
    }

    if right - left < min_width {
        if direction.has_west() {
            left = right - min_width;
        } else {
            right = left + min_width;
        }
    }

    if bottom - top < min_height {
        if direction.has_north() {
            top = bottom - min_height;
        } else {
            bottom = top + min_height;
        }
    }

    Frame {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }
}

pub fn logical_width(physical_width: i32, scale: f64) -> f64 {
    (physical_width as f64 / scale).max(MIN_OVERLAY_WIDTH)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResizePlan {
    pub frame: Frame,
    pub move_before_resize: bool,
}

pub fn resize_plan(
    current: Frame,
    target_width: i32,
    target_height: i32,
    bounds: Option<WorkBounds>,
) -> ResizePlan {
    let x = current.x + (current.width - target_width) / 2;
    let y = current.y + current.height - target_height;
    let (x, y) = clamped_position(bounds, x, y, target_width, target_height);
    ResizePlan {
        frame: Frame {
            x,
            y,
            width: target_width,
            height: target_height,
        },
        move_before_resize: target_height > current.height,
    }
}

#[cfg(any(target_os = "windows", test))]
pub fn snapped_frame_near_point(
    frame: Frame,
    bounds: Option<WorkBounds>,
    snap_point: Option<Point>,
) -> Frame {
    let Some(bounds) = bounds else {
        return frame;
    };
    let (x, y) = snapped_clamped_position(bounds, frame);
    let mut snapped = Frame { x, y, ..frame };
    if let Some(point) = snap_point {
        if point.x <= bounds.left + SNAP_EDGE_MARGIN_PX {
            snapped.x = bounds.left;
        } else if point.x >= bounds.right - SNAP_EDGE_MARGIN_PX {
            snapped.x = bounds.right - frame.width;
        }

        if point.y <= bounds.top + SNAP_EDGE_MARGIN_PX {
            snapped.y = bounds.top;
        } else if point.y >= bounds.bottom - SNAP_EDGE_MARGIN_PX {
            snapped.y = bounds.bottom - frame.height;
        }

        let (x, y) = clamped_position(
            Some(bounds),
            snapped.x,
            snapped.y,
            frame.width,
            frame.height,
        );
        snapped.x = x;
        snapped.y = y;
    }
    snapped
}

pub fn frame_in_single_work_area_near_point(
    frame: Frame,
    work_areas: &[WorkBounds],
    point: Option<Point>,
) -> Frame {
    if work_areas.iter().any(|area| area.contains_frame(frame)) {
        return frame;
    }

    let target = point
        .and_then(|point| {
            work_areas
                .iter()
                .copied()
                .find(|area| area.contains_point(point))
        })
        .or_else(|| best_work_area_for_frame(frame, work_areas));

    target.map_or(frame, |bounds| clamped_frame_to_work_area(frame, bounds))
}

pub fn initial_position(
    bounds: WorkBounds,
    width: i32,
    height: i32,
    bottom_margin: i32,
) -> (i32, i32) {
    let x = bounds.left + (bounds.right - bounds.left - width) / 2;
    let y = bounds.bottom - height - bottom_margin;
    clamped_position(Some(bounds), x, y, width, height)
}

fn clamp_height(height: f64) -> f64 {
    if height.is_finite() {
        height.max(MIN_OVERLAY_HEIGHT)
    } else {
        COLLAPSED_WINDOW_HEIGHT
    }
}

fn clamped_position(
    bounds: Option<WorkBounds>,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> (i32, i32) {
    let Some(bounds) = bounds else {
        return (x, y);
    };

    let max_x = (bounds.right - width).max(bounds.left);
    let max_y = (bounds.bottom - height).max(bounds.top);
    (x.clamp(bounds.left, max_x), y.clamp(bounds.top, max_y))
}

#[cfg(any(target_os = "windows", test))]
fn snapped_clamped_position(bounds: WorkBounds, frame: Frame) -> (i32, i32) {
    let mut x = frame.x;
    let mut y = frame.y;

    if (x - bounds.left).abs() <= SNAP_EDGE_MARGIN_PX {
        x = bounds.left;
    } else if (x + frame.width - bounds.right).abs() <= SNAP_EDGE_MARGIN_PX {
        x = bounds.right - frame.width;
    }

    if (y - bounds.top).abs() <= SNAP_EDGE_MARGIN_PX {
        y = bounds.top;
    } else if (y + frame.height - bounds.bottom).abs() <= SNAP_EDGE_MARGIN_PX {
        y = bounds.bottom - frame.height;
    }

    clamped_position(Some(bounds), x, y, frame.width, frame.height)
}

fn clamped_frame_to_work_area(frame: Frame, bounds: WorkBounds) -> Frame {
    let (x, y) = clamped_position(Some(bounds), frame.x, frame.y, frame.width, frame.height);
    Frame { x, y, ..frame }
}

fn best_work_area_for_frame(frame: Frame, work_areas: &[WorkBounds]) -> Option<WorkBounds> {
    work_areas.iter().copied().max_by_key(|area| {
        (
            frame_intersection_area(frame, *area),
            -frame_distance_to_area_squared(frame, *area),
        )
    })
}

fn frame_intersection_area(frame: Frame, area: WorkBounds) -> i64 {
    let width = (frame.x + frame.width).min(area.right) - frame.x.max(area.left);
    let height = (frame.y + frame.height).min(area.bottom) - frame.y.max(area.top);
    i64::from(width.max(0)) * i64::from(height.max(0))
}

fn frame_distance_to_area_squared(frame: Frame, area: WorkBounds) -> i64 {
    let dx = range_distance(frame.x, frame.x + frame.width, area.left, area.right);
    let dy = range_distance(frame.y, frame.y + frame.height, area.top, area.bottom);
    dx * dx + dy * dy
}

fn range_distance(start_a: i32, end_a: i32, start_b: i32, end_b: i32) -> i64 {
    if end_a < start_b {
        i64::from(start_b - end_a)
    } else if end_b < start_a {
        i64::from(start_a - end_b)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESKTOP: WorkBounds = WorkBounds {
        left: 0,
        top: 0,
        right: 3840,
        bottom: 1080,
    };
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
    fn resize_plan_preserves_bottom_edge() {
        let current = Frame {
            x: 1440,
            y: 856,
            width: 960,
            height: 180,
        };
        let plan = resize_plan(current, 960, 430, Some(DESKTOP));

        assert_eq!(plan.frame.y + plan.frame.height, current.y + current.height);
        assert_eq!(plan.frame.x, current.x);
        assert!(plan.move_before_resize);
    }

    #[test]
    fn resize_plan_clamps_to_top_when_upward_growth_hits_screen_edge() {
        let current = Frame {
            x: 100,
            y: 80,
            width: 960,
            height: 180,
        };
        let plan = resize_plan(current, 960, 430, Some(DESKTOP));

        assert_eq!(plan.frame.y, 0);
    }

    #[test]
    fn resized_frame_corner_changes_both_axes() {
        let frame = Frame {
            x: 100,
            y: 200,
            width: 960,
            height: 180,
        };

        let resized = resized_frame(frame, ResizeDirection::NorthWest, -120, -80, 1.0);

        assert_eq!(
            resized,
            Frame {
                x: -20,
                y: 120,
                width: 1080,
                height: 260,
            },
        );
    }

    #[test]
    fn layout_keeps_history_at_least_compact_height() {
        let mut layout = OverlayLayout::default();

        layout.set_height(OverlayMode::Compact, 600.0);

        assert_eq!(layout.height(OverlayMode::Compact), 600.0);
        assert_eq!(layout.height(OverlayMode::History), 600.0);
    }

    #[test]
    fn windows_snap_point_does_not_stick_to_internal_monitor_seam() {
        let frame = Frame {
            x: 1920,
            y: 400,
            width: 960,
            height: 180,
        };
        let snapped =
            snapped_frame_near_point(frame, Some(DESKTOP), Some(Point { x: 1920, y: 500 }));

        assert_eq!(snapped.x, 1920);
    }

    #[test]
    fn windows_snap_keeps_cross_monitor_visible_frame_inside_desktop_bounds() {
        let frame = Frame {
            x: 1500,
            y: 400,
            width: 960,
            height: 180,
        };
        let snapped =
            snapped_frame_near_point(frame, Some(DESKTOP), Some(Point { x: 1900, y: 500 }));

        assert_eq!(snapped, frame);
    }

    #[test]
    fn macos_release_keeps_fully_contained_frame_unchanged() {
        let frame = Frame {
            x: 400,
            y: 500,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_work_area_near_point(
                frame,
                &[LEFT_SCREEN],
                Some(Point { x: 1600, y: 600 })
            ),
            frame,
        );
    }

    #[test]
    fn macos_release_assigns_cross_monitor_frame_to_cursor_work_area() {
        let frame = Frame {
            x: 1500,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_work_area_near_point(
                frame,
                &[LEFT_SCREEN, RIGHT_SCREEN],
                Some(Point { x: 2500, y: 500 })
            ),
            Frame { x: 1920, ..frame },
        );
        assert_eq!(
            frame_in_single_work_area_near_point(
                frame,
                &[LEFT_SCREEN, RIGHT_SCREEN],
                Some(Point { x: 1500, y: 500 })
            ),
            Frame { x: 960, ..frame },
        );
    }

    #[test]
    fn macos_release_falls_back_to_largest_overlap_without_cursor() {
        let frame = Frame {
            x: 1700,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_work_area_near_point(frame, &[LEFT_SCREEN, RIGHT_SCREEN], None),
            Frame { x: 1920, ..frame },
        );
    }
}
