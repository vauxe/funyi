use serde::Deserialize;

const COLLAPSED_WINDOW_WIDTH: f64 = 960.0;
const COLLAPSED_WINDOW_HEIGHT: f64 = 180.0;
const MIN_OVERLAY_WIDTH: f64 = 280.0;
const MIN_OVERLAY_HEIGHT: f64 = 128.0;

pub fn collapsed_logical_size() -> (f64, f64) {
    (COLLAPSED_WINDOW_WIDTH, COLLAPSED_WINDOW_HEIGHT)
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
pub struct MonitorArea {
    pub bounds: WorkBounds,
    pub work_area: WorkBounds,
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

pub fn logical_height(physical_height: i32, scale: f64) -> f64 {
    clamp_height(physical_height as f64 / scale)
}

pub fn frame_in_single_monitor_work_area_near_point(
    frame: Frame,
    monitor_areas: &[MonitorArea],
    point: Option<Point>,
) -> Frame {
    if monitor_areas
        .iter()
        .any(|monitor| monitor.work_area.contains_frame(frame))
    {
        return frame;
    }

    let target = point
        .and_then(|point| {
            monitor_areas
                .iter()
                .copied()
                .find(|monitor| monitor.bounds.contains_point(point))
                .map(|monitor| monitor.work_area)
        })
        .or_else(|| best_monitor_work_area_for_frame(frame, monitor_areas));

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

fn clamped_frame_to_work_area(frame: Frame, bounds: WorkBounds) -> Frame {
    let (x, y) = clamped_position(Some(bounds), frame.x, frame.y, frame.width, frame.height);
    Frame { x, y, ..frame }
}

fn best_monitor_work_area_for_frame(
    frame: Frame,
    monitor_areas: &[MonitorArea],
) -> Option<WorkBounds> {
    monitor_areas
        .iter()
        .map(|monitor| monitor.work_area)
        .max_by_key(|area| {
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
    const LEFT_MONITOR: MonitorArea = MonitorArea {
        bounds: LEFT_SCREEN,
        work_area: LEFT_SCREEN,
    };
    const RIGHT_MONITOR: MonitorArea = MonitorArea {
        bounds: RIGHT_SCREEN,
        work_area: RIGHT_SCREEN,
    };

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
    fn release_keeps_fully_contained_frame_unchanged() {
        let frame = Frame {
            x: 400,
            y: 500,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_monitor_work_area_near_point(
                frame,
                &[LEFT_MONITOR],
                Some(Point { x: 1600, y: 600 })
            ),
            frame,
        );
    }

    #[test]
    fn release_assigns_cross_monitor_frame_to_cursor_work_area() {
        let frame = Frame {
            x: 1500,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_monitor_work_area_near_point(
                frame,
                &[LEFT_MONITOR, RIGHT_MONITOR],
                Some(Point { x: 2500, y: 500 })
            ),
            Frame { x: 1920, ..frame },
        );
        assert_eq!(
            frame_in_single_monitor_work_area_near_point(
                frame,
                &[LEFT_MONITOR, RIGHT_MONITOR],
                Some(Point { x: 1500, y: 500 })
            ),
            Frame { x: 960, ..frame },
        );
    }

    #[test]
    fn release_assigns_taskbar_point_to_monitor_physical_bounds() {
        let left_work_area = WorkBounds {
            bottom: 1040,
            ..LEFT_SCREEN
        };
        let right_work_area = WorkBounds {
            bottom: 1040,
            ..RIGHT_SCREEN
        };
        let monitor_areas = [
            MonitorArea {
                bounds: LEFT_SCREEN,
                work_area: left_work_area,
            },
            MonitorArea {
                bounds: RIGHT_SCREEN,
                work_area: right_work_area,
            },
        ];
        let frame = Frame {
            x: 1700,
            y: 1050,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_monitor_work_area_near_point(
                frame,
                &monitor_areas,
                Some(Point { x: 2500, y: 1079 })
            ),
            Frame {
                x: 1920,
                y: 860,
                ..frame
            }
        );
    }

    #[test]
    fn release_falls_back_to_largest_overlap_without_cursor() {
        let frame = Frame {
            x: 1700,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(
            frame_in_single_monitor_work_area_near_point(
                frame,
                &[LEFT_MONITOR, RIGHT_MONITOR],
                None
            ),
            Frame { x: 1920, ..frame },
        );
    }
}
