const COLLAPSED_WINDOW_WIDTH: f64 = 960.0;
const COLLAPSED_WINDOW_HEIGHT: f64 = 180.0;
const HISTORY_WINDOW_WIDTH: f64 = 960.0;
const HISTORY_WINDOW_HEIGHT: f64 = 430.0;
const MIN_OVERLAY_WIDTH: f64 = 520.0;
const MIN_OVERLAY_HEIGHT: f64 = 128.0;
const SNAP_EDGE_MARGIN_PX: i32 = 42;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlayMode {
    Compact,
    History,
}

impl Default for OverlayMode {
    fn default() -> Self {
        Self::Compact
    }
}

impl OverlayMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "compact" => Ok(Self::Compact),
            "history" => Ok(Self::History),
            _ => Err(format!("unknown overlay mode: {value}")),
        }
    }

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "East" => Ok(Self::East),
            "North" => Ok(Self::North),
            "NorthEast" => Ok(Self::NorthEast),
            "NorthWest" => Ok(Self::NorthWest),
            "South" => Ok(Self::South),
            "SouthEast" => Ok(Self::SouthEast),
            "SouthWest" => Ok(Self::SouthWest),
            "West" => Ok(Self::West),
            _ => Err(format!("unknown resize direction: {value}")),
        }
    }

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

pub fn compact_visible_frame(full_frame: Frame, height: i32) -> Frame {
    Frame {
        y: full_frame.y + full_frame.height - height,
        height,
        ..full_frame
    }
}

pub fn full_position_from_visible_frame(
    mode: OverlayMode,
    full_frame: Frame,
    visible_frame: Frame,
) -> (i32, i32) {
    if mode == OverlayMode::History {
        return (visible_frame.x, visible_frame.y);
    }
    (
        visible_frame.x,
        visible_frame.y - (full_frame.height - visible_frame.height),
    )
}

#[cfg_attr(all(windows, not(test)), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResizePlan {
    pub frame: Frame,
    pub move_before_resize: bool,
}

#[cfg_attr(all(windows, not(test)), allow(dead_code))]
pub fn resize_plan(
    current: Frame,
    target_width: i32,
    target_height: i32,
    bounds: Option<WorkBounds>,
) -> ResizePlan {
    let x = current.x + (current.width - target_width) / 2;
    let y = current.y + current.height - target_height;
    let (x, y) = clamped_position(bounds, x, y, target_width, target_height, false);
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

pub fn snapped_frame_near_point(
    frame: Frame,
    bounds: Option<WorkBounds>,
    snap_point: Option<Point>,
) -> Frame {
    let Some(bounds) = bounds else {
        return frame;
    };
    let (x, y) = clamped_position(
        Some(bounds),
        frame.x,
        frame.y,
        frame.width,
        frame.height,
        true,
    );
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
            false,
        );
        snapped.x = x;
        snapped.y = y;
    }
    snapped
}

pub fn initial_position(
    bounds: WorkBounds,
    width: i32,
    height: i32,
    bottom_margin: i32,
) -> (i32, i32) {
    let x = bounds.left + (bounds.right - bounds.left - width) / 2;
    let y = bounds.bottom - height - bottom_margin;
    clamped_position(Some(bounds), x, y, width, height, false)
}

pub fn combined_bounds(bounds: impl IntoIterator<Item = WorkBounds>) -> Option<WorkBounds> {
    let mut bounds = bounds.into_iter();
    let first = bounds.next()?;
    Some(bounds.fold(first, |combined, item| WorkBounds {
        left: combined.left.min(item.left),
        top: combined.top.min(item.top),
        right: combined.right.max(item.right),
        bottom: combined.bottom.max(item.bottom),
    }))
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
    snap_to_edge: bool,
) -> (i32, i32) {
    let Some(bounds) = bounds else {
        return (x, y);
    };
    let mut x = x;
    let mut y = y;

    if snap_to_edge {
        if (x - bounds.left).abs() <= SNAP_EDGE_MARGIN_PX {
            x = bounds.left;
        } else if (x + width - bounds.right).abs() <= SNAP_EDGE_MARGIN_PX {
            x = bounds.right - width;
        }

        if (y - bounds.top).abs() <= SNAP_EDGE_MARGIN_PX {
            y = bounds.top;
        } else if (y + height - bounds.bottom).abs() <= SNAP_EDGE_MARGIN_PX {
            y = bounds.bottom - height;
        }
    }

    let max_x = (bounds.right - width).max(bounds.left);
    let max_y = (bounds.bottom - height).max(bounds.top);
    (x.clamp(bounds.left, max_x), y.clamp(bounds.top, max_y))
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
    fn compact_snap_uses_visible_frame_and_offsets_full_window() {
        let full_frame = Frame {
            x: 100,
            y: 500,
            width: 960,
            height: 430,
        };
        let visible_frame = compact_visible_frame(full_frame, 180);
        let bounds = WorkBounds {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1080,
        };

        let snapped_visible =
            snapped_frame_near_point(visible_frame, Some(bounds), Some(Point { x: 200, y: 10 }));
        let (_, full_y) =
            full_position_from_visible_frame(OverlayMode::Compact, full_frame, snapped_visible);

        assert_eq!(snapped_visible.y, bounds.top);
        assert_eq!(
            full_y,
            bounds.top - (full_frame.height - visible_frame.height)
        );
    }

    #[test]
    fn snap_uses_outer_desktop_edges_not_internal_monitor_seams() {
        let frame = Frame {
            x: 1920,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(snapped_frame_near_point(frame, Some(DESKTOP), None).x, 1920);
    }

    #[test]
    fn snap_sticks_to_outer_edges() {
        let near_left = Frame {
            x: 20,
            y: 400,
            width: 960,
            height: 180,
        };
        let near_right = Frame {
            x: 2870,
            y: 400,
            width: 960,
            height: 180,
        };

        assert_eq!(
            snapped_frame_near_point(near_left, Some(DESKTOP), None).x,
            0
        );
        assert_eq!(
            snapped_frame_near_point(near_right, Some(DESKTOP), None).x,
            2880
        );
    }

    #[test]
    fn snap_point_sticks_to_both_edges_at_screen_corner() {
        let frame = Frame {
            x: 2800,
            y: 600,
            width: 960,
            height: 180,
        };
        let snapped =
            snapped_frame_near_point(frame, Some(DESKTOP), Some(Point { x: 3838, y: 1078 }));

        assert_eq!(
            snapped,
            Frame {
                x: 2880,
                y: 900,
                ..frame
            },
        );
    }

    #[test]
    fn snap_point_does_not_stick_to_internal_monitor_seam() {
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
    fn combined_bounds_returns_virtual_desktop_bounds() {
        let bounds = combined_bounds([
            WorkBounds {
                left: -1728,
                top: 0,
                right: 0,
                bottom: 1117,
            },
            WorkBounds {
                left: 0,
                top: 0,
                right: 3024,
                bottom: 1890,
            },
        ]);

        assert_eq!(
            bounds,
            Some(WorkBounds {
                left: -1728,
                top: 0,
                right: 3024,
                bottom: 1890,
            }),
        );
    }
}
