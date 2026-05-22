const COLLAPSED_WINDOW_WIDTH: f64 = 960.0;
const COLLAPSED_WINDOW_HEIGHT: f64 = 180.0;
const HISTORY_WINDOW_WIDTH: f64 = 960.0;
const HISTORY_WINDOW_HEIGHT: f64 = 430.0;
const SNAP_EDGE_MARGIN_PX: i32 = 42;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlayMode {
    Compact,
    History,
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
