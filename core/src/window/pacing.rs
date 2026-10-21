use crate::time::Duration;

/// How a window paces its redraws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pacing {
    /// Each redraw waits for the compositor to ask for the next frame (the
    /// frame callback, where the platform has one, e.g. Wayland).
    #[default]
    Compositor,

    /// The window does not wait for the compositor: it redraws at most once
    /// per interval, the monitor's refresh interval when `None` (no limit
    /// where the monitor's rate is unknown).
    ///
    /// A compositor that sends its frame callback only once it has shown
    /// the previous frame holds an app whose CPU and GPU work together take
    /// longer than a refresh to every other refresh, or worse.
    Limited(Option<Duration>),

    /// The window does not wait for the compositor and redraws as soon as
    /// it is asked to.
    Unlimited,
}
