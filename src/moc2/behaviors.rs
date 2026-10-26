//! Automatic Cubism 2 behaviours: eye blinking and drag smoothing.

use super::runtime::Moc2Runtime;

const DEFAULT_BLINK_INTERVAL_MS: f32 = 4000.0;
const DEFAULT_CLOSING_MS: f32 = 100.0;
const DEFAULT_CLOSED_MS: f32 = 50.0;
const DEFAULT_OPENING_MS: f32 = 150.0;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum EyeState {
    First,
    Interval,
    Closing,
    Closed,
    Opening,
}

/// Cubism 2 automatic eye blinking.
///
/// The blink interval uses a small deterministic pseudo-random generator so
/// playback is reproducible across runs.
#[derive(Debug, Clone)]
pub struct Moc2EyeBlink {
    state: EyeState,
    state_start_time_ms: f32,
    next_blink_time_ms: f32,
    interval_ms: f32,
    closing_ms: f32,
    closed_ms: f32,
    opening_ms: f32,
    close_if_zero: bool,
    left_id: String,
    right_id: String,
    rng_state: u32,
}

impl Default for Moc2EyeBlink {
    fn default() -> Self {
        Self {
            state: EyeState::First,
            state_start_time_ms: 0.0,
            next_blink_time_ms: 0.0,
            interval_ms: DEFAULT_BLINK_INTERVAL_MS,
            closing_ms: DEFAULT_CLOSING_MS,
            closed_ms: DEFAULT_CLOSED_MS,
            opening_ms: DEFAULT_OPENING_MS,
            close_if_zero: true,
            left_id: "PARAM_EYE_L_OPEN".to_owned(),
            right_id: "PARAM_EYE_R_OPEN".to_owned(),
            rng_state: 0x9e37_79b9,
        }
    }
}

impl Moc2EyeBlink {
    /// Creates an eye blink controller with the default Cubism 2 timings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Changes the parameter ids driven by the blink controller.
    pub fn set_eye_parameter_ids(&mut self, left: impl Into<String>, right: impl Into<String>) {
        self.left_id = left.into();
        self.right_id = right.into();
    }

    /// Controls whether a closed eye is `0.0` or `-1.0`.
    pub fn set_close_if_zero(&mut self, close_if_zero: bool) {
        self.close_if_zero = close_if_zero;
    }

    /// Returns whether both eyes are currently fully open.
    pub fn is_open(&self) -> bool {
        matches!(self.state, EyeState::First | EyeState::Interval)
    }

    fn random_fraction(&mut self) -> f32 {
        let mut value = self.rng_state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.rng_state = value;
        (value >> 8) as f32 / (1u32 << 24) as f32
    }

    fn next_blink_time(&mut self, now_ms: f32) -> f32 {
        now_ms + self.random_fraction() * (2.0 * self.interval_ms - 1.0)
    }

    pub(crate) fn update(&mut self, runtime: &mut Moc2Runtime, now_seconds: f32) {
        let now_ms = now_seconds * 1000.0;
        let value = match self.state {
            EyeState::First => {
                self.state = EyeState::Interval;
                self.next_blink_time_ms = self.next_blink_time(now_ms);
                1.0
            }
            EyeState::Interval => {
                if self.next_blink_time_ms < now_ms {
                    self.state = EyeState::Closing;
                    self.state_start_time_ms = now_ms;
                }
                1.0
            }
            EyeState::Closing => {
                let mut t = (now_ms - self.state_start_time_ms) / self.closing_ms;
                if t >= 1.0 {
                    t = 1.0;
                    self.state = EyeState::Closed;
                    self.state_start_time_ms = now_ms;
                }
                1.0 - t
            }
            EyeState::Closed => {
                let t = (now_ms - self.state_start_time_ms) / self.closed_ms;
                if t >= 1.0 {
                    self.state = EyeState::Opening;
                    self.state_start_time_ms = now_ms;
                }
                0.0
            }
            EyeState::Opening => {
                let mut t = (now_ms - self.state_start_time_ms) / self.opening_ms;
                if t >= 1.0 {
                    t = 1.0;
                    self.state = EyeState::Interval;
                    self.next_blink_time_ms = self.next_blink_time(now_ms);
                }
                t
            }
        };

        let value = if self.close_if_zero { value } else { -value };
        let left = runtime.ensure_parameter_index(&self.left_id);
        let right = runtime.ensure_parameter_index(&self.right_id);
        runtime.write_parameter_weighted(left, value, 1.0);
        runtime.write_parameter_weighted(right, value, 1.0);
    }
}

const FRAME_RATE: f32 = 30.0;
const TIME_TO_MAX_SPEED: f32 = 0.15;
const FACE_PARAM_MAX_V: f32 = 40.0 / 7.5;
const MAX_V: f32 = FACE_PARAM_MAX_V / FRAME_RATE;
const FRAME_TO_MAX_SPEED: f32 = TIME_TO_MAX_SPEED * FRAME_RATE;
const EPSILON: f32 = 0.01;

/// Smoothed drag target used for head and body angles.
#[derive(Debug, Clone, Default)]
pub(crate) struct Moc2TargetPoint {
    target_x: f32,
    target_y: f32,
    x: f32,
    y: f32,
    velocity_x: f32,
    velocity_y: f32,
    last_time: Option<f32>,
}

impl Moc2TargetPoint {
    pub(crate) fn set_point(&mut self, x: f32, y: f32) {
        self.target_x = x;
        self.target_y = y;
    }

    pub(crate) fn x(&self) -> f32 {
        self.x
    }

    pub(crate) fn y(&self) -> f32 {
        self.y
    }

    pub(crate) fn update(&mut self, now: f32) {
        let Some(last_time) = self.last_time else {
            self.last_time = Some(now);
            return;
        };
        let delta_time_weight = (now - last_time) * FRAME_RATE;
        self.last_time = Some(now);

        let max_acceleration = delta_time_weight * MAX_V / FRAME_TO_MAX_SPEED;
        let dx = self.target_x - self.x;
        let dy = self.target_y - self.y;
        if dx.abs() <= EPSILON && dy.abs() <= EPSILON {
            return;
        }

        let distance = (dx * dx + dy * dy).sqrt();
        let velocity_x = MAX_V * dx / distance;
        let velocity_y = MAX_V * dy / distance;
        let mut acceleration_x = velocity_x - self.velocity_x;
        let mut acceleration_y = velocity_y - self.velocity_y;
        let acceleration =
            (acceleration_x * acceleration_x + acceleration_y * acceleration_y).sqrt();
        if acceleration > max_acceleration && acceleration > 0.0 {
            acceleration_x = acceleration_x * max_acceleration / acceleration;
            acceleration_y = acceleration_y * max_acceleration / acceleration;
        }
        self.velocity_x += acceleration_x;
        self.velocity_y += acceleration_y;

        let max_velocity = 0.5
            * ((max_acceleration * max_acceleration + 16.0 * max_acceleration * distance
                - 8.0 * max_acceleration * distance)
                .sqrt()
                - max_acceleration);
        let current_velocity =
            (self.velocity_x * self.velocity_x + self.velocity_y * self.velocity_y).sqrt();
        if current_velocity > max_velocity && current_velocity > 0.0 {
            self.velocity_x = self.velocity_x * max_velocity / current_velocity;
            self.velocity_y = self.velocity_y * max_velocity / current_velocity;
        }

        if self.velocity_x * dx > 0.0 && self.velocity_x.abs() > dx.abs() {
            self.velocity_x = dx;
        }
        if self.velocity_y * dy > 0.0 && self.velocity_y.abs() > dy.abs() {
            self.velocity_y = dy;
        }

        self.x += self.velocity_x;
        self.y += self.velocity_y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_point_smooths_towards_the_drag_position() {
        let mut point = Moc2TargetPoint::default();
        point.set_point(1.0, 0.0);
        point.update(0.0);
        assert_eq!(point.x(), 0.0);

        for step in 1..=60 {
            point.update(step as f32 / 60.0);
        }
        assert!(point.x() > 0.0);
        assert!(point.x() <= 1.0);
    }
}
