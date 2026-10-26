//! Cubism 2 `.mtn` motion parsing and playback.

use std::sync::Arc;

use super::{Moc2Error, runtime::Moc2Runtime};

/// Motion priority values used by the Cubism 2 framework.
pub mod priority {
    /// No motion is running.
    pub const NONE: i32 = 0;
    /// Idle motion priority.
    pub const IDLE: i32 = 1;
    /// Normal interaction motion priority.
    pub const NORMAL: i32 = 2;
    /// Forced motion priority.
    pub const FORCE: i32 = 3;
}

/// Layout axis animated by a `LAYOUT:` motion channel.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Moc2MotionLayout {
    /// Horizontal position.
    X,
    /// Vertical position.
    Y,
    /// Horizontal anchor position.
    AnchorX,
    /// Vertical anchor position.
    AnchorY,
    /// Horizontal scale.
    ScaleX,
    /// Vertical scale.
    ScaleY,
}

/// What a motion channel drives.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Moc2MotionChannelKind {
    /// A float parameter.
    Parameter,
    /// A `VISIBLE:` parameter read by Cubism 2 pose groups.
    PartsVisible,
    /// A layout value. These channels are parsed but not applied to the model.
    Layout(Moc2MotionLayout),
}

/// One animated channel of a `.mtn` motion.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2MotionChannel {
    id: String,
    kind: Moc2MotionChannelKind,
    values: Vec<f32>,
    fade_in_ms: Option<f32>,
    fade_out_ms: Option<f32>,
}

impl Moc2MotionChannel {
    /// Returns the channel target id, including the `VISIBLE:` or `LAYOUT:`
    /// prefix used by the source motion file.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns what the channel drives.
    pub fn kind(&self) -> Moc2MotionChannelKind {
        self.kind
    }

    /// Returns one value per source frame.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns the per-channel fade-in declared by the motion file.
    pub fn fade_in_ms(&self) -> Option<f32> {
        self.fade_in_ms
    }

    /// Returns the per-channel fade-out declared by the motion file.
    pub fn fade_out_ms(&self) -> Option<f32> {
        self.fade_out_ms
    }

    pub(crate) fn value_at(&self, frame: isize) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        let index = frame.clamp(0, self.values.len() as isize - 1) as usize;
        self.values[index]
    }
}

/// A parsed Cubism 2 `.mtn` motion.
///
/// Motions store one value per channel for every frame at [`fps`](Self::fps).
/// Playback is handled by [`crate::moc2::Moc2Runtime::start_motion`].
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2Motion {
    fps: f32,
    frame_count: usize,
    fade_in_ms: f32,
    fade_out_ms: f32,
    channels: Vec<Moc2MotionChannel>,
}

impl Moc2Motion {
    /// Parses motion data from a UTF-8 byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Moc2Error> {
        let source = std::str::from_utf8(bytes)
            .map_err(|_| Moc2Error::InvalidData("motion file is not valid UTF-8"))?;
        Self::from_text(source)
    }

    /// Parses motion data from text.
    pub fn from_text(source: &str) -> Result<Self, Moc2Error> {
        let mut fps = 30.0f32;
        let mut fade_in_ms = 1000.0f32;
        let mut fade_out_ms = 1000.0f32;
        let mut channel_fade_in = std::collections::HashMap::new();
        let mut channel_fade_out = std::collections::HashMap::new();
        let mut channels: Vec<Moc2MotionChannel> = Vec::new();

        for line in source.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(directive) = line.strip_prefix('$') {
                let Some((key, value)) = directive.split_once('=') else {
                    continue;
                };
                let key = key.trim();
                let value = parse_number(value)?;
                if let Some(id) = key.strip_prefix("fadein:") {
                    channel_fade_in.insert(id.trim().to_owned(), value);
                } else if let Some(id) = key.strip_prefix("fadeout:") {
                    channel_fade_out.insert(id.trim().to_owned(), value);
                } else if key == "fadein" {
                    fade_in_ms = value;
                } else if key == "fadeout" {
                    fade_out_ms = value;
                } else if key == "fps" {
                    fps = value;
                }
                continue;
            }

            let Some((key, values)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if key.is_empty() {
                continue;
            }
            let kind = if let Some(axis) = key.strip_prefix("LAYOUT:") {
                Moc2MotionChannelKind::Layout(match axis.trim() {
                    "ANCHOR_X" => Moc2MotionLayout::AnchorX,
                    "ANCHOR_Y" => Moc2MotionLayout::AnchorY,
                    "SCALE_X" => Moc2MotionLayout::ScaleX,
                    "SCALE_Y" => Moc2MotionLayout::ScaleY,
                    "Y" => Moc2MotionLayout::Y,
                    _ => Moc2MotionLayout::X,
                })
            } else if key.starts_with("VISIBLE:") {
                Moc2MotionChannelKind::PartsVisible
            } else {
                Moc2MotionChannelKind::Parameter
            };
            channels.push(Moc2MotionChannel {
                id: key.to_owned(),
                kind,
                values: parse_values(values)?,
                fade_in_ms: channel_fade_in.get(key).copied(),
                fade_out_ms: channel_fade_out.get(key).copied(),
            });
        }

        if !(fps.is_finite() && fps > 0.0) {
            return Err(Moc2Error::InvalidData("motion fps must be positive"));
        }

        let frame_count = channels
            .iter()
            .map(|channel| channel.values.len())
            .max()
            .unwrap_or(0);

        Ok(Self {
            fps,
            frame_count,
            fade_in_ms,
            fade_out_ms,
            channels,
        })
    }

    /// Returns the frame rate the motion was authored at.
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Returns the number of frames in the longest channel.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Returns the motion duration in seconds.
    pub fn duration_seconds(&self) -> f32 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.frame_count as f32 / self.fps
        }
    }

    /// Returns the fade-in declared by the motion file header.
    pub fn fade_in_ms(&self) -> f32 {
        self.fade_in_ms
    }

    /// Returns the fade-out declared by the motion file header.
    pub fn fade_out_ms(&self) -> f32 {
        self.fade_out_ms
    }

    /// Returns all animated channels.
    pub fn channels(&self) -> &[Moc2MotionChannel] {
        &self.channels
    }
}

fn parse_number(value: &str) -> Result<f32, Moc2Error> {
    value
        .trim()
        .parse::<f32>()
        .map_err(|_| Moc2Error::InvalidData("motion directive value is not a number"))
}

fn parse_values(source: &str) -> Result<Vec<f32>, Moc2Error> {
    let mut values = Vec::new();
    for token in source.split([',', ' ', '\t']) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value = token
            .parse::<f32>()
            .map_err(|_| Moc2Error::InvalidData("motion value is not a number"))?;
        values.push(value);
    }
    Ok(values)
}

/// Cubism 2 fade easing: a half cosine ramp clamped to `0.0..=1.0`.
pub(crate) fn easing_sine(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    0.5 - 0.5 * (value * std::f32::consts::PI).cos()
}

/// One queued motion with its playback state.
#[derive(Debug)]
struct Moc2MotionEntry {
    motion: Arc<Moc2Motion>,
    fade_in_ms: f32,
    fade_out_ms: f32,
    start_time: Option<f32>,
    fade_in_start_time: f32,
    end_time: Option<f32>,
    finished: bool,
    channel_targets: Vec<Option<usize>>,
}

impl Moc2MotionEntry {
    fn new(
        motion: Arc<Moc2Motion>,
        runtime: &mut Moc2Runtime,
        fade_in_ms: f32,
        fade_out_ms: f32,
    ) -> Self {
        let channel_targets = motion
            .channels()
            .iter()
            .map(|channel| match channel.kind() {
                Moc2MotionChannelKind::Parameter | Moc2MotionChannelKind::PartsVisible => {
                    Some(runtime.ensure_parameter_index(channel.id()))
                }
                Moc2MotionChannelKind::Layout(_) => None,
            })
            .collect();

        Self {
            motion,
            fade_in_ms,
            fade_out_ms,
            start_time: None,
            fade_in_start_time: 0.0,
            end_time: None,
            finished: false,
            channel_targets,
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn start_fade_out(&mut self, now: f32, fade_out_ms: f32) {
        let end = now + fade_out_ms.max(0.0) / 1000.0;
        self.end_time = Some(match self.end_time {
            Some(existing) => existing.min(end),
            None => end,
        });
    }

    fn fade_weight(&self, now: f32) -> f32 {
        let fade_in = if self.fade_in_ms <= 0.0 {
            1.0
        } else {
            easing_sine((now - self.fade_in_start_time) * 1000.0 / self.fade_in_ms)
        };
        let fade_out = match self.end_time {
            Some(end) if self.fade_out_ms > 0.0 => {
                easing_sine((end - now) * 1000.0 / self.fade_out_ms)
            }
            _ => 1.0,
        };
        fade_in * fade_out
    }

    fn update(&mut self, runtime: &mut Moc2Runtime, now: f32) -> bool {
        if self.finished {
            return false;
        }

        if self.start_time.is_none() {
            self.start_time = Some(now);
            self.fade_in_start_time = now;
            if self.end_time.is_none() {
                let duration = self.motion.duration_seconds();
                if duration > 0.0 {
                    self.end_time = Some(now + duration);
                }
            }
        }

        let start_time = self.start_time.unwrap_or(now);
        let weight = self.fade_weight(now);
        let frame_position = (now - start_time) * self.motion.fps();
        let current_frame = frame_position.max(0.0).floor();
        let frame_index = current_frame as isize;
        let frame_fraction = frame_position.max(0.0) - current_frame;

        for (channel, target) in self.motion.channels().iter().zip(&self.channel_targets) {
            let Some(parameter_index) = *target else {
                continue;
            };
            match channel.kind() {
                Moc2MotionChannelKind::PartsVisible => {
                    runtime.write_parameter(parameter_index, channel.value_at(frame_index));
                }
                Moc2MotionChannelKind::Parameter => {
                    let (minimum, maximum) = runtime
                        .parameter_range(parameter_index)
                        .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
                    let threshold = 0.4 * (maximum - minimum);
                    let start_value = channel.value_at(frame_index);
                    let end_value = channel.value_at(frame_index + 1);
                    let delta = end_value - start_value;
                    let interpolated = if delta.abs() > threshold {
                        start_value
                    } else {
                        start_value + delta * frame_fraction
                    };
                    let current = runtime
                        .parameter_value_at(parameter_index)
                        .unwrap_or(interpolated);
                    runtime.write_parameter(
                        parameter_index,
                        current + (interpolated - current) * weight,
                    );
                }
                Moc2MotionChannelKind::Layout(_) => {}
            }
        }

        if frame_index >= self.motion.frame_count() as isize {
            self.finished = true;
        }
        if let Some(end) = self.end_time
            && end < now
        {
            self.finished = true;
        }

        true
    }
}

/// Queue of Cubism 2 motions with priority bookkeeping.
#[derive(Debug, Default)]
pub struct Moc2MotionManager {
    entries: Vec<Moc2MotionEntry>,
    current_priority: i32,
    reserve_priority: i32,
}

impl Moc2MotionManager {
    /// Creates an empty motion manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether no motion is currently playing.
    pub fn is_finished(&self) -> bool {
        self.entries.iter().all(Moc2MotionEntry::is_finished)
    }

    /// Returns the number of queued motions, including fading-out motions.
    pub fn active_motions(&self) -> usize {
        self.entries.len()
    }

    /// Returns the priority of the most recently started motion.
    pub fn current_priority(&self) -> i32 {
        self.current_priority
    }

    /// Starts a motion, fading out anything already playing.
    pub(crate) fn start(
        &mut self,
        motion: Arc<Moc2Motion>,
        fade_in_ms: f32,
        fade_out_ms: f32,
        runtime: &mut Moc2Runtime,
        now: f32,
    ) {
        for entry in &mut self.entries {
            entry.start_fade_out(now, entry.fade_out_ms);
        }
        self.entries.push(Moc2MotionEntry::new(
            motion,
            runtime,
            fade_in_ms,
            fade_out_ms,
        ));
    }

    /// Reserves priority for a motion, following Cubism 2 rules.
    fn reserve_motion(&mut self, priority: i32) -> bool {
        if self.reserve_priority >= priority {
            return false;
        }
        if self.current_priority >= priority {
            return false;
        }
        self.reserve_priority = priority;
        true
    }

    /// Starts a motion at a specific priority.
    ///
    /// Returns `false` when a motion with equal or higher priority is running.
    pub(crate) fn start_with_priority(
        &mut self,
        motion: Arc<Moc2Motion>,
        fade_in_ms: f32,
        fade_out_ms: f32,
        priority: i32,
        runtime: &mut Moc2Runtime,
        now: f32,
    ) -> bool {
        if priority == priority::FORCE {
            self.reserve_priority = priority;
        } else if !self.reserve_motion(priority) {
            return false;
        }
        if priority == self.reserve_priority {
            self.reserve_priority = 0;
        }
        self.current_priority = priority;
        self.start(motion, fade_in_ms, fade_out_ms, runtime, now);
        true
    }

    /// Stops every queued motion without fading.
    pub fn stop_all(&mut self) {
        self.entries.clear();
        self.current_priority = 0;
        self.reserve_priority = 0;
    }

    /// Advances and applies all queued motions.
    ///
    /// Returns whether at least one motion produced parameter values.
    pub(crate) fn update(&mut self, runtime: &mut Moc2Runtime, now: f32) -> bool {
        let mut updated = false;
        for entry in &mut self.entries {
            if entry.update(runtime, now) {
                updated = true;
            }
        }
        self.entries.retain(|entry| !entry.is_finished());
        if self.entries.is_empty() {
            self.current_priority = 0;
        }
        updated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# Live2D Animator Motion Data\n\
        $fps=30\n\
        $fadein=0\n\
        $fadeout=1000\n\
        $fadein:PARAM_EYE_L_OPEN=0\n\
        $fadeout:PARAM_EYE_L_OPEN=500\n\
        PARAM_EYE_L_OPEN=1,0.5,0\n\
        PARAM_ANGLE_X=0,1,2,3,\n\
        VISIBLE:PARTS_01=1\n\
        LAYOUT:ANCHOR_X=0.25\n";

    #[test]
    fn motion_parses_headers_and_channels() {
        let motion = Moc2Motion::from_text(SAMPLE).unwrap();

        assert_eq!(motion.fps(), 30.0);
        assert_eq!(motion.frame_count(), 4);
        assert!((motion.duration_seconds() - 4.0 / 30.0).abs() < 1e-6);
        assert_eq!(motion.channels().len(), 4);

        let eye = &motion.channels()[0];
        assert_eq!(eye.kind(), Moc2MotionChannelKind::Parameter);
        assert_eq!(eye.values(), [1.0, 0.5, 0.0]);
        assert_eq!(eye.fade_in_ms(), Some(0.0));
        assert_eq!(eye.fade_out_ms(), Some(500.0));

        assert_eq!(
            motion.channels()[2].kind(),
            Moc2MotionChannelKind::PartsVisible
        );
        assert_eq!(
            motion.channels()[3].kind(),
            Moc2MotionChannelKind::Layout(Moc2MotionLayout::AnchorX)
        );
    }

    #[test]
    fn fading_uses_cosine_easing() {
        assert_eq!(easing_sine(-1.0), 0.0);
        assert_eq!(easing_sine(1.5), 1.0);
        assert!((easing_sine(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn invalid_motion_numbers_are_reported() {
        let error = Moc2Motion::from_text("PARAM_X=1,two\n").unwrap_err();
        assert!(matches!(error, Moc2Error::InvalidData(_)));
    }
}
