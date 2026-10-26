//! Cubism 2 hair physics simulation.
//!
//! Each `physics_hair` entry is a two-point pendulum. Input parameters move or
//! rotate the pivot point, the chain is integrated every frame, and output
//! parameters receive the chain angle or angular velocity.

use super::{
    json::{Moc2Physics, Moc2PhysicsSourceKind, Moc2PhysicsTargetKind},
    runtime::Moc2Runtime,
};

/// A two-point physics chain state.
#[derive(Debug, Clone, Default)]
struct PhysicsPoint {
    mass: f32,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    ax: f32,
    ay: f32,
    fx: f32,
    fy: f32,
    last_x: f32,
    last_y: f32,
    last_vx: f32,
    last_vy: f32,
}

impl PhysicsPoint {
    fn reset_last(&mut self) {
        self.last_x = self.x;
        self.last_y = self.y;
        self.last_vx = self.vx;
        self.last_vy = self.vy;
    }
}

#[derive(Debug, Clone)]
struct PhysicsSource {
    parameter_index: usize,
    kind: Moc2PhysicsSourceKind,
    scale: f32,
    weight: f32,
}

#[derive(Debug, Clone)]
struct PhysicsTarget {
    parameter_index: usize,
    kind: Moc2PhysicsTargetKind,
    scale: f32,
    weight: f32,
}

#[derive(Debug, Clone)]
struct PhysicsHairState {
    length: f32,
    resistance: f32,
    mass: f32,
    sources: Vec<PhysicsSource>,
    targets: Vec<PhysicsTarget>,
    point_1: PhysicsPoint,
    point_2: PhysicsPoint,
    gravity_angle_degrees: f32,
    angle_degrees: f32,
    previous_angle_degrees: f32,
    angle_velocity: f32,
    initial_time: Option<f32>,
    last_time: f32,
}

impl PhysicsHairState {
    fn new(hair: &super::json::Moc2PhysicsHair, runtime: &mut Moc2Runtime) -> PhysicsHairState {
        let sources = hair
            .sources()
            .iter()
            .map(|source| PhysicsSource {
                parameter_index: runtime.ensure_parameter_index(source.id()),
                kind: source.kind(),
                scale: source.scale(),
                weight: source.weight(),
            })
            .collect();
        let targets = hair
            .targets()
            .iter()
            .map(|target| PhysicsTarget {
                parameter_index: runtime.ensure_parameter_index(target.id()),
                kind: target.kind(),
                scale: target.scale(),
                weight: target.weight(),
            })
            .collect();

        let mut state = PhysicsHairState {
            length: hair.length(),
            resistance: hair.resistance(),
            mass: hair.mass(),
            sources,
            targets,
            point_1: PhysicsPoint::default(),
            point_2: PhysicsPoint::default(),
            gravity_angle_degrees: 0.0,
            angle_degrees: 0.0,
            previous_angle_degrees: 0.0,
            angle_velocity: 0.0,
            initial_time: None,
            last_time: 0.0,
        };
        state.setup();
        state
    }

    fn setup(&mut self) {
        self.previous_angle_degrees = self.current_angle();
        self.point_2.reset_last();
        self.point_1.mass = self.mass;
        self.point_2.mass = self.mass;
        self.point_2.y = self.length;
        self.previous_angle_degrees = self.current_angle();
        self.point_2.reset_last();
    }

    /// Returns the chain angle in degrees.
    fn current_angle(&self) -> f32 {
        let dx = self.point_1.x - self.point_2.x;
        let dy = self.point_1.y - self.point_2.y;
        -dx.atan2(-dy).to_degrees()
    }

    fn update_sources(&mut self, runtime: &Moc2Runtime) {
        for source in self.sources.iter().rev() {
            let value = source.scale
                * runtime
                    .parameter_value_at(source.parameter_index)
                    .unwrap_or(0.0);
            match source.kind {
                Moc2PhysicsSourceKind::X => {
                    self.point_1.x += (value - self.point_1.x) * source.weight;
                }
                Moc2PhysicsSourceKind::Y => {
                    self.point_1.y += (value - self.point_1.y) * source.weight;
                }
                Moc2PhysicsSourceKind::Angle => {
                    self.gravity_angle_degrees +=
                        (value - self.gravity_angle_degrees) * source.weight;
                }
            }
        }
    }

    fn integrate(&mut self, delta_seconds: f32) {
        let delta_seconds = delta_seconds.max(0.033);
        let inverse_delta = 1.0 / delta_seconds;

        self.point_1.vx = (self.point_1.x - self.point_1.last_x) * inverse_delta;
        self.point_1.vy = (self.point_1.y - self.point_1.last_y) * inverse_delta;
        self.point_1.ax = (self.point_1.vx - self.point_1.last_vx) * inverse_delta;
        self.point_1.ay = (self.point_1.vy - self.point_1.last_vy) * inverse_delta;
        self.point_1.fx = self.point_1.ax * self.point_1.mass;
        self.point_1.fy = self.point_1.ay * self.point_1.mass;
        self.point_1.reset_last();

        let angle = -(self.point_1.y - self.point_2.y).atan2(self.point_1.x - self.point_2.x);
        let (sin, cos) = angle.sin_cos();
        let gravity = 9.8 * self.point_2.mass;
        let gravity_angle = self.gravity_angle_degrees.to_radians();
        let gravity_force = gravity * (angle - gravity_angle).cos();

        self.point_2.fx =
            gravity_force * sin - self.point_1.fx * sin * sin - self.point_2.vx * self.resistance;
        self.point_2.fy =
            gravity_force * cos - self.point_1.fy * sin * cos - self.point_2.vy * self.resistance;
        self.point_2.ax = divide_or_zero(self.point_2.fx, self.point_2.mass);
        self.point_2.ay = divide_or_zero(self.point_2.fy, self.point_2.mass);
        self.point_2.vx += self.point_2.ax * delta_seconds;
        self.point_2.vy += self.point_2.ay * delta_seconds;
        self.point_2.x += self.point_2.vx * delta_seconds;
        self.point_2.y += self.point_2.vy * delta_seconds;

        let dx = self.point_1.x - self.point_2.x;
        let dy = self.point_1.y - self.point_2.y;
        let distance = (dx * dx + dy * dy).sqrt();
        if distance > 0.0 {
            self.point_2.x =
                self.point_1.x + self.length * (self.point_2.x - self.point_1.x) / distance;
            self.point_2.y =
                self.point_1.y + self.length * (self.point_2.y - self.point_1.y) / distance;
        }

        self.point_2.vx = (self.point_2.x - self.point_2.last_x) * inverse_delta;
        self.point_2.vy = (self.point_2.y - self.point_2.last_y) * inverse_delta;
        self.point_2.reset_last();
    }

    fn apply_targets(&self, runtime: &mut Moc2Runtime) {
        for target in self.targets.iter().rev() {
            let value = match target.kind {
                Moc2PhysicsTargetKind::Angle => target.scale * self.angle_degrees,
                Moc2PhysicsTargetKind::AngleVelocity => target.scale * self.angle_velocity,
            };
            runtime.write_parameter_weighted(target.parameter_index, value, target.weight);
        }
    }
}

fn divide_or_zero(numerator: f32, denominator: f32) -> f32 {
    if denominator.abs() > f32::EPSILON {
        numerator / denominator
    } else {
        0.0
    }
}

/// Runtime state for the physics chains of one model.
#[derive(Debug, Clone)]
pub struct Moc2PhysicsRuntime {
    hairs: Vec<PhysicsHairState>,
}

impl Moc2PhysicsRuntime {
    pub(crate) fn new(physics: &Moc2Physics, runtime: &mut Moc2Runtime) -> Self {
        Self {
            hairs: physics
                .hairs()
                .iter()
                .map(|hair| PhysicsHairState::new(hair, runtime))
                .collect(),
        }
    }

    /// Returns the number of simulated chains.
    pub fn hair_count(&self) -> usize {
        self.hairs.len()
    }

    pub(crate) fn update(&mut self, runtime: &mut Moc2Runtime, now: f32) {
        for hair in &mut self.hairs {
            if hair.initial_time.is_none() {
                hair.initial_time = Some(now);
                hair.last_time = now;
                let dx = hair.point_1.x - hair.point_2.x;
                let dy = hair.point_1.y - hair.point_2.y;
                hair.length = (dx * dx + dy * dy).sqrt();
                continue;
            }

            let delta_seconds = now - hair.last_time;
            if delta_seconds != 0.0 {
                hair.update_sources(runtime);
                hair.integrate(delta_seconds);
                hair.angle_degrees = hair.current_angle();
                hair.angle_velocity =
                    (hair.angle_degrees - hair.previous_angle_degrees) / delta_seconds;
                hair.previous_angle_degrees = hair.angle_degrees;
            }

            hair.apply_targets(runtime);
            hair.last_time = now;
        }
    }
}
