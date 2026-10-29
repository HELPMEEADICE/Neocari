//! Cubism 2 runtime: parameter state, animation pipeline, and mesh generation.
//!
//! [`Moc2Runtime`] owns everything needed to animate a loaded [`Moc2Scene`]:
//! parameter values, part opacities, deformer state, motions, expressions,
//! physics, pose fading, and the generated drawable meshes. A typical frame is:
//!
//! 1. Set parameters or start motions and expressions.
//! 2. Call [`update`](Self::update) with the frame delta.
//! 3. Upload [`meshes`](Self::meshes) to a renderer.

use std::{collections::BTreeMap, sync::Arc};

use crate::moc3::{Moc3DrawableMesh, Moc3DrawableVertex};

use super::{
    behaviors::{Moc2EyeBlink, Moc2TargetPoint},
    deformers::{self, Moc2DeformerState},
    expression::Moc2ExpressionManager,
    json::{Moc2Expression, Moc2Physics, Moc2Pose},
    layout::Moc2ModelMatrix,
    motion::{Moc2Motion, Moc2MotionManager},
    physics::Moc2PhysicsRuntime,
    pose::Moc2PoseRuntime,
    scene::{Moc2Canvas, Moc2Parameter, Moc2Scene},
};

const PARAMETER_FLOAT_MIN: f32 = -1_000_000.0;
const PARAMETER_FLOAT_MAX: f32 = 1_000_000.0;

/// Options controlling automatic Cubism 2 behaviours.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Moc2RuntimeOptions {
    /// Automatically blink when no motion is currently playing.
    pub auto_blink: bool,
    /// Add the built-in breathing motion to angle and breath parameters.
    pub auto_breath: bool,
}

impl Default for Moc2RuntimeOptions {
    fn default() -> Self {
        Self {
            auto_blink: true,
            auto_breath: true,
        }
    }
}

/// Mutable state for one Cubism 2 model.
#[derive(Debug)]
pub struct Moc2Runtime {
    scene: Moc2Scene,
    parameter_values: Vec<f32>,
    saved_parameter_values: Vec<f32>,
    part_opacities: Vec<f32>,
    deformer_states: Vec<Moc2DeformerState>,
    meshes: Vec<Moc3DrawableMesh>,
    point_scratch: Vec<f32>,
    sorted_indices: Vec<usize>,
    motion_manager: Moc2MotionManager,
    expression_manager: Moc2ExpressionManager,
    physics: Option<Moc2PhysicsRuntime>,
    pose: Option<Moc2PoseRuntime>,
    eye_blink: Moc2EyeBlink,
    drag: Moc2TargetPoint,
    options: Moc2RuntimeOptions,
    time: f32,
    layout: BTreeMap<String, f32>,
}

impl Moc2Runtime {
    /// Creates a runtime for a parsed scene.
    ///
    /// Returns `None` when the scene cannot produce valid drawable meshes.
    pub fn new(scene: Moc2Scene) -> Option<Self> {
        let parameter_values = scene
            .parameters
            .iter()
            .map(|parameter| parameter.clamp(parameter.default_value()))
            .collect::<Vec<_>>();
        let part_opacities = scene
            .parts
            .iter()
            .map(|part| if part.is_visible() { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let deformer_states = deformers::create_states(&scene);
        let meshes = blank_meshes(&scene)?;

        let mut runtime = Self {
            saved_parameter_values: parameter_values.clone(),
            scene,
            parameter_values,
            part_opacities,
            deformer_states,
            meshes,
            point_scratch: Vec::new(),
            sorted_indices: Vec::new(),
            motion_manager: Moc2MotionManager::new(),
            expression_manager: Moc2ExpressionManager::new(),
            physics: None,
            pose: None,
            eye_blink: Moc2EyeBlink::new(),
            drag: Moc2TargetPoint::default(),
            options: Moc2RuntimeOptions::default(),
            time: 0.0,
            layout: BTreeMap::new(),
        };
        runtime.update_meshes()?;
        Some(runtime)
    }

    /// Returns the parsed scene owned by this runtime.
    pub fn scene(&self) -> &Moc2Scene {
        &self.scene
    }

    /// Returns the model canvas.
    pub fn canvas(&self) -> Moc2Canvas {
        self.scene.canvas()
    }

    /// Returns the accumulated animation time in seconds.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Returns the runtime options.
    pub fn options(&self) -> Moc2RuntimeOptions {
        self.options
    }

    /// Replaces the runtime options.
    pub fn set_options(&mut self, options: Moc2RuntimeOptions) {
        self.options = options;
    }

    /// Enables or disables automatic blinking.
    pub fn set_auto_blink(&mut self, enable: bool) {
        self.options.auto_blink = enable;
    }

    /// Enables or disables the built-in breathing motion.
    pub fn set_auto_breath(&mut self, enable: bool) {
        self.options.auto_breath = enable;
    }

    /// Sets the model layout used by [`model_matrix`](Self::model_matrix).
    pub fn set_layout(&mut self, layout: BTreeMap<String, f32>) {
        self.layout = layout;
    }

    /// Returns the model layout.
    pub fn layout(&self) -> &BTreeMap<String, f32> {
        &self.layout
    }

    /// Returns the model-to-clip-space matrix for this canvas and layout.
    pub fn model_matrix(&self) -> crate::core::Matrix44 {
        Moc2ModelMatrix::from_layout(self.scene.canvas(), Some(&self.layout)).matrix()
    }

    /// Returns all parameter ids in model order.
    pub fn parameter_ids(&self) -> impl Iterator<Item = &str> {
        self.scene.parameters.iter().map(Moc2Parameter::id)
    }

    /// Returns the model-order index of a parameter id.
    pub fn parameter_index(&self, id: &str) -> Option<usize> {
        self.scene.parameter_index(id)
    }

    /// Returns metadata for a parameter index.
    pub fn parameter_info(&self, index: usize) -> Option<&Moc2Parameter> {
        self.scene.parameters.get(index)
    }

    /// Returns the current value of a parameter id.
    pub fn parameter_value(&self, id: &str) -> Option<f32> {
        self.parameter_index(id)
            .and_then(|index| self.parameter_value_at(index))
    }

    /// Returns all current parameter values in model order.
    pub fn parameter_values(&self) -> &[f32] {
        &self.parameter_values
    }

    /// Sets a parameter by id, clamped to its declared range.
    ///
    /// The value is also stored as the base value restored at the start of the
    /// next [`update`](Self::update), so application-driven edits persist.
    pub fn set_parameter(&mut self, id: &str, value: f32) -> bool {
        let Some(index) = self.parameter_index(id) else {
            return false;
        };
        self.write_parameter(index, value);
        if let Some(saved) = self.saved_parameter_values.get_mut(index) {
            *saved = self.parameter_values[index];
        }
        true
    }

    /// Blends a parameter towards `value` by `weight`, following Cubism 2's
    /// weighted `setParamFloat` semantics.
    pub fn set_parameter_weighted(&mut self, id: &str, value: f32, weight: f32) -> bool {
        let Some(index) = self.parameter_index(id) else {
            return false;
        };
        self.write_parameter_weighted(index, value, weight);
        if let Some(saved) = self.saved_parameter_values.get_mut(index) {
            *saved = self.parameter_values[index];
        }
        true
    }

    /// Adds `delta * weight` to a parameter.
    pub fn add_parameter(&mut self, id: &str, delta: f32, weight: f32) -> bool {
        let Some(index) = self.parameter_index(id) else {
            return false;
        };
        self.add_parameter_value(index, delta, weight);
        if let Some(saved) = self.saved_parameter_values.get_mut(index) {
            *saved = self.parameter_values[index];
        }
        true
    }

    /// Multiplies a parameter by `factor`, blended by `weight`.
    pub fn multiply_parameter(&mut self, id: &str, factor: f32, weight: f32) -> bool {
        let Some(index) = self.parameter_index(id) else {
            return false;
        };
        self.multiply_parameter_value(index, factor, weight);
        if let Some(saved) = self.saved_parameter_values.get_mut(index) {
            *saved = self.parameter_values[index];
        }
        true
    }

    /// Resets every parameter to its declared default value.
    pub fn reset_parameters(&mut self) {
        for (index, parameter) in self.scene.parameters.iter().enumerate() {
            self.parameter_values[index] = parameter.clamp(parameter.default_value());
        }
        self.save_parameters();
    }

    /// Stores the current parameters as the base restored by [`update`](Self::update).
    pub fn save_parameters(&mut self) {
        self.saved_parameter_values
            .copy_from_slice(&self.parameter_values);
    }

    /// Restores the parameters stored by [`save_parameters`](Self::save_parameters).
    pub fn load_parameters(&mut self) {
        self.parameter_values
            .copy_from_slice(&self.saved_parameter_values);
    }

    /// Returns all part ids in model order.
    pub fn part_ids(&self) -> impl Iterator<Item = &str> {
        self.scene.parts.iter().map(super::scene::Moc2Part::id)
    }

    /// Returns the model-order index of a part id.
    pub fn part_index(&self, id: &str) -> Option<usize> {
        self.scene.part_index(id)
    }

    /// Returns a part opacity by id.
    pub fn part_opacity(&self, id: &str) -> Option<f32> {
        self.part_index(id).map(|index| self.part_opacity_at(index))
    }

    /// Sets a part opacity by id, clamped to `0.0..=1.0`.
    pub fn set_part_opacity(&mut self, id: &str, value: f32) -> bool {
        let Some(index) = self.part_index(id) else {
            return false;
        };
        self.set_part_opacity_by_index(index, value);
        true
    }

    /// Sets the drag target used for head and body angles.
    ///
    /// Values are normalised screen coordinates in `-1.0..=1.0`.
    pub fn set_drag(&mut self, x: f32, y: f32) {
        self.drag.set_point(x, y);
    }

    /// Starts a motion, fading out any motion that is already playing.
    pub fn start_motion(&mut self, motion: Arc<Moc2Motion>, fade_in_ms: f32, fade_out_ms: f32) {
        let now = self.time;
        let mut manager = std::mem::take(&mut self.motion_manager);
        manager.start(motion, fade_in_ms, fade_out_ms, self, now);
        self.motion_manager = manager;
    }

    /// Starts a motion at a Cubism 2 motion priority.
    ///
    /// Returns `false` when a motion of equal or higher priority is playing.
    pub fn start_motion_with_priority(
        &mut self,
        motion: Arc<Moc2Motion>,
        fade_in_ms: f32,
        fade_out_ms: f32,
        priority: i32,
    ) -> bool {
        let now = self.time;
        let mut manager = std::mem::take(&mut self.motion_manager);
        let started =
            manager.start_with_priority(motion, fade_in_ms, fade_out_ms, priority, self, now);
        self.motion_manager = manager;
        started
    }

    /// Stops every motion immediately.
    pub fn stop_all_motions(&mut self) {
        self.motion_manager.stop_all();
    }

    /// Returns whether no motion is currently playing.
    pub fn is_motion_finished(&self) -> bool {
        self.motion_manager.is_finished()
    }

    /// Returns the number of queued motions, including fading-out motions.
    pub fn active_motions(&self) -> usize {
        self.motion_manager.active_motions()
    }

    /// Starts an expression, replacing the active expression.
    pub fn start_expression(&mut self, expression: Arc<Moc2Expression>) {
        let now = self.time;
        let mut manager = std::mem::take(&mut self.expression_manager);
        manager.start(expression, self, now);
        self.expression_manager = manager;
    }

    /// Stops all expressions.
    pub fn stop_expressions(&mut self) {
        self.expression_manager.stop_all();
    }

    /// Returns whether no expression is active.
    pub fn is_expression_finished(&self) -> bool {
        self.expression_manager.is_finished()
    }

    /// Returns the number of active expressions.
    pub fn active_expressions(&self) -> usize {
        self.expression_manager.active_expressions()
    }

    /// Enables physics for this model.
    pub fn set_physics(&mut self, physics: Moc2Physics) {
        self.physics = Some(Moc2PhysicsRuntime::new(&physics, self));
    }

    /// Disables physics.
    pub fn clear_physics(&mut self) {
        self.physics = None;
    }

    /// Returns whether physics is enabled.
    pub fn has_physics(&self) -> bool {
        self.physics.is_some()
    }

    /// Returns the physics runtime when physics is enabled.
    pub fn physics(&self) -> Option<&Moc2PhysicsRuntime> {
        self.physics.as_ref()
    }

    /// Returns the eye blink controller.
    pub fn eye_blink(&self) -> &Moc2EyeBlink {
        &self.eye_blink
    }

    /// Returns the mutable eye blink controller.
    ///
    /// Use this to customise the eye parameter ids or blink timings.
    pub fn eye_blink_mut(&mut self) -> &mut Moc2EyeBlink {
        &mut self.eye_blink
    }

    /// Enables pose groups for this model.
    ///
    /// The initial `VISIBLE:` part state is applied immediately.
    pub fn set_pose(&mut self, pose: Moc2Pose) {
        self.pose = Some(Moc2PoseRuntime::new(&pose, self));
    }

    /// Disables pose groups.
    pub fn clear_pose(&mut self) {
        self.pose = None;
    }

    /// Returns whether pose groups are enabled.
    pub fn has_pose(&self) -> bool {
        self.pose.is_some()
    }

    /// Returns the pose runtime when pose groups are enabled.
    pub fn pose(&self) -> Option<&Moc2PoseRuntime> {
        self.pose.as_ref()
    }

    /// Re-applies the initial state of every pose group.
    pub fn reset_pose(&mut self) {
        if let Some(pose) = self.pose.take() {
            pose.initialize(self);
            self.pose = Some(pose);
        }
    }

    /// Advances the model by `delta_seconds` and rebuilds its meshes.
    ///
    /// The pipeline follows the Cubism 2 framework: restore the saved parameter
    /// state, apply motions, fall back to automatic blinking, apply expressions
    /// and drag/breath offsets, run physics, then fade pose groups.
    pub fn update(&mut self, delta_seconds: f32) -> Option<()> {
        self.update_with_parameters(delta_seconds, |_| {})
    }

    /// Advances animation, applies final parameter overrides, then evaluates
    /// geometry once. The callback runs after physics and pose, allowing lip
    /// sync to override motions without requiring a second mesh evaluation.
    pub fn update_with_parameters(
        &mut self,
        delta_seconds: f32,
        apply_parameters: impl FnOnce(&mut Self),
    ) -> Option<()> {
        let delta_seconds = delta_seconds.max(0.0);
        self.time += delta_seconds;
        let now = self.time;

        self.drag.update(now);

        self.load_parameters();
        let motion_updated = {
            let mut manager = std::mem::take(&mut self.motion_manager);
            let updated = manager.update(self, now);
            self.motion_manager = manager;
            updated
        };
        self.save_parameters();

        if !motion_updated && self.options.auto_blink {
            let mut eye_blink = std::mem::take(&mut self.eye_blink);
            eye_blink.update(self, now);
            self.eye_blink = eye_blink;
        }

        {
            let mut manager = std::mem::take(&mut self.expression_manager);
            manager.update(self, now);
            self.expression_manager = manager;
        }

        self.apply_drag_and_breath(now);

        if let Some(mut physics) = self.physics.take() {
            physics.update(self, now);
            self.physics = Some(physics);
        }

        if let Some(pose) = self.pose.take() {
            pose.update(self, delta_seconds);
            self.pose = Some(pose);
        }

        apply_parameters(self);
        self.update_meshes()
    }

    /// Rebuilds drawable meshes from the current parameter and part state.
    pub fn update_meshes(&mut self) -> Option<()> {
        deformers::evaluate(
            &self.scene,
            &mut self.deformer_states,
            &self.parameter_values,
        )?;

        let max_points = self
            .scene
            .drawables
            .iter()
            .map(|drawable| drawable.point_count)
            .max()
            .unwrap_or(0);
        self.point_scratch.resize(max_points.max(1) * 2, 0.0);

        let Self {
            scene,
            parameter_values,
            part_opacities,
            deformer_states,
            meshes,
            point_scratch,
            sorted_indices,
            ..
        } = self;

        for (drawable_index, drawable) in scene.drawables.iter().enumerate() {
            let sample = drawable.pivots.sample(parameter_values)?;
            let draw_order = sample.interpolate_int(&drawable.draw_orders)?;
            let mut opacity = sample.interpolate_float(&drawable.opacities)?
                * part_opacities
                    .get(drawable.part_index)
                    .copied()
                    .unwrap_or(1.0);
            let point_count = drawable.point_count;
            let scratch = point_scratch.get_mut(..point_count * 2)?;
            sample.interpolate_points(&drawable.positions, scratch)?;

            let mut visible = !sample.is_outside();
            if let Some(target) = drawable.target {
                let state = deformer_states.get(target)?;
                if state.is_available() {
                    let deformer = scene.deformers.get(target)?;
                    for point in scratch.as_chunks_mut::<2>().0 {
                        *point = deformers::transform_point(deformer, state, *point)?;
                    }
                    opacity *= state.total_opacity;
                } else {
                    visible = false;
                }
            }

            let mesh = meshes.get_mut(drawable_index)?;
            for (vertex, position) in mesh
                .vertices_mut()
                .iter_mut()
                .zip(scratch.as_chunks::<2>().0)
            {
                vertex.set_position(*position);
            }
            mesh.set_opacity(if visible { opacity } else { 0.0 });
            mesh.set_draw_order(draw_order as f32);
        }

        sorted_indices.clear();
        sorted_indices.extend(0..meshes.len());
        sorted_indices.sort_by(|left, right| {
            meshes[*left]
                .draw_order()
                .total_cmp(&meshes[*right].draw_order())
                .then_with(|| left.cmp(right))
        });
        for (rank, index) in sorted_indices.iter().enumerate() {
            meshes[*index].set_render_order(rank as i32);
        }

        Some(())
    }

    /// Returns the drawable meshes in model order.
    ///
    /// Renderers should sort them with [`crate::render::common::draw_order_indices`]
    /// or the WGPU backend, which use the resolved render order.
    pub fn meshes(&self) -> &[Moc3DrawableMesh] {
        &self.meshes
    }

    /// Returns the drawable id of the topmost drawable containing a point.
    ///
    /// Coordinates are in model pixels. Returns `None` when no drawable with an
    /// id contains the point.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<&str> {
        let mut indices = (0..self.meshes.len()).collect::<Vec<_>>();
        indices.sort_by_key(|index| self.meshes[*index].render_order());

        for index in indices.into_iter().rev() {
            let mesh = &self.meshes[index];
            if mesh.opacity() <= 0.0 {
                continue;
            }
            if !mesh_contains_point(mesh, x, y) {
                continue;
            }
            let id = self.scene.drawables.get(index)?.id();
            if !id.is_empty() {
                return Some(id);
            }
        }
        None
    }

    pub(crate) fn parameter_value_at(&self, index: usize) -> Option<f32> {
        self.parameter_values.get(index).copied()
    }

    pub(crate) fn parameter_range(&self, index: usize) -> Option<(f32, f32)> {
        let parameter = self.scene.parameters.get(index)?;
        Some((parameter.minimum(), parameter.maximum()))
    }

    pub(crate) fn ensure_parameter_index(&mut self, id: &str) -> usize {
        if let Some(index) = self.scene.parameter_index(id) {
            return index;
        }
        let index = self.scene.parameters.len();
        self.scene.parameters.push(Moc2Parameter::new(
            id,
            PARAMETER_FLOAT_MIN,
            PARAMETER_FLOAT_MAX,
            0.0,
        ));
        self.scene.parameter_index.insert(id.to_owned(), index);
        self.parameter_values.push(0.0);
        self.saved_parameter_values.push(0.0);
        index
    }

    pub(crate) fn write_parameter(&mut self, index: usize, value: f32) {
        let Some(parameter) = self.scene.parameters.get(index) else {
            return;
        };
        let value = parameter.clamp(value);
        if let Some(slot) = self.parameter_values.get_mut(index) {
            *slot = value;
        }
    }

    pub(crate) fn write_parameter_weighted(&mut self, index: usize, value: f32, weight: f32) {
        let current = self.parameter_value_at(index).unwrap_or(0.0);
        self.write_parameter(index, current * (1.0 - weight) + value * weight);
    }

    pub(crate) fn add_parameter_value(&mut self, index: usize, delta: f32, weight: f32) {
        let current = self.parameter_value_at(index).unwrap_or(0.0);
        self.write_parameter(index, current + delta * weight);
    }

    pub(crate) fn multiply_parameter_value(&mut self, index: usize, factor: f32, weight: f32) {
        let current = self.parameter_value_at(index).unwrap_or(0.0);
        self.write_parameter(index, current * (1.0 + (factor - 1.0) * weight));
    }

    pub(crate) fn part_opacity_at(&self, index: usize) -> f32 {
        self.part_opacities.get(index).copied().unwrap_or(0.0)
    }

    pub(crate) fn set_part_opacity_by_index(&mut self, index: usize, value: f32) {
        if let Some(slot) = self.part_opacities.get_mut(index) {
            *slot = value.clamp(0.0, 1.0);
        }
    }

    pub(crate) fn into_scene_and_meshes(self) -> (Moc2Scene, Vec<Moc3DrawableMesh>) {
        (self.scene, self.meshes)
    }

    fn apply_drag_and_breath(&mut self, now: f32) {
        let drag_x = self.drag.x();
        let drag_y = self.drag.y();

        for (id, delta, weight) in [
            ("PARAM_ANGLE_X", drag_x * 30.0, 1.0),
            ("PARAM_ANGLE_Y", drag_y * 30.0, 1.0),
            ("PARAM_ANGLE_Z", drag_x * drag_y * -30.0, 1.0),
            ("PARAM_BODY_ANGLE_X", drag_x * 10.0, 1.0),
            ("PARAM_EYE_BALL_X", drag_x, 1.0),
            ("PARAM_EYE_BALL_Y", drag_y, 1.0),
        ] {
            let index = self.ensure_parameter_index(id);
            self.add_parameter_value(index, delta, weight);
        }

        if !self.options.auto_breath {
            return;
        }

        let t = now * 2.0 * std::f32::consts::PI;
        for (id, delta, weight) in [
            ("PARAM_ANGLE_X", 15.0 * (t / 6.5345).sin(), 0.5),
            ("PARAM_ANGLE_Y", 8.0 * (t / 3.5345).sin(), 0.5),
            ("PARAM_ANGLE_Z", 10.0 * (t / 5.5345).sin(), 0.5),
            ("PARAM_BODY_ANGLE_X", 4.0 * (t / 15.5345).sin(), 0.5),
        ] {
            let index = self.ensure_parameter_index(id);
            self.add_parameter_value(index, delta, weight);
        }
        let breath = self.ensure_parameter_index("PARAM_BREATH");
        self.write_parameter_weighted(breath, 0.5 + 0.5 * (t / 3.2345).sin(), 1.0);
    }
}

fn blank_meshes(scene: &Moc2Scene) -> Option<Vec<Moc3DrawableMesh>> {
    scene
        .drawables
        .iter()
        .map(|drawable| {
            if drawable.uvs.len() != drawable.point_count * 2 {
                return None;
            }
            let vertices = (0..drawable.point_count)
                .map(|index| {
                    Moc3DrawableVertex::new(
                        [0.0, 0.0],
                        [drawable.uvs[index * 2], drawable.uvs[index * 2 + 1]],
                    )
                })
                .collect::<Vec<_>>();
            let masks = drawable
                .clips
                .iter()
                .map(|index| i32::try_from(*index).ok())
                .collect::<Option<Vec<i32>>>()?;
            Some(Moc3DrawableMesh::from_parts(
                drawable.texture,
                drawable.blend_flags,
                0.0,
                0.0,
                vertices,
                drawable.indices.clone(),
                masks,
            ))
        })
        .collect()
}

fn mesh_contains_point(mesh: &Moc3DrawableMesh, x: f32, y: f32) -> bool {
    let vertices = mesh.vertices();
    for triangle in mesh.indices().as_chunks::<3>().0 {
        let (Some(a), Some(b), Some(c)) = (
            vertices.get(triangle[0] as usize),
            vertices.get(triangle[1] as usize),
            vertices.get(triangle[2] as usize),
        ) else {
            continue;
        };
        if point_in_triangle(x, y, a.position(), b.position(), c.position()) {
            return true;
        }
    }
    false
}

fn point_in_triangle(x: f32, y: f32, a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    if x < a[0].min(b[0]).min(c[0])
        || x > a[0].max(b[0]).max(c[0])
        || y < a[1].min(b[1]).min(c[1])
        || y > a[1].max(b[1]).max(c[1])
    {
        return false;
    }

    let dx = x - c[0];
    let dy = y - c[1];
    let dx21 = c[0] - b[0];
    let dy12 = b[1] - c[1];
    let determinant = dy12 * (a[0] - c[0]) + dx21 * (a[1] - c[1]);
    let s = dy12 * dx + dx21 * dy;
    let t = (c[1] - a[1]) * dx + (a[0] - c[0]) * dy;

    if determinant < 0.0 {
        s <= 0.0 && t <= 0.0 && s + t >= determinant
    } else {
        s >= 0.0 && t >= 0.0 && s + t <= determinant
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::moc2::{
        json::{Moc2Expression, Moc2Physics, Moc2Pose},
        motion::Moc2Motion,
        motion_priority::{FORCE, NORMAL},
        scene::{
            Moc2Canvas, Moc2Drawable, Moc2Parameter, Moc2Part, Moc2PivotDimension, Moc2PivotTable,
        },
    };

    fn test_scene() -> Moc2Scene {
        let pivots = Moc2PivotTable::new(vec![Moc2PivotDimension {
            parameter_index: 0,
            values: vec![0.0, 1.0],
        }])
        .unwrap();
        let drawable = Moc2Drawable {
            id: "D_face".to_owned(),
            target: None,
            texture: 0,
            blend_flags: 0,
            culling: true,
            point_count: 3,
            indices: vec![0, 1, 2],
            uvs: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            pivots,
            positions: vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 0.0, 11.0, 0.0, 10.0, 1.0,
            ],
            draw_orders: vec![500, 500],
            opacities: vec![1.0, 1.0],
            clips: Vec::new(),
            part_index: 0,
        };
        let parts = vec![Moc2Part {
            id: "P_face".to_owned(),
            visible: true,
            drawables: vec![0],
        }];
        Moc2Scene::from_parts(
            11,
            Moc2Canvas::new(100, 100),
            vec![Moc2Parameter::new("ParamX", 0.0, 1.0, 0.0)],
            parts,
            vec![drawable],
            Vec::new(),
        )
    }

    fn runtime() -> Moc2Runtime {
        let mut runtime = Moc2Runtime::new(test_scene()).unwrap();
        runtime.set_auto_blink(false);
        runtime.set_auto_breath(false);
        runtime
    }

    #[test]
    fn parameters_deform_drawable_meshes() {
        let mut runtime = runtime();
        runtime.update_meshes().unwrap();
        assert_eq!(runtime.meshes()[0].vertices()[0].position(), [0.0, 0.0]);

        assert!(runtime.set_parameter("ParamX", 1.0));
        runtime.update_meshes().unwrap();
        assert_eq!(runtime.meshes()[0].vertices()[0].position(), [10.0, 0.0]);
        assert_eq!(runtime.meshes()[0].vertices()[2].position(), [10.0, 1.0]);
    }

    #[test]
    fn late_parameter_overrides_match_two_mesh_evaluations() {
        let mut single = runtime();
        let mut reference = runtime();
        let motion = Arc::new(Moc2Motion::from_text("$fps=30\nParamX=0,0.2,0.4,0.6\n").unwrap());
        for runtime in [&mut single, &mut reference] {
            runtime.start_motion(Arc::clone(&motion), 0.0, 0.0);
        }
        for frame in 0..120 {
            let value = (frame as f32 * 0.1).sin().max(0.0);
            reference.update(1.0 / 60.0).unwrap();
            if value > 0.0 {
                reference.set_parameter("ParamX", value);
                reference.update_meshes().unwrap();
            }
            single
                .update_with_parameters(1.0 / 60.0, |runtime| {
                    if value > 0.0 {
                        runtime.set_parameter("ParamX", value);
                    }
                })
                .unwrap();
            assert_eq!(single.meshes(), reference.meshes(), "frame {frame}");
            assert_eq!(single.parameter_values, reference.parameter_values);
            assert_eq!(
                single.saved_parameter_values,
                reference.saved_parameter_values
            );
        }
    }

    #[test]
    fn part_opacity_scales_drawable_opacity() {
        let mut runtime = runtime();
        assert!(runtime.set_part_opacity("P_face", 0.25));
        runtime.update_meshes().unwrap();
        assert!((runtime.meshes()[0].opacity() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn motions_drive_parameters_over_time() {
        let mut runtime = runtime();
        // Frame deltas below the Cubism 2 smoothing threshold are interpolated.
        let motion = Arc::new(Moc2Motion::from_text("$fps=30\nParamX=0,0.2,0.4,0.6\n").unwrap());
        runtime.start_motion(motion, 0.0, 0.0);

        runtime.update(1.0 / 60.0).unwrap();
        assert!((runtime.parameter_value("ParamX").unwrap() - 0.0).abs() < 1e-5);

        runtime.update(1.0 / 60.0).unwrap();
        let value = runtime.parameter_value("ParamX").unwrap();
        assert!((value - 0.1).abs() < 1e-5, "value: {value}");

        runtime.update(1.0 / 60.0).unwrap();
        let value = runtime.parameter_value("ParamX").unwrap();
        assert!((value - 0.2).abs() < 1e-5, "value: {value}");
    }

    #[test]
    fn expressions_modify_parameters_with_blend_modes() {
        let mut runtime = runtime();
        runtime.set_parameter("ParamX", 0.5);
        let expression = Arc::new(
            Moc2Expression::from_json_str(
                r#"{"fade_in": 500, "params": [{"id": "ParamX", "val": 1, "calc": "set"}]}"#,
            )
            .unwrap(),
        );
        runtime.start_expression(expression);
        runtime.update(0.0).unwrap();
        // The expression is still fading in.
        assert!((runtime.parameter_value("ParamX").unwrap() - 0.5).abs() < 1e-5);
        runtime.update(0.5).unwrap();
        assert!((runtime.parameter_value("ParamX").unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pose_visibility_controls_part_opacity() {
        let mut runtime = runtime();
        let pose = Moc2Pose::from_json_str(r#"{"parts_visible": [{"group": [{"id": "P_face"}]}]}"#)
            .unwrap();
        runtime.set_pose(pose);

        // `set_pose` applies the initial `VISIBLE:` state.
        assert_eq!(runtime.part_opacity("P_face"), Some(0.0));
        assert_eq!(runtime.parameter_value("VISIBLE:P_face"), Some(0.0));

        // A group without a visible member falls back to its first member, which
        // matches the Cubism 2 pose implementation.
        runtime.update(0.0).unwrap();
        assert_eq!(runtime.part_opacity("P_face"), Some(1.0));

        runtime.set_parameter("VISIBLE:P_face", 1.0);
        runtime.update(0.1).unwrap();
        assert!(runtime.part_opacity("P_face").unwrap() > 0.0);
    }

    #[test]
    fn hit_test_uses_triangles() {
        let mut runtime = runtime();
        runtime.set_parameter("ParamX", 1.0);
        runtime.update_meshes().unwrap();

        assert_eq!(runtime.hit_test(10.2, 0.2), Some("D_face"));
        assert_eq!(runtime.hit_test(-5.0, 0.2), None);
    }

    fn multi_part_scene() -> Moc2Scene {
        Moc2Scene::from_parts(
            11,
            Moc2Canvas::new(100, 100),
            vec![Moc2Parameter::new("ParamX", 0.0, 1.0, 0.0)],
            vec![
                Moc2Part {
                    id: "P_A".to_owned(),
                    visible: true,
                    drawables: Vec::new(),
                },
                Moc2Part {
                    id: "P_B".to_owned(),
                    visible: true,
                    drawables: Vec::new(),
                },
            ],
            Vec::new(),
            Vec::new(),
        )
    }

    #[test]
    fn pose_groups_switch_visible_parts() {
        let mut runtime = Moc2Runtime::new(multi_part_scene()).unwrap();
        runtime.set_auto_blink(false);
        runtime.set_auto_breath(false);
        let pose = Moc2Pose::from_json_str(
            r#"{"parts_visible": [{"group": [{"id": "P_A"}, {"id": "P_B"}]}]}"#,
        )
        .unwrap();
        runtime.set_pose(pose);

        assert!(runtime.set_parameter("VISIBLE:P_A", 1.0));
        runtime.update(0.5).unwrap();
        assert!((runtime.part_opacity("P_A").unwrap() - 1.0).abs() < 1e-6);
        assert!(runtime.part_opacity("P_B").unwrap() < 0.2);

        assert!(runtime.set_parameter("VISIBLE:P_A", 0.0));
        assert!(runtime.set_parameter("VISIBLE:P_B", 1.0));
        runtime.update(0.5).unwrap();
        assert!(runtime.part_opacity("P_A").unwrap() < 0.2);
        assert!((runtime.part_opacity("P_B").unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn physics_chains_write_target_parameters() {
        let mut runtime = runtime();
        let physics = Moc2Physics::from_json_str(
            r#"{"physics_hair": [{
                "setup": { "length": 0.2, "regist": 1, "mass": 0.3 },
                "src": [{ "id": "ParamX", "ptype": "x", "scale": 1, "weight": 1 }],
                "targets": [{ "id": "ParamOut", "ptype": "angle", "scale": 1, "weight": 1 }]
            }]}"#,
        )
        .unwrap();
        runtime.set_physics(physics);
        assert!(
            runtime
                .physics()
                .is_some_and(|physics| physics.hair_count() == 1)
        );

        assert!(runtime.set_parameter("ParamX", 1.0));
        runtime.update(1.0 / 60.0).unwrap();
        for _ in 0..20 {
            runtime.update(1.0 / 60.0).unwrap();
        }

        let value = runtime
            .parameter_value("ParamOut")
            .expect("target parameter");
        assert!(value.is_finite());
        assert!(value.abs() > 1e-3, "physics should tilt the chain: {value}");
    }

    #[test]
    fn motion_priorities_block_lower_or_equal_motions() {
        let mut runtime = runtime();
        let first = Arc::new(Moc2Motion::from_text("ParamX=0,0.2,0.4\n").unwrap());
        let second = Arc::new(Moc2Motion::from_text("ParamX=0.4,0.2,0\n").unwrap());

        assert!(runtime.start_motion_with_priority(Arc::clone(&first), 0.0, 0.0, NORMAL));
        assert!(!runtime.start_motion_with_priority(Arc::clone(&second), 0.0, 0.0, NORMAL));
        assert!(runtime.start_motion_with_priority(Arc::clone(&second), 0.0, 0.0, FORCE));
    }

    #[test]
    fn automatic_blinking_closes_the_eyes() {
        let mut runtime = runtime();
        runtime.set_auto_breath(false);
        runtime.set_auto_blink(true);

        let mut saw_closed_eye = false;
        for _ in 0..600 {
            runtime.update(1.0 / 60.0).unwrap();
            if runtime
                .parameter_value("PARAM_EYE_L_OPEN")
                .is_some_and(|value| value < 0.5)
            {
                saw_closed_eye = true;
                break;
            }
        }
        assert!(saw_closed_eye);
    }
}
