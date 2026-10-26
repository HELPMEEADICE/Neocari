//! Cubism 2 deformer evaluation.
//!
//! Deformers form a parent-first hierarchy. Every frame is evaluated top-down:
//! each deformer interpolates its own keyforms, then applies its parent's
//! transform to produce the grid or affine that drawables use.

use std::f32::consts::PI;

use super::scene::{Moc2Affine, Moc2Deformer, Moc2DeformerKind, Moc2PivotSample, Moc2Scene};

/// Runtime state of one deformer.
#[derive(Debug, Clone)]
pub(crate) struct Moc2DeformerState {
    pub(crate) interpolated_points: Vec<f32>,
    pub(crate) transformed_points: Vec<f32>,
    pub(crate) interpolated_affine: Moc2Affine,
    pub(crate) transformed_affine: Option<Moc2Affine>,
    pub(crate) total_opacity: f32,
    pub(crate) total_scale: f32,
    pub(crate) available: bool,
    pub(crate) outside: bool,
}

impl Moc2DeformerState {
    fn new(deformer: &Moc2Deformer) -> Self {
        let grid_len = deformer.kind.grid_point_count() * 2;
        let needs_transform = deformer.target.is_some();
        Self {
            interpolated_points: if grid_len == 0 {
                Vec::new()
            } else {
                vec![0.0; grid_len]
            },
            transformed_points: if needs_transform && grid_len > 0 {
                vec![0.0; grid_len]
            } else {
                Vec::new()
            },
            interpolated_affine: Moc2Affine::identity(),
            transformed_affine: None,
            total_opacity: 1.0,
            total_scale: 1.0,
            available: true,
            outside: false,
        }
    }

    /// Returns whether the deformer can transform geometry this frame.
    pub(crate) fn is_available(&self) -> bool {
        self.available && !self.outside
    }
}

/// Creates one runtime state per scene deformer.
pub(crate) fn create_states(scene: &Moc2Scene) -> Vec<Moc2DeformerState> {
    scene.deformers.iter().map(Moc2DeformerState::new).collect()
}

/// Evaluates every deformer for the current parameter values.
///
/// Returns `None` when the scene is inconsistent (for example when a pivot table
/// references a missing keyform).
pub(crate) fn evaluate(
    scene: &Moc2Scene,
    states: &mut [Moc2DeformerState],
    parameter_values: &[f32],
) -> Option<()> {
    if states.len() != scene.deformers.len() {
        return None;
    }

    for index in 0..scene.deformers.len() {
        let deformer = &scene.deformers[index];
        let sample = deformer.pivots.sample(parameter_values)?;
        let opacity = if deformer.opacities.is_empty() {
            1.0
        } else {
            sample.interpolate_float(&deformer.opacities)?
        };
        let (parents, current) = states.split_at_mut(index);
        let state = &mut current[0];
        state.outside = sample.is_outside();

        match &deformer.kind {
            Moc2DeformerKind::Warp { points, .. } => {
                sample.interpolate_points(points, &mut state.interpolated_points)?;
                match deformer.target {
                    None => {
                        state.total_scale = 1.0;
                        state.total_opacity = opacity;
                        state.available = true;
                    }
                    Some(parent_index) => {
                        let parent_state = parents.get(parent_index)?;
                        let parent = scene.deformers.get(parent_index)?;
                        state.total_scale = parent_state.total_scale;
                        state.total_opacity = parent_state.total_opacity * opacity;
                        if parent_state.is_available() {
                            let interpolated = &state.interpolated_points;
                            let transformed = &mut state.transformed_points;
                            if transformed.len() != interpolated.len() {
                                return None;
                            }
                            let (interpolated, _) = interpolated.as_chunks::<2>();
                            let (transformed, _) = transformed.as_chunks_mut::<2>();
                            for (source, target) in interpolated.iter().zip(transformed.iter_mut())
                            {
                                *target = transform_point(parent, parent_state, *source)?;
                            }
                            state.available = true;
                        } else {
                            state.available = false;
                        }
                    }
                }
            }
            Moc2DeformerKind::Rotation { affines } => {
                let affine = interpolate_affine(&sample, affines)?;
                state.interpolated_affine = affine;
                match deformer.target {
                    None => {
                        state.transformed_affine = None;
                        state.total_scale = affine.scale_x;
                        state.total_opacity = opacity;
                        state.available = true;
                    }
                    Some(parent_index) => {
                        let parent_state = parents.get(parent_index)?;
                        let parent = scene.deformers.get(parent_index)?;
                        let origin = [affine.origin_x, affine.origin_y];
                        let direction = [
                            0.0,
                            match parent.kind {
                                Moc2DeformerKind::Rotation { .. } => -10.0,
                                Moc2DeformerKind::Warp { .. } => -0.1,
                            },
                        ];
                        let transformed_origin = transform_point(parent, parent_state, origin)?;
                        let transformed_direction =
                            direction_on_destination(parent, parent_state, origin, direction)?;
                        let rotation_angle = angle_between(direction, transformed_direction);
                        state.transformed_affine = Some(Moc2Affine {
                            origin_x: transformed_origin[0],
                            origin_y: transformed_origin[1],
                            scale_x: affine.scale_x,
                            scale_y: affine.scale_y,
                            rotation_degrees: affine.rotation_degrees - rotation_angle.to_degrees(),
                            reflect_x: affine.reflect_x,
                            reflect_y: affine.reflect_y,
                        });
                        state.total_scale = parent_state.total_scale * affine.scale_x;
                        state.total_opacity = parent_state.total_opacity * opacity;
                        state.available = parent_state.is_available();
                    }
                }
            }
        }
    }

    Some(())
}

fn interpolate_affine(sample: &Moc2PivotSample, affines: &[Moc2Affine]) -> Option<Moc2Affine> {
    let mut affine = Moc2Affine::identity();
    affine.origin_x = 0.0;
    affine.origin_y = 0.0;
    affine.scale_x = 0.0;
    affine.scale_y = 0.0;
    affine.rotation_degrees = 0.0;

    for (corner, weight) in sample.corners() {
        let keyform = affines.get(corner)?;
        affine.origin_x += keyform.origin_x * weight;
        affine.origin_y += keyform.origin_y * weight;
        affine.scale_x += keyform.scale_x * weight;
        affine.scale_y += keyform.scale_y * weight;
        affine.rotation_degrees += keyform.rotation_degrees * weight;
    }

    let reflect = affines.get(*sample.indices().first()?)?;
    affine.reflect_x = reflect.reflect_x;
    affine.reflect_y = reflect.reflect_y;
    Some(affine)
}

/// Transforms a model-space point through one deformer state.
pub(crate) fn transform_point(
    deformer: &Moc2Deformer,
    state: &Moc2DeformerState,
    point: [f32; 2],
) -> Option<[f32; 2]> {
    match &deformer.kind {
        Moc2DeformerKind::Warp {
            x_cells, y_cells, ..
        } => {
            let grid = if state.transformed_points.is_empty() {
                &state.interpolated_points
            } else {
                &state.transformed_points
            };
            warp_map_point(point, grid, *x_cells, *y_cells)
        }
        Moc2DeformerKind::Rotation { .. } => {
            let affine = state
                .transformed_affine
                .as_ref()
                .unwrap_or(&state.interpolated_affine);
            Some(rotate_point(point, affine, state.total_scale))
        }
    }
}

fn rotate_point(point: [f32; 2], affine: &Moc2Affine, total_scale: f32) -> [f32; 2] {
    let radians = affine.rotation_degrees.to_radians();
    let (sin, cos) = radians.sin_cos();
    let scale_x = total_scale * if affine.reflect_x { -1.0 } else { 1.0 };
    let scale_y = total_scale * if affine.reflect_y { -1.0 } else { 1.0 };
    [
        cos * scale_x * point[0] - sin * scale_y * point[1] + affine.origin_x,
        sin * scale_x * point[0] + cos * scale_y * point[1] + affine.origin_y,
    ]
}

fn direction_on_destination(
    deformer: &Moc2Deformer,
    state: &Moc2DeformerState,
    origin: [f32; 2],
    direction: [f32; 2],
) -> Option<[f32; 2]> {
    let transformed_origin = transform_point(deformer, state, origin)?;
    let mut step = 1.0f32;

    for _ in 0..10 {
        for sign in [1.0f32, -1.0] {
            let test = [
                origin[0] + sign * step * direction[0],
                origin[1] + sign * step * direction[1],
            ];
            let mapped = transform_point(deformer, state, test)?;
            let delta = [
                (mapped[0] - transformed_origin[0]) * sign,
                (mapped[1] - transformed_origin[1]) * sign,
            ];
            if delta[0] != 0.0 || delta[1] != 0.0 {
                return Some(delta);
            }
        }
        step *= 0.1;
    }

    None
}

fn angle_between(from: [f32; 2], to: [f32; 2]) -> f32 {
    let mut difference = from[1].atan2(from[0]) - to[1].atan2(to[0]);
    while difference < -PI {
        difference += 2.0 * PI;
    }
    while difference > PI {
        difference -= 2.0 * PI;
    }
    difference
}

/// Maps a point through a warp grid, including Cubism 2's extrapolation rules
/// for points outside the grid.
pub(crate) fn warp_map_point(
    point: [f32; 2],
    grid: &[f32],
    x_cells: usize,
    y_cells: usize,
) -> Option<[f32; 2]> {
    if x_cells == 0 || y_cells == 0 {
        return None;
    }
    let row = x_cells;
    let col = y_cells;
    let row_stride = row + 1;
    let grid_len = row_stride.checked_mul(col + 1)?.checked_mul(2)?;
    if grid.len() < grid_len {
        return None;
    }

    let at = |x: usize, y: usize| -> [f32; 2] {
        let offset = (x + y * row_stride) * 2;
        [grid[offset], grid[offset + 1]]
    };

    let normalized_x = point[0];
    let normalized_y = point[1];
    let grid_row = normalized_x * row as f32;
    let grid_col = normalized_y * col as f32;

    if grid_row < 0.0 || grid_col < 0.0 || row as f32 <= grid_row || col as f32 <= grid_col {
        let corner_00 = at(0, 0);
        let corner_r0 = at(row, 0);
        let corner_0c = at(0, col);
        let corner_rc = at(row, col);

        let diagonal = [corner_rc[0] - corner_00[0], corner_rc[1] - corner_00[1]];
        let anti_diagonal = [corner_r0[0] - corner_0c[0], corner_r0[1] - corner_0c[1]];
        let scale = [
            (diagonal[0] + anti_diagonal[0]) * 0.5,
            (diagonal[1] + anti_diagonal[1]) * 0.5,
        ];
        let skew = [
            (diagonal[0] - anti_diagonal[0]) * 0.5,
            (diagonal[1] - anti_diagonal[1]) * 0.5,
        ];
        let center = [
            0.25 * (corner_00[0] + corner_r0[0] + corner_0c[0] + corner_rc[0])
                - 0.5 * (scale[0] + skew[0]),
            0.25 * (corner_00[1] + corner_r0[1] + corner_0c[1] + corner_rc[1])
                - 0.5 * (scale[1] + skew[1]),
        ];

        if !(-2.0..3.0).contains(&normalized_x) || !(-2.0..3.0).contains(&normalized_y) {
            return finite([
                center[0] + normalized_x * scale[0] + normalized_y * skew[0],
                center[1] + normalized_x * scale[1] + normalized_y * skew[1],
            ]);
        }

        let cell = if normalized_x <= 0.0 {
            if normalized_y <= 0.0 {
                WarpCell {
                    bary_u: 0.5 * (normalized_x + 2.0),
                    bary_v: 0.5 * (normalized_y + 2.0),
                    point_1: [center[0] - 2.0 * scale[0], center[1] - 2.0 * scale[1]],
                    point_2: [center[0] - 2.0 * skew[0], center[1] - 2.0 * skew[1]],
                    point_3: [
                        center[0] - 2.0 * scale[0] - 2.0 * skew[0],
                        center[1] - 2.0 * scale[1] - 2.0 * skew[1],
                    ],
                    corner: corner_00,
                }
            } else if normalized_y >= 1.0 {
                WarpCell {
                    bary_u: 0.5 * (normalized_x + 2.0),
                    bary_v: 0.5 * (normalized_y - 1.0),
                    point_1: [
                        center[0] - 2.0 * scale[0] + 3.0 * skew[0],
                        center[1] - 2.0 * scale[1] + 3.0 * skew[1],
                    ],
                    point_2: corner_0c,
                    point_3: [
                        center[0] - 2.0 * scale[0] + skew[0],
                        center[1] - 2.0 * scale[1] + skew[1],
                    ],
                    corner: [center[0] + 3.0 * skew[0], center[1] + 3.0 * skew[1]],
                }
            } else {
                let cell_y = clamp_cell(grid_col.floor() as isize, col);
                let fraction_1 = cell_y as f32 / col as f32;
                let fraction_2 = (cell_y + 1) as f32 / col as f32;
                WarpCell {
                    bary_u: 0.5 * (normalized_x + 2.0),
                    bary_v: grid_col - cell_y as f32,
                    point_1: [
                        center[0] - 2.0 * scale[0] + fraction_2 * skew[0],
                        center[1] - 2.0 * scale[1] + fraction_2 * skew[1],
                    ],
                    point_2: at(0, cell_y),
                    point_3: [
                        center[0] - 2.0 * scale[0] + fraction_1 * skew[0],
                        center[1] - 2.0 * scale[1] + fraction_1 * skew[1],
                    ],
                    corner: at(0, cell_y + 1),
                }
            }
        } else if normalized_x >= 1.0 {
            if normalized_y <= 0.0 {
                WarpCell {
                    bary_u: 0.5 * (normalized_x - 1.0),
                    bary_v: 0.5 * (normalized_y + 2.0),
                    point_1: corner_r0,
                    point_2: [
                        center[0] + 3.0 * scale[0] - 2.0 * skew[0],
                        center[1] + 3.0 * scale[1] - 2.0 * skew[1],
                    ],
                    point_3: [
                        center[0] + scale[0] - 2.0 * skew[0],
                        center[1] + scale[1] - 2.0 * skew[1],
                    ],
                    corner: [center[0] + 3.0 * scale[0], center[1] + 3.0 * scale[1]],
                }
            } else if normalized_y >= 1.0 {
                WarpCell {
                    bary_u: 0.5 * (normalized_x - 1.0),
                    bary_v: 0.5 * (normalized_y - 1.0),
                    point_1: [
                        center[0] + scale[0] + 3.0 * skew[0],
                        center[1] + scale[1] + 3.0 * skew[1],
                    ],
                    point_2: [
                        center[0] + 3.0 * scale[0] + skew[0],
                        center[1] + 3.0 * scale[1] + skew[1],
                    ],
                    point_3: corner_rc,
                    corner: [
                        center[0] + 3.0 * scale[0] + 3.0 * skew[0],
                        center[1] + 3.0 * scale[1] + 3.0 * skew[1],
                    ],
                }
            } else {
                let cell_y = clamp_cell(grid_col.floor() as isize, col);
                let fraction_1 = cell_y as f32 / col as f32;
                let fraction_2 = (cell_y + 1) as f32 / col as f32;
                WarpCell {
                    bary_u: 0.5 * (normalized_x - 1.0),
                    bary_v: grid_col - cell_y as f32,
                    point_1: at(row, cell_y + 1),
                    point_2: [
                        center[0] + 3.0 * scale[0] + fraction_1 * skew[0],
                        center[1] + 3.0 * scale[1] + fraction_1 * skew[1],
                    ],
                    point_3: at(row, cell_y),
                    corner: [
                        center[0] + 3.0 * scale[0] + fraction_2 * skew[0],
                        center[1] + 3.0 * scale[1] + fraction_2 * skew[1],
                    ],
                }
            }
        } else if normalized_y <= 0.0 {
            let cell_x = clamp_cell(grid_row.floor() as isize, row);
            let fraction_1 = cell_x as f32 / row as f32;
            let fraction_2 = (cell_x + 1) as f32 / row as f32;
            WarpCell {
                bary_u: grid_row - cell_x as f32,
                bary_v: 0.5 * (normalized_y + 2.0),
                point_1: at(cell_x, 0),
                point_2: [
                    center[0] + fraction_2 * scale[0] - 2.0 * skew[0],
                    center[1] + fraction_2 * scale[1] - 2.0 * skew[1],
                ],
                point_3: [
                    center[0] + fraction_1 * scale[0] - 2.0 * skew[0],
                    center[1] + fraction_1 * scale[1] - 2.0 * skew[1],
                ],
                corner: at(cell_x + 1, 0),
            }
        } else {
            let cell_x = clamp_cell(grid_row.floor() as isize, row);
            let fraction_1 = cell_x as f32 / row as f32;
            let fraction_2 = (cell_x + 1) as f32 / row as f32;
            WarpCell {
                bary_u: grid_row - cell_x as f32,
                bary_v: 0.5 * (normalized_y - 1.0),
                point_1: [
                    center[0] + fraction_1 * scale[0] + 3.0 * skew[0],
                    center[1] + fraction_1 * scale[1] + 3.0 * skew[1],
                ],
                point_2: at(cell_x + 1, col),
                point_3: at(cell_x, col),
                corner: [
                    center[0] + fraction_2 * scale[0] + 3.0 * skew[0],
                    center[1] + fraction_2 * scale[1] + 3.0 * skew[1],
                ],
            }
        };

        return finite(cell.evaluate());
    }

    let cell_x = grid_row.floor() as usize;
    let cell_y = grid_col.floor() as usize;
    let fraction_x = grid_row - cell_x as f32;
    let fraction_y = grid_col - cell_y as f32;

    let corner_00 = at(cell_x, cell_y);
    let corner_10 = at(cell_x + 1, cell_y);
    let corner_01 = at(cell_x, cell_y + 1);
    let corner_11 = at(cell_x + 1, cell_y + 1);

    let output = if fraction_x + fraction_y < 1.0 {
        [
            corner_00[0] * (1.0 - fraction_x - fraction_y)
                + corner_10[0] * fraction_x
                + corner_01[0] * fraction_y,
            corner_00[1] * (1.0 - fraction_x - fraction_y)
                + corner_10[1] * fraction_x
                + corner_01[1] * fraction_y,
        ]
    } else {
        [
            corner_11[0] * (fraction_x - 1.0 + fraction_y)
                + corner_01[0] * (1.0 - fraction_x)
                + corner_10[0] * (1.0 - fraction_y),
            corner_11[1] * (fraction_x - 1.0 + fraction_y)
                + corner_01[1] * (1.0 - fraction_x)
                + corner_10[1] * (1.0 - fraction_y),
        ]
    };

    finite(output)
}

fn finite(value: [f32; 2]) -> Option<[f32; 2]> {
    value
        .iter()
        .all(|component| component.is_finite())
        .then_some(value)
}

fn clamp_cell(cell: isize, count: usize) -> usize {
    if cell == count as isize {
        count.saturating_sub(1)
    } else {
        cell.clamp(0, count as isize - 1) as usize
    }
}

struct WarpCell {
    bary_u: f32,
    bary_v: f32,
    point_1: [f32; 2],
    point_2: [f32; 2],
    point_3: [f32; 2],
    corner: [f32; 2],
}

impl WarpCell {
    fn evaluate(&self) -> [f32; 2] {
        let (u, v) = (self.bary_u, self.bary_v);
        if u + v <= 1.0 {
            [
                self.point_3[0]
                    + (self.point_2[0] - self.point_3[0]) * u
                    + (self.point_1[0] - self.point_3[0]) * v,
                self.point_3[1]
                    + (self.point_2[1] - self.point_3[1]) * u
                    + (self.point_1[1] - self.point_3[1]) * v,
            ]
        } else {
            [
                self.corner[0]
                    + (self.point_1[0] - self.corner[0]) * (1.0 - u)
                    + (self.point_2[0] - self.corner[0]) * (1.0 - v),
                self.corner[1]
                    + (self.point_1[1] - self.corner[1]) * (1.0 - u)
                    + (self.point_2[1] - self.corner[1]) * (1.0 - v),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moc2::scene::{Moc2Parameter, Moc2PivotDimension, Moc2PivotTable};

    fn unit_grid() -> Vec<f32> {
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    }

    #[test]
    fn identity_warp_grid_maps_points_unchanged() {
        let grid = unit_grid();
        let inside = warp_map_point([0.25, 0.75], &grid, 1, 1).unwrap();
        assert!((inside[0] - 0.25).abs() < 1e-5);
        assert!((inside[1] - 0.75).abs() < 1e-5);

        let edge = warp_map_point([1.5, 0.5], &grid, 1, 1).unwrap();
        assert!((edge[0] - 1.5).abs() < 1e-5);
        assert!((edge[1] - 0.5).abs() < 1e-5);

        let far = warp_map_point([7.0, -3.0], &grid, 1, 1).unwrap();
        assert!((far[0] - 7.0).abs() < 1e-5);
        assert!((far[1] + 3.0).abs() < 1e-5);
    }

    #[test]
    fn warp_grid_translates_points() {
        let grid = vec![10.0, 20.0, 11.0, 20.0, 10.0, 21.0, 11.0, 21.0];
        let mapped = warp_map_point([0.5, 0.5], &grid, 1, 1).unwrap();
        assert!((mapped[0] - 10.5).abs() < 1e-5);
        assert!((mapped[1] - 20.5).abs() < 1e-5);
    }

    fn rotation_deformer(
        id: &str,
        target: Option<usize>,
        affine: Moc2Affine,
        parameter_index: usize,
    ) -> Moc2Deformer {
        let pivots = Moc2PivotTable::new(vec![Moc2PivotDimension {
            parameter_index,
            values: vec![0.0],
        }])
        .unwrap();
        Moc2Deformer {
            id: id.to_owned(),
            target,
            pivots,
            opacities: Vec::new(),
            kind: Moc2DeformerKind::Rotation {
                affines: vec![affine],
            },
        }
    }

    #[test]
    fn nested_rotation_composes_parent_angle() {
        let parent = rotation_deformer(
            "parent",
            None,
            Moc2Affine {
                origin_x: 10.0,
                origin_y: 10.0,
                scale_x: 1.0,
                scale_y: 1.0,
                rotation_degrees: 90.0,
                reflect_x: false,
                reflect_y: false,
            },
            0,
        );
        let child = rotation_deformer(
            "child",
            Some(0),
            Moc2Affine {
                origin_x: 0.0,
                origin_y: 0.0,
                scale_x: 1.0,
                scale_y: 1.0,
                rotation_degrees: 0.0,
                reflect_x: false,
                reflect_y: false,
            },
            0,
        );

        let scene = Moc2Scene::from_parts(
            11,
            crate::moc2::scene::Moc2Canvas::new(100, 100),
            vec![Moc2Parameter::new("ParamAngleZ", -100.0, 100.0, 0.0)],
            Vec::new(),
            Vec::new(),
            vec![parent, child],
        );
        let mut states = create_states(&scene);
        evaluate(&scene, &mut states, &[0.0]).unwrap();

        let mapped = transform_point(&scene.deformers[1], &states[1], [1.0, 0.0]).unwrap();
        assert!((mapped[0] - 10.0).abs() < 1e-4, "mapped: {mapped:?}");
        assert!((mapped[1] - 11.0).abs() < 1e-4, "mapped: {mapped:?}");
    }
}
