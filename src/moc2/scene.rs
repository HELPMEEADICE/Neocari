//! Typed Cubism 2 model data and parameter pivot evaluation.
//!
//! A `.moc` file stores every deformer and drawable as a table of keyforms that
//! are selected by a set of parameter values. [`Moc2PivotTable`] reproduces the
//! Cubism 2 pivot selection and multi-linear blending rules so callers can
//! evaluate those keyforms for the current parameter state.

use std::collections::HashMap;

/// Tolerance used by the Cubism 2 pivot lookup (the `GOSA` constant).
const GOSA: f32 = 0.0001;
/// Highest number of simultaneously interpolated parameters in Cubism 2.
pub(crate) const MAX_PIVOT_DIMENSIONS: usize = 6;
/// Number of corners a fully interpolated pivot table can produce.
const MAX_PIVOT_CORNERS: usize = 1 << MAX_PIVOT_DIMENSIONS;

/// Canvas size declared by a `.moc` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Moc2Canvas {
    width: u32,
    height: u32,
}

impl Moc2Canvas {
    /// Creates a canvas description from the values stored in the model.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns the canvas width in model pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the canvas height in model pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// One float parameter declared by a Cubism 2 model.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2Parameter {
    pub(crate) id: String,
    pub(crate) minimum: f32,
    pub(crate) maximum: f32,
    pub(crate) default: f32,
}

impl Moc2Parameter {
    /// Creates a parameter definition.
    pub(crate) fn new(id: impl Into<String>, minimum: f32, maximum: f32, default: f32) -> Self {
        let (minimum, maximum) = if minimum <= maximum {
            (minimum, maximum)
        } else {
            (maximum, minimum)
        };
        Self {
            id: id.into(),
            minimum,
            maximum,
            default,
        }
    }

    /// Returns the parameter id, such as `PARAM_ANGLE_X`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the declared minimum value.
    pub fn minimum(&self) -> f32 {
        self.minimum
    }

    /// Returns the declared maximum value.
    pub fn maximum(&self) -> f32 {
        self.maximum
    }

    /// Returns the declared default value.
    pub fn default_value(&self) -> f32 {
        self.default
    }

    /// Clamps a value to the declared parameter range.
    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.minimum, self.maximum)
    }
}

/// One part of a Cubism 2 model.
///
/// Parts group drawables, own an initial visibility flag, and are the unit used
/// by `pose.json` fading.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2Part {
    pub(crate) id: String,
    pub(crate) visible: bool,
    pub(crate) drawables: Vec<usize>,
}

impl Moc2Part {
    /// Returns the part id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns whether the part is visible in the model's initial state.
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Returns model-order drawable indices owned by this part.
    pub fn drawables(&self) -> &[usize] {
        &self.drawables
    }
}

/// One affine keyform of a Cubism 2 rotation deformer.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Moc2Affine {
    pub(crate) origin_x: f32,
    pub(crate) origin_y: f32,
    pub(crate) scale_x: f32,
    pub(crate) scale_y: f32,
    pub(crate) rotation_degrees: f32,
    pub(crate) reflect_x: bool,
    pub(crate) reflect_y: bool,
}

impl Moc2Affine {
    /// Creates an identity affine keyform.
    pub(crate) fn identity() -> Self {
        Self {
            origin_x: 0.0,
            origin_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            rotation_degrees: 0.0,
            reflect_x: false,
            reflect_y: false,
        }
    }

    /// Returns the pivot position in model pixels.
    pub fn origin(&self) -> [f32; 2] {
        [self.origin_x, self.origin_y]
    }

    /// Returns the horizontal and vertical scale.
    pub fn scale(&self) -> [f32; 2] {
        [self.scale_x, self.scale_y]
    }

    /// Returns the rotation in degrees.
    pub fn rotation_degrees(&self) -> f32 {
        self.rotation_degrees
    }

    /// Returns whether the horizontal axis is mirrored.
    pub fn reflect_x(&self) -> bool {
        self.reflect_x
    }

    /// Returns whether the vertical axis is mirrored.
    pub fn reflect_y(&self) -> bool {
        self.reflect_y
    }
}

/// Geometry payload of a Cubism 2 deformer.
#[derive(Debug, Clone, PartialEq)]
pub enum Moc2DeformerKind {
    /// A warp grid with `x_cells * y_cells` cells.
    Warp {
        /// Number of grid cells along the x axis (the model file's `row`).
        x_cells: usize,
        /// Number of grid cells along the y axis (the model file's `col`).
        y_cells: usize,
        /// Pivot-major grid positions, two floats per grid point.
        points: Vec<f32>,
    },
    /// A rotation deformer with affine keyforms.
    Rotation {
        /// Affine keyforms, one per pivot table entry.
        affines: Vec<Moc2Affine>,
    },
}

impl Moc2DeformerKind {
    /// Returns the number of grid points for a warp deformer, or `0`.
    pub fn grid_point_count(&self) -> usize {
        match self {
            Self::Warp {
                x_cells, y_cells, ..
            } => (x_cells + 1) * (y_cells + 1),
            Self::Rotation { .. } => 0,
        }
    }
}

/// One warp or rotation deformer of a Cubism 2 model.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2Deformer {
    pub(crate) id: String,
    pub(crate) target: Option<usize>,
    pub(crate) pivots: Moc2PivotTable,
    pub(crate) opacities: Vec<f32>,
    pub(crate) kind: Moc2DeformerKind,
}

impl Moc2Deformer {
    /// Returns the deformer id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the parent deformer index, if the deformer is nested.
    pub fn target(&self) -> Option<usize> {
        self.target
    }

    /// Returns the deformer kind and its keyform payload.
    pub fn kind(&self) -> &Moc2DeformerKind {
        &self.kind
    }

    /// Returns the interpolation table used by this deformer.
    pub fn pivots(&self) -> &Moc2PivotTable {
        &self.pivots
    }

    /// Returns the opacity keyforms, or an empty slice when none are declared.
    pub fn opacities(&self) -> &[f32] {
        &self.opacities
    }
}

/// One drawable (art mesh) of a Cubism 2 model.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2Drawable {
    pub(crate) id: String,
    pub(crate) target: Option<usize>,
    pub(crate) texture: i32,
    pub(crate) blend_flags: u8,
    pub(crate) culling: bool,
    pub(crate) point_count: usize,
    pub(crate) indices: Vec<u16>,
    pub(crate) uvs: Vec<f32>,
    pub(crate) pivots: Moc2PivotTable,
    pub(crate) positions: Vec<f32>,
    pub(crate) draw_orders: Vec<i32>,
    pub(crate) opacities: Vec<f32>,
    pub(crate) clips: Vec<usize>,
    pub(crate) part_index: usize,
}

impl Moc2Drawable {
    /// Returns the drawable id. Some exporters leave this empty.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the deformer index that transforms this drawable.
    pub fn target(&self) -> Option<usize> {
        self.target
    }

    /// Returns the zero-based texture index.
    pub fn texture_index(&self) -> i32 {
        self.texture
    }

    /// Returns the number of mesh vertices.
    pub fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns triangle indices into the vertex arrays.
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// Returns the texture coordinates, two floats per vertex.
    pub fn uvs(&self) -> &[f32] {
        &self.uvs
    }

    /// Returns the interpolation table used by this drawable.
    pub fn pivots(&self) -> &Moc2PivotTable {
        &self.pivots
    }

    /// Returns the drawable opacity keyforms.
    pub fn opacities(&self) -> &[f32] {
        &self.opacities
    }

    /// Returns the draw-order keyforms.
    pub fn draw_orders(&self) -> &[i32] {
        &self.draw_orders
    }

    /// Returns model-order indices of drawables used as clipping masks.
    pub fn clips(&self) -> &[usize] {
        &self.clips
    }

    /// Returns the owning part index.
    pub fn part_index(&self) -> usize {
        self.part_index
    }

    /// Returns whether back-face culling is enabled for this drawable.
    ///
    /// Cubism 2 stores culling as an inverted flag: bit 5 *disables* culling.
    pub fn is_culling(&self) -> bool {
        self.culling
    }

    /// Returns the blend flags used by [`crate::moc3::Moc3DrawableMesh`].
    pub fn blend_flags(&self) -> u8 {
        self.blend_flags
    }
}

/// One axis of a Cubism 2 parameter pivot table.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2PivotDimension {
    pub(crate) parameter_index: usize,
    pub(crate) values: Vec<f32>,
}

impl Moc2PivotDimension {
    /// Returns the model parameter index driving this axis.
    pub fn parameter_index(&self) -> usize {
        self.parameter_index
    }

    /// Returns the pivot values declared along this axis.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// A Cubism 2 pivot table: one axis per driving parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2PivotTable {
    pub(crate) dimensions: Vec<Moc2PivotDimension>,
    pub(crate) pivot_count: usize,
}

impl Moc2PivotTable {
    /// Builds a pivot table and validates the pivot count.
    pub(crate) fn new(dimensions: Vec<Moc2PivotDimension>) -> Option<Self> {
        if dimensions.len() > MAX_PIVOT_DIMENSIONS
            || dimensions
                .iter()
                .any(|dimension| dimension.values.is_empty())
        {
            return None;
        }

        let mut pivot_count = 1usize;
        for dimension in &dimensions {
            pivot_count = pivot_count.checked_mul(dimension.values.len())?;
        }

        Some(Self {
            dimensions,
            pivot_count,
        })
    }

    /// Returns the pivot axes in evaluation order.
    pub fn dimensions(&self) -> &[Moc2PivotDimension] {
        &self.dimensions
    }

    /// Returns the total number of keyform entries in the table.
    pub fn pivot_count(&self) -> usize {
        self.pivot_count
    }

    /// Evaluates the pivot table for the current parameter values.
    ///
    /// Returns `None` when the table is malformed, or when a parameter index is
    /// outside `parameter_values`.
    pub fn sample(&self, parameter_values: &[f32]) -> Option<Moc2PivotSample> {
        let mut pivot_indices = [0usize; MAX_PIVOT_DIMENSIONS];
        let mut dimension_fractions = [0.0f32; MAX_PIVOT_DIMENSIONS];
        let mut outside = false;

        for (dimension_index, dimension) in self.dimensions.iter().enumerate() {
            let value = *parameter_values.get(dimension.parameter_index)?;
            let values = &dimension.values;
            let (pivot_index, fraction, outside_value) = locate_pivot(values, value);
            pivot_indices[dimension_index] = pivot_index;
            dimension_fractions[dimension_index] = fraction;
            outside |= outside_value;
        }

        let corner_count = 1usize << self.dimensions.len();
        let mut indices = [0usize; MAX_PIVOT_CORNERS];
        let mut fractions = [0.0f32; MAX_PIVOT_DIMENSIONS];
        let mut interpolated_dimensions = 0usize;
        let mut stride = 1usize;
        let mut bit_group = 1usize;

        for (dimension_index, dimension) in self.dimensions.iter().enumerate() {
            let pivot_index = pivot_indices[dimension_index];
            let fraction = dimension_fractions[dimension_index];

            if fraction == 0.0 {
                let base = pivot_index * stride;
                for index in &mut indices[..corner_count] {
                    *index += base;
                }
            } else {
                let low = pivot_index * stride;
                let high = (pivot_index + 1) * stride;
                for (corner, index) in indices[..corner_count].iter_mut().enumerate() {
                    *index += if (corner / bit_group).is_multiple_of(2) {
                        low
                    } else {
                        high
                    };
                }
                fractions[interpolated_dimensions] = fraction;
                interpolated_dimensions += 1;
                bit_group *= 2;
            }

            stride = stride.checked_mul(dimension.values.len())?;
        }

        if stride != self.pivot_count {
            return None;
        }

        Some(Moc2PivotSample {
            indices,
            fractions,
            corner_count: 1usize << interpolated_dimensions,
            interpolated_dimensions,
            outside,
        })
    }
}

fn locate_pivot(values: &[f32], value: f32) -> (usize, f32, bool) {
    if values.len() == 1 {
        let inside = values[0] - GOSA < value && value < values[0] + GOSA;
        return (0, 0.0, !inside);
    }

    let first = values[0];
    if value < first - GOSA {
        return (0, 0.0, true);
    }
    if value < first + GOSA {
        return (0, 0.0, false);
    }

    let last = values.len() - 1;
    let mut previous = first;
    for (index, current) in values.iter().enumerate().skip(1) {
        if value < *current + GOSA {
            if *current - GOSA < value {
                return (index, 0.0, false);
            }
            let span = *current - previous;
            let fraction = if span > 0.0 {
                (value - previous) / span
            } else {
                0.0
            };
            return (index - 1, fraction.clamp(0.0, 1.0), false);
        }
        previous = *current;
    }

    (last, 0.0, true)
}

/// The keyform corners selected for one parameter state.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Moc2PivotSample {
    indices: [usize; MAX_PIVOT_CORNERS],
    fractions: [f32; MAX_PIVOT_DIMENSIONS],
    corner_count: usize,
    interpolated_dimensions: usize,
    outside: bool,
}

impl Moc2PivotSample {
    /// Returns whether any driving parameter is outside the declared key range.
    ///
    /// Cubism 2 hides drawables and skips deformer transforms in that state.
    pub fn is_outside(&self) -> bool {
        self.outside
    }

    /// Returns the number of keyform corners used by this sample.
    pub fn corner_count(&self) -> usize {
        self.corner_count
    }

    /// Returns the interpolated keyform indices.
    ///
    /// Only the first [`corner_count`](Self::corner_count) entries are relevant.
    pub fn indices(&self) -> &[usize] {
        &self.indices[..self.corner_count]
    }

    /// Returns the blend fractions for each interpolated axis.
    pub fn fractions(&self) -> &[f32] {
        &self.fractions[..self.interpolated_dimensions]
    }

    fn weight(&self, corner: usize) -> f32 {
        let mut weight = 1.0;
        let mut bits = corner;
        for fraction in &self.fractions[..self.interpolated_dimensions] {
            weight *= if bits & 1 == 0 {
                1.0 - *fraction
            } else {
                *fraction
            };
            bits >>= 1;
        }
        weight
    }

    fn weight_count(&self) -> usize {
        1usize << self.interpolated_dimensions
    }

    /// Iterates over the selected keyform indices and their blend weights.
    pub fn corners(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        (0..self.weight_count()).map(|corner| (self.indices[corner], self.weight(corner)))
    }

    /// Interpolates a keyform-major float array.
    ///
    /// `values` must contain `pivot_count * stride` entries.
    pub fn interpolate_float(&self, values: &[f32]) -> Option<f32> {
        let mut total = 0.0;
        for corner in 0..self.weight_count() {
            let value = *values.get(*self.indices.get(corner)?)?;
            total += value * self.weight(corner);
        }
        Some(total)
    }

    /// Interpolates a keyform-major integer array, rounding the result.
    pub fn interpolate_int(&self, values: &[i32]) -> Option<i32> {
        let mut total = 0.0f32;
        for corner in 0..self.weight_count() {
            let value = *values.get(*self.indices.get(corner)?)?;
            total += value as f32 * self.weight(corner);
        }
        Some((total + 0.5).floor() as i32)
    }

    /// Interpolates a keyform-major vertex array into `output`.
    ///
    /// `values` must contain `pivot_count * output.len()` entries.
    pub fn interpolate_points(&self, values: &[f32], output: &mut [f32]) -> Option<()> {
        output.fill(0.0);
        let stride = output.len();
        for corner in 0..self.weight_count() {
            let offset = self.indices.get(corner)?.checked_mul(stride)?;
            let slice = values.get(offset..offset.checked_add(stride)?)?;
            let weight = self.weight(corner);
            for (target, value) in output.iter_mut().zip(slice) {
                *target += value * weight;
            }
        }
        Some(())
    }
}

/// A complete parsed Cubism 2 model.
///
/// The scene owns parameters, parts, deformers, and drawables in model order.
/// Deformers are stored parent-first so a single forward pass can evaluate the
/// whole hierarchy. Use [`crate::moc2::Moc2Runtime`] to animate and draw it.
#[derive(Debug, Clone)]
pub struct Moc2Scene {
    pub(crate) format_version: u8,
    pub(crate) canvas: Moc2Canvas,
    pub(crate) parameters: Vec<Moc2Parameter>,
    pub(crate) parts: Vec<Moc2Part>,
    pub(crate) drawables: Vec<Moc2Drawable>,
    pub(crate) deformers: Vec<Moc2Deformer>,
    pub(crate) parameter_index: HashMap<String, usize>,
    pub(crate) drawable_index: HashMap<String, usize>,
    pub(crate) part_index: HashMap<String, usize>,
}

impl Moc2Scene {
    pub(crate) fn from_parts(
        format_version: u8,
        canvas: Moc2Canvas,
        parameters: Vec<Moc2Parameter>,
        parts: Vec<Moc2Part>,
        drawables: Vec<Moc2Drawable>,
        deformers: Vec<Moc2Deformer>,
    ) -> Self {
        let parameter_index = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.id.clone(), index))
            .collect();
        let drawable_index = drawables
            .iter()
            .enumerate()
            .filter(|(_, drawable)| !drawable.id.is_empty())
            .map(|(index, drawable)| (drawable.id.clone(), index))
            .collect();
        let part_index = parts
            .iter()
            .enumerate()
            .map(|(index, part)| (part.id.clone(), index))
            .collect();

        Self {
            format_version,
            canvas,
            parameters,
            parts,
            drawables,
            deformers,
            parameter_index,
            drawable_index,
            part_index,
        }
    }

    /// Returns the `.moc` serialization version.
    pub fn format_version(&self) -> u8 {
        self.format_version
    }

    /// Returns the declared canvas size.
    pub fn canvas(&self) -> Moc2Canvas {
        self.canvas
    }

    /// Returns the canvas width in model pixels.
    pub fn canvas_width(&self) -> u32 {
        self.canvas.width
    }

    /// Returns the canvas height in model pixels.
    pub fn canvas_height(&self) -> u32 {
        self.canvas.height
    }

    /// Returns all parameter definitions in model order.
    pub fn parameters(&self) -> &[Moc2Parameter] {
        &self.parameters
    }

    /// Returns the model-order index of a parameter id.
    pub fn parameter_index(&self, id: &str) -> Option<usize> {
        self.parameter_index.get(id).copied()
    }

    /// Returns all parts in model order.
    pub fn parts(&self) -> &[Moc2Part] {
        &self.parts
    }

    /// Returns the model-order index of a part id.
    pub fn part_index(&self, id: &str) -> Option<usize> {
        self.part_index.get(id).copied()
    }

    /// Returns all drawables in model order.
    pub fn drawables(&self) -> &[Moc2Drawable] {
        &self.drawables
    }

    /// Returns the model-order index of a drawable id.
    pub fn drawable_index(&self, id: &str) -> Option<usize> {
        self.drawable_index.get(id).copied()
    }

    /// Returns all deformers in evaluation (parent-first) order.
    pub fn deformers(&self) -> &[Moc2Deformer] {
        &self.deformers
    }

    /// Returns the model-order index of a deformer id.
    pub fn deformer_index(&self, id: &str) -> Option<usize> {
        self.deformers.iter().position(|deformer| deformer.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dimension(parameter_index: usize, values: &[f32]) -> Moc2PivotDimension {
        Moc2PivotDimension {
            parameter_index,
            values: values.to_vec(),
        }
    }

    #[test]
    fn single_axis_interpolates_between_keyforms() {
        let table =
            Moc2PivotTable::new(vec![dimension(0, &[0.0, 1.0])]).expect("valid pivot table");
        let sample = table.sample(&[0.25]).expect("sample");

        assert_eq!(sample.indices(), &[0, 1]);
        assert_eq!(sample.fractions(), &[0.25]);
        assert!(!sample.is_outside());
        assert!((sample.interpolate_float(&[10.0, 20.0]).unwrap() - 12.5).abs() < 1e-6);
        assert_eq!(sample.interpolate_int(&[10, 20]).unwrap(), 13);

        let mut output = [0.0; 4];
        sample
            .interpolate_points(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &mut output)
            .unwrap();
        assert_eq!(output, [0.5, 0.5, 1.5, 1.5]);
    }

    #[test]
    fn pivot_exactly_on_a_keyform_has_no_fraction() {
        let table = Moc2PivotTable::new(vec![dimension(0, &[-1.0, 0.0, 1.0])]).unwrap();
        let sample = table.sample(&[0.0]).unwrap();

        assert_eq!(sample.indices(), &[1]);
        assert_eq!(sample.corner_count(), 1);
        assert_eq!(sample.interpolate_float(&[5.0, 7.0, 9.0]).unwrap(), 7.0);
    }

    #[test]
    fn values_outside_the_key_range_are_reported() {
        let table = Moc2PivotTable::new(vec![dimension(0, &[0.0, 1.0])]).unwrap();

        let below = table.sample(&[-2.0]).unwrap();
        assert!(below.is_outside());
        assert_eq!(below.indices(), &[0]);

        let above = table.sample(&[3.0]).unwrap();
        assert!(above.is_outside());
        assert_eq!(above.indices(), &[1]);
    }

    #[test]
    fn two_axes_blend_all_four_corners() {
        let table = Moc2PivotTable::new(vec![dimension(0, &[0.0, 1.0]), dimension(1, &[0.0, 1.0])])
            .unwrap();
        let sample = table.sample(&[0.5, 0.5]).unwrap();

        assert_eq!(sample.corner_count(), 4);
        assert_eq!(sample.indices(), &[0, 1, 2, 3]);
        let values = [0.0, 10.0, 20.0, 30.0];
        assert!((sample.interpolate_float(&values).unwrap() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn zero_length_pivot_axis_does_not_change_the_keyform_index() {
        let table =
            Moc2PivotTable::new(vec![dimension(0, &[0.0]), dimension(1, &[0.0, 1.0])]).unwrap();
        let sample = table.sample(&[0.0, 0.5]).unwrap();

        assert_eq!(sample.corner_count(), 2);
        assert_eq!(sample.indices(), &[0, 1]);
    }
}
