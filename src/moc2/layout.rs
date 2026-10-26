//! Cubism 2 model placement matrix.
//!
//! Cubism 2 model coordinates are canvas pixels with the origin at the top-left
//! corner and the y axis pointing down. The Live2D V2 framework maps that space
//! to clip space with [`Moc2ModelMatrix::new`]: the canvas is scaled so its width
//! spans two clip units and the y axis is flipped.

use std::collections::BTreeMap;

use crate::core::Matrix44;

use super::scene::Moc2Canvas;

/// Builds the model-to-clip-space matrix used by Cubism 2 models.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2ModelMatrix {
    width: f32,
    height: f32,
    matrix: Matrix44,
}

impl Moc2ModelMatrix {
    /// Creates the matrix for a model canvas.
    ///
    /// The matrix starts as the identity. Call [`set_width`](Self::set_width)
    /// (usually with `2.0`) and then
    /// [`set_center_position`](Self::set_center_position) to reproduce the V2
    /// framework defaults, or use [`from_layout`](Self::from_layout).
    pub fn new(canvas: Moc2Canvas) -> Self {
        Self {
            width: canvas.width() as f32,
            height: canvas.height() as f32,
            matrix: Matrix44::identity(),
        }
    }

    /// Creates the default V2 matrix (`width = 2`, centered on the origin) and
    /// applies optional `layout` values from `model.json`.
    pub fn from_layout(canvas: Moc2Canvas, layout: Option<&BTreeMap<String, f32>>) -> Self {
        let mut matrix = Self::new(canvas);
        matrix.set_width(2.0);
        matrix.set_center_position(0.0, 0.0);
        if let Some(layout) = layout {
            matrix.apply_layout(layout);
        }
        matrix
    }

    /// Applies the keys supported by Cubism 2 `layout` objects.
    pub fn apply_layout(&mut self, layout: &BTreeMap<String, f32>) {
        for (key, value) in layout {
            match key.as_str() {
                "width" => self.set_width(*value),
                "height" => self.set_height(*value),
                _ => {}
            }
        }
        for (key, value) in layout {
            match key.as_str() {
                "x" => self.set_x(*value),
                "y" => self.set_y(*value),
                "center_x" => self.center_x(*value),
                "center_y" => self.center_y(*value),
                "top" => self.top(*value),
                "bottom" => self.bottom(*value),
                "left" => self.left(*value),
                "right" => self.right(*value),
                _ => {}
            }
        }
    }

    /// Returns the matrix that maps model pixels to clip space.
    pub fn matrix(&self) -> Matrix44 {
        self.matrix
    }

    /// Places the model so its center is at the given clip-space position.
    pub fn set_center_position(&mut self, x: f32, y: f32) {
        let scaled_width = self.width * self.scale_x();
        let scaled_height = self.height * self.scale_y();
        self.matrix
            .translate(x - scaled_width / 2.0, y - scaled_height / 2.0);
    }

    /// Aligns the model top edge with `y`.
    pub fn top(&mut self, y: f32) {
        self.set_y(y);
    }

    /// Aligns the model bottom edge with `y`.
    pub fn bottom(&mut self, y: f32) {
        let height = self.height * self.scale_y();
        self.matrix.translate_y(y - height);
    }

    /// Aligns the model left edge with `x`.
    pub fn left(&mut self, x: f32) {
        self.set_x(x);
    }

    /// Aligns the model right edge with `x`.
    pub fn right(&mut self, x: f32) {
        let width = self.width * self.scale_x();
        self.matrix.translate_x(x - width);
    }

    /// Centers the model horizontally on `x`.
    pub fn center_x(&mut self, x: f32) {
        let width = self.width * self.scale_x();
        self.matrix.translate_x(x - width / 2.0);
    }

    /// Centers the model vertically on `y`.
    pub fn center_y(&mut self, y: f32) {
        let height = self.height * self.scale_y();
        self.matrix.translate_y(y - height / 2.0);
    }

    /// Sets the horizontal translation.
    pub fn set_x(&mut self, x: f32) {
        self.matrix.translate_x(x);
    }

    /// Sets the vertical translation.
    pub fn set_y(&mut self, y: f32) {
        self.matrix.translate_y(y);
    }

    /// Scales the model so its height spans `height` clip units.
    pub fn set_height(&mut self, height: f32) {
        let scale_x = height / self.height;
        self.matrix.scale(scale_x, -scale_x);
    }

    /// Scales the model so its width spans `width` clip units.
    pub fn set_width(&mut self, width: f32) {
        let scale_x = width / self.width;
        self.matrix.scale(scale_x, -scale_x);
    }
}

impl Moc2ModelMatrix {
    fn scale_x(&self) -> f32 {
        self.matrix.scale_x()
    }

    fn scale_y(&self) -> f32 {
        self.matrix.scale_y()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(matrix: &Matrix44, x: f32, y: f32) -> [f32; 2] {
        [matrix.transform_x(x), matrix.transform_y(y)]
    }

    #[test]
    fn default_matrix_spans_two_units_and_flips_y() {
        let matrix = Moc2ModelMatrix::from_layout(Moc2Canvas::new(400, 650), None).matrix();

        let top_left = transform(&matrix, 0.0, 0.0);
        let top_right = transform(&matrix, 400.0, 0.0);
        let bottom_left = transform(&matrix, 0.0, 650.0);

        assert!((top_left[0] + 1.0).abs() < 1e-6);
        assert!((top_right[0] - 1.0).abs() < 1e-6);
        assert!(top_left[1] > bottom_left[1]);
        assert!((top_left[1] - 1.625).abs() < 1e-6);
    }

    #[test]
    fn layout_width_and_centering_are_applied() {
        let mut layout = BTreeMap::new();
        layout.insert("width".to_owned(), 1.0);
        layout.insert("center_x".to_owned(), 0.25);
        let matrix =
            Moc2ModelMatrix::from_layout(Moc2Canvas::new(200, 200), Some(&layout)).matrix();

        let left = transform(&matrix, 0.0, 0.0);
        let right = transform(&matrix, 200.0, 0.0);
        assert!((right[0] - left[0] - 1.0).abs() < 1e-6);
        assert!((left[0] + right[0]).abs() / 2.0 > 0.2);
    }
}
