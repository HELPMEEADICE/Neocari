use crate::core::draw_order_from_raw;

use super::{Moc3ArtMeshKeyforms, Moc3ArtMeshes};

const DRAWABLE_BLEND_ADDITIVE: u8 = 1 << 0;
const DRAWABLE_BLEND_MULTIPLICATIVE: u8 = 1 << 1;
/// Internal flag bit for Cubism 2 `Screen` composition.
///
/// MOC3 art meshes use this bit for a constant flag that Cubism 3/4/5 exporters
/// set on virtually every drawable, so MOC3 flags must be decoded with
/// [`Moc3DrawableBlendMode::from_moc3_flags`] rather than
/// [`Moc3DrawableBlendMode::from_flags`].
const DRAWABLE_BLEND_SCREEN: u8 = 1 << 2;
/// Inverted clipping-mask flag, shared by MOC2 and MOC3.
const DRAWABLE_MASK_INVERTED: u8 = 1 << 3;

#[derive(Debug, Copy, Clone, PartialEq)]
/// One vertex in a generated drawable mesh.
pub struct Moc3DrawableVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl Moc3DrawableVertex {
    /// Creates a drawable vertex from model-space position and UV coordinates.
    pub fn new(position: [f32; 2], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }

    /// Returns the model-space position.
    pub fn position(&self) -> [f32; 2] {
        self.position
    }

    pub(crate) fn set_position(&mut self, position: [f32; 2]) {
        self.position = position;
    }

    /// Returns the texture coordinate.
    pub fn uv(&self) -> [f32; 2] {
        self.uv
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A drawable mesh generated from MOC3 art mesh data.
///
/// Renderers draw these meshes with the texture, opacity, blend mode, mask
/// information, and colors carried by the mesh.
pub struct Moc3DrawableMesh {
    texture_index: i32,
    drawable_flags: u8,
    blend_mode: Moc3DrawableBlendMode,
    opacity: f32,
    draw_order: f32,
    render_order: i32,
    multiply_color: [f32; 3],
    screen_color: [f32; 3],
    vertices: Vec<Moc3DrawableVertex>,
    indices: Vec<u16>,
    masks: Vec<i32>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Blend mode for a drawable mesh.
pub enum Moc3DrawableBlendMode {
    /// Standard alpha blending.
    Normal,
    /// Additive blending.
    Additive,
    /// Multiplicative blending.
    Multiplicative,
    /// Screen composition used by Cubism 2 drawables.
    Screen,
}

impl Moc3DrawableBlendMode {
    /// Decodes a blend mode from the internal drawable flag byte.
    ///
    /// The internal encoding is used by Cubism 2 drawables: bit 0 additive, bit 1
    /// multiplicative and bit 2 `Screen`. MOC3 art meshes decode their raw flags
    /// with [`from_moc3_flags`](Self::from_moc3_flags) instead, because MOC3 uses
    /// bit 2 for a constant flag rather than for `Screen`.
    pub fn from_flags(flags: u8) -> Self {
        if flags & DRAWABLE_BLEND_ADDITIVE != 0 {
            Self::Additive
        } else if flags & DRAWABLE_BLEND_MULTIPLICATIVE != 0 {
            Self::Multiplicative
        } else if flags & DRAWABLE_BLEND_SCREEN != 0 {
            Self::Screen
        } else {
            Self::Normal
        }
    }

    /// Decodes the blend mode of a raw MOC3 art-mesh constant-flag byte.
    ///
    /// MOC3 stores additive (bit 0) and multiplicative (bit 1) composition there;
    /// bit 2 is a constant flag that Cubism 3/4/5 exporters set on virtually every
    /// art mesh, and bit 3 marks inverted clipping masks. Bit 2 is **not** the
    /// Cubism 2 `Screen` blend: decoding it as one screen-composites every
    /// drawable (`One/OneMinusSrc`), which bleaches MOC3 models to white and
    /// drags the face and joint highlights over the clothes.
    pub fn from_moc3_flags(flags: u8) -> Self {
        if flags & DRAWABLE_BLEND_ADDITIVE != 0 {
            Self::Additive
        } else if flags & DRAWABLE_BLEND_MULTIPLICATIVE != 0 {
            Self::Multiplicative
        } else {
            Self::Normal
        }
    }
}

impl Moc3DrawableMesh {
    /// Creates a drawable mesh from raw parts.
    pub fn from_parts(
        texture_index: i32,
        drawable_flags: u8,
        opacity: f32,
        draw_order: f32,
        vertices: Vec<Moc3DrawableVertex>,
        indices: Vec<u16>,
        masks: Vec<i32>,
    ) -> Self {
        Self::from_parts_with_render_order(
            texture_index,
            drawable_flags,
            opacity,
            draw_order,
            draw_order_from_raw(draw_order),
            vertices,
            indices,
            masks,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Creates a drawable mesh from raw parts with an explicit render order.
    pub fn from_parts_with_render_order(
        texture_index: i32,
        drawable_flags: u8,
        opacity: f32,
        draw_order: f32,
        render_order: i32,
        vertices: Vec<Moc3DrawableVertex>,
        indices: Vec<u16>,
        masks: Vec<i32>,
    ) -> Self {
        Self {
            texture_index,
            drawable_flags,
            blend_mode: Moc3DrawableBlendMode::from_flags(drawable_flags),
            opacity,
            draw_order,
            render_order,
            multiply_color: [1.0, 1.0, 1.0],
            screen_color: [0.0, 0.0, 0.0],
            vertices,
            indices,
            masks,
        }
    }

    /// Returns the texture index used by this drawable.
    pub fn texture_index(&self) -> i32 {
        self.texture_index
    }

    /// Returns the multiply color applied to this drawable.
    pub fn multiply_color(&self) -> [f32; 3] {
        self.multiply_color
    }

    /// Returns the screen color applied to this drawable.
    pub fn screen_color(&self) -> [f32; 3] {
        self.screen_color
    }

    pub(crate) fn set_multiply_color(&mut self, color: [f32; 3]) {
        self.multiply_color = color;
    }

    pub(crate) fn set_screen_color(&mut self, color: [f32; 3]) {
        self.screen_color = color;
    }

    /// Returns the raw drawable flags from the model.
    pub fn drawable_flags(&self) -> u8 {
        self.drawable_flags
    }

    /// Returns the blend mode resolved for this drawable.
    pub fn blend_mode(&self) -> Moc3DrawableBlendMode {
        self.blend_mode
    }

    pub(crate) fn set_blend_mode(&mut self, blend_mode: Moc3DrawableBlendMode) {
        self.blend_mode = blend_mode;
    }

    /// Returns whether this drawable uses inverted mask semantics.
    pub fn is_inverted_mask(&self) -> bool {
        self.drawable_flags & DRAWABLE_MASK_INVERTED != 0
    }

    /// Returns the drawable opacity.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    pub(crate) fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity;
    }

    /// Returns the raw draw order value.
    pub fn draw_order(&self) -> f32 {
        self.draw_order
    }

    pub(crate) fn set_draw_order(&mut self, draw_order: f32) {
        self.draw_order = draw_order;
    }

    /// Returns the resolved render order value.
    pub fn render_order(&self) -> i32 {
        self.render_order
    }

    pub(crate) fn set_render_order(&mut self, render_order: i32) {
        self.render_order = render_order;
    }

    /// Returns mesh vertices in model coordinates.
    pub fn vertices(&self) -> &[Moc3DrawableVertex] {
        &self.vertices
    }

    pub(crate) fn vertices_mut(&mut self) -> &mut [Moc3DrawableVertex] {
        &mut self.vertices
    }

    /// Returns triangle indices into [`vertices`](Self::vertices).
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// Returns drawable indices used as masks for this drawable.
    pub fn masks(&self) -> &[i32] {
        &self.masks
    }
}

/// Builds one drawable mesh for an art mesh keyform.
pub fn build_moc3_drawable_mesh(
    art_meshes: &Moc3ArtMeshes,
    keyforms: &Moc3ArtMeshKeyforms,
    art_mesh_index: usize,
    local_keyform_index: usize,
) -> Option<Moc3DrawableMesh> {
    let mesh = *art_meshes.meshes().get(art_mesh_index)?;
    let keyform = *keyforms
        .art_mesh_keyforms(art_mesh_index)?
        .get(local_keyform_index)?;
    let positions = keyforms.art_mesh_keyform_positions(art_mesh_index, local_keyform_index)?;
    let uvs = art_meshes.art_mesh_uvs(art_mesh_index)?;
    if positions.len() != uvs.len() || positions.len() % 2 != 0 {
        return None;
    }

    let vertices = positions
        .as_chunks::<2>()
        .0
        .iter()
        .zip(uvs.as_chunks::<2>().0.iter())
        .map(|(position, uv)| Moc3DrawableVertex::new(*position, *uv))
        .collect::<Vec<_>>();

    let mut indices = Vec::with_capacity(mesh.position_index_count() as usize);
    for position_index in art_meshes.art_mesh_position_indices(art_mesh_index)? {
        let position_index = u16::try_from(*position_index).ok()?;
        if usize::from(position_index) >= vertices.len() {
            return None;
        }
        indices.push(position_index);
    }

    let mut drawable = Moc3DrawableMesh::from_parts_with_render_order(
        mesh.texture_index(),
        mesh.drawable_flags(),
        keyform.opacity(),
        keyform.draw_order(),
        art_meshes.art_mesh_render_order(art_mesh_index)?,
        vertices,
        indices,
        art_meshes.art_mesh_masks(art_mesh_index)?.to_vec(),
    );
    drawable.set_blend_mode(Moc3DrawableBlendMode::from_moc3_flags(
        mesh.drawable_flags(),
    ));
    Some(drawable)
}

/// Builds drawable meshes for all art meshes using their first keyform.
pub fn build_moc3_drawable_meshes(
    art_meshes: &Moc3ArtMeshes,
    keyforms: &Moc3ArtMeshKeyforms,
) -> Option<Vec<Moc3DrawableMesh>> {
    let mut meshes = Vec::with_capacity(art_meshes.meshes().len());

    for art_mesh_index in 0..art_meshes.meshes().len() {
        meshes.push(build_moc3_drawable_mesh(
            art_meshes,
            keyforms,
            art_mesh_index,
            0,
        )?);
    }

    Some(meshes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moc3_constant_flag_is_not_decoded_as_screen_blend() {
        // Cubism 3/4/5 exporters set bit 2 on ~every art mesh. Only the blend bits
        // (0 additive, 1 multiplicative) may participate, or the whole model is
        // composited with `Screen` and bleaches to white.
        assert_eq!(
            Moc3DrawableBlendMode::from_moc3_flags(0b0100),
            Moc3DrawableBlendMode::Normal
        );
        assert_eq!(
            Moc3DrawableBlendMode::from_moc3_flags(0b0110),
            Moc3DrawableBlendMode::Multiplicative
        );
        assert_eq!(
            Moc3DrawableBlendMode::from_moc3_flags(0b0101),
            Moc3DrawableBlendMode::Additive
        );
        assert_eq!(
            Moc3DrawableBlendMode::from_moc3_flags(0b1100),
            Moc3DrawableBlendMode::Normal
        );
    }

    #[test]
    fn cubism2_screen_flag_still_maps_to_screen_blend() {
        // `moc2` synthesises its own flag byte; `Screen` must keep its dedicated bit.
        assert_eq!(
            Moc3DrawableBlendMode::from_flags(DRAWABLE_BLEND_SCREEN),
            Moc3DrawableBlendMode::Screen
        );
        assert_eq!(
            Moc3DrawableBlendMode::from_flags(DRAWABLE_BLEND_MULTIPLICATIVE),
            Moc3DrawableBlendMode::Multiplicative
        );
        assert_eq!(
            Moc3DrawableBlendMode::from_flags(DRAWABLE_BLEND_ADDITIVE),
            Moc3DrawableBlendMode::Additive
        );
        assert_eq!(
            Moc3DrawableBlendMode::from_flags(0),
            Moc3DrawableBlendMode::Normal
        );
    }
}
