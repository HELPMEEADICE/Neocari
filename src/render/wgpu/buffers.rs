use std::{ops::Range, sync::Arc};

use wgpu::util::DeviceExt;

use crate::moc3::{Moc3DrawableBlendMode, Moc3DrawableMesh};
use crate::render::common::{
    ClippingRect, DrawableInfo, DrawableVertex, append_vertices_from_drawable,
    draw_order_indices_from, draw_order_indices_from_into, encode_indices,
    encode_vertices_from_drawable,
};

pub fn drawable_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 8,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32,
            offset: 16,
            shader_location: 2,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 20,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 32,
            shader_location: 4,
        },
    ];

    wgpu::VertexBufferLayout {
        array_stride: DrawableVertex::STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

#[derive(Debug)]
pub struct WgpuDrawableBuffers {
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
    vertex_range: Range<u64>,
    index_range: Range<u64>,
    first_index: u32,
    base_vertex: i32,
    vertex_count: u32,
    index_count: u32,
    vertex_snapshot: Option<DrawableVertexSnapshot>,
    indices: Option<Vec<u16>>,
    info: DrawableInfo,
}

#[derive(Debug)]
struct DrawableVertexSnapshot {
    position_uv_bits: Vec<[u32; 4]>,
    opacity_bits: u32,
    multiply_bits: [u32; 3],
    screen_bits: [u32; 3],
}

#[derive(Debug, Copy, Clone)]
struct DrawableSnapshotChanges {
    vertex_data_changed: bool,
    bounds_changed: bool,
}

impl DrawableVertexSnapshot {
    fn from_mesh(mesh: &Moc3DrawableMesh) -> Self {
        let mut snapshot = Self {
            position_uv_bits: Vec::with_capacity(mesh.vertices().len()),
            opacity_bits: 0,
            multiply_bits: [0; 3],
            screen_bits: [0; 3],
        };
        snapshot.update(mesh);
        snapshot
    }

    fn changes(&self, mesh: &Moc3DrawableMesh) -> DrawableSnapshotChanges {
        if self.position_uv_bits.len() != mesh.vertices().len() {
            return DrawableSnapshotChanges {
                vertex_data_changed: true,
                bounds_changed: true,
            };
        }

        let mut vertex_data_changed = self.opacity_bits != mesh.opacity().to_bits()
            || self.multiply_bits != color_bits(mesh.multiply_color())
            || self.screen_bits != color_bits(mesh.screen_color());
        let mut bounds_changed = false;
        for (cached, vertex) in self.position_uv_bits.iter().zip(mesh.vertices()) {
            let position = vertex.position();
            let uv = vertex.uv();
            let current = [
                position[0].to_bits(),
                position[1].to_bits(),
                uv[0].to_bits(),
                uv[1].to_bits(),
            ];
            if *cached != current {
                vertex_data_changed = true;
                bounds_changed |= cached[0] != current[0] || cached[1] != current[1];
            }
        }

        DrawableSnapshotChanges {
            vertex_data_changed,
            bounds_changed,
        }
    }

    fn update(&mut self, mesh: &Moc3DrawableMesh) {
        self.position_uv_bits.clear();
        self.position_uv_bits
            .extend(mesh.vertices().iter().map(|vertex| {
                let position = vertex.position();
                let uv = vertex.uv();
                [
                    position[0].to_bits(),
                    position[1].to_bits(),
                    uv[0].to_bits(),
                    uv[1].to_bits(),
                ]
            }));
        self.opacity_bits = mesh.opacity().to_bits();
        self.multiply_bits = color_bits(mesh.multiply_color());
        self.screen_bits = color_bits(mesh.screen_color());
    }
}

impl WgpuDrawableBuffers {
    /// Returns the GPU resource containing this drawable's vertices. Static mesh
    /// buffers can share this resource with other drawables.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    /// Returns the vertex range occupied by this drawable.
    pub fn vertex_buffer_slice(&self) -> wgpu::BufferSlice<'_> {
        self.vertex_buffer.slice(self.vertex_range.clone())
    }

    /// Returns the GPU resource containing this drawable's indices. Static mesh
    /// buffers can share this resource with other drawables.
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    /// Returns the index range occupied by this drawable.
    pub fn index_buffer_slice(&self) -> wgpu::BufferSlice<'_> {
        self.index_buffer.slice(self.index_range.clone())
    }

    pub(super) fn draw_index_range(&self) -> Range<u32> {
        self.first_index..self.first_index + self.index_count
    }

    pub(super) fn base_vertex(&self) -> i32 {
        self.base_vertex
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn is_empty(&self) -> bool {
        self.vertex_count == 0 || self.index_count == 0
    }

    pub fn is_visible(&self) -> bool {
        !self.is_empty() && self.info.is_visible()
    }

    pub fn info(&self) -> &DrawableInfo {
        &self.info
    }

    pub fn texture_index(&self) -> i32 {
        self.info.texture_index()
    }

    pub fn blend_mode(&self) -> Moc3DrawableBlendMode {
        self.info.blend_mode()
    }

    pub fn opacity(&self) -> f32 {
        self.info.opacity()
    }

    pub fn draw_order(&self) -> f32 {
        self.info.draw_order()
    }

    pub fn render_order(&self) -> i32 {
        self.info.render_order()
    }

    pub fn masks(&self) -> &[i32] {
        self.info.masks()
    }

    pub fn inverted_mask(&self) -> bool {
        self.info.inverted_mask()
    }

    pub fn bounds(&self) -> Option<ClippingRect> {
        self.info.bounds()
    }
}

#[derive(Debug)]
pub struct WgpuMeshBuffers {
    drawables: Vec<WgpuDrawableBuffers>,
    draw_order_indices: Vec<usize>,
    render_order_seen: Vec<bool>,
    vertex_upload_bytes: Vec<u8>,
    packed_vertex_buffer: Option<Arc<wgpu::Buffer>>,
    packed_index_buffer: Option<Arc<wgpu::Buffer>>,
    static_geometry: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WgpuMeshUpdate {
    uploaded_drawables: usize,
    bounds_changed: bool,
    visibility_changed: bool,
}

impl WgpuMeshUpdate {
    pub fn uploaded_drawables(&self) -> usize {
        self.uploaded_drawables
    }

    pub fn bounds_changed(&self) -> bool {
        self.bounds_changed
    }

    pub fn visibility_changed(&self) -> bool {
        self.visibility_changed
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WgpuMeshUpdateError {
    #[error("drawable count changed from {expected} to {actual}")]
    DrawableCount { expected: usize, actual: usize },
    #[error("drawable {drawable_index} vertex count changed from {expected} to {actual}")]
    VertexCount {
        drawable_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("drawable {drawable_index} index count changed from {expected} to {actual}")]
    IndexCount {
        drawable_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("drawable {drawable_index} indices changed")]
    Indices { drawable_index: usize },
    #[error("drawable {drawable_index} texture index changed from {expected} to {actual}")]
    TextureIndex {
        drawable_index: usize,
        expected: i32,
        actual: i32,
    },
    #[error("drawable {drawable_index} blend mode changed from {expected:?} to {actual:?}")]
    BlendMode {
        drawable_index: usize,
        expected: Moc3DrawableBlendMode,
        actual: Moc3DrawableBlendMode,
    },
    #[error("drawable {drawable_index} masks changed")]
    Masks { drawable_index: usize },
    #[error("static mesh buffers cannot be updated")]
    StaticGeometry,
    #[error("drawable {drawable_index} inverted mask changed from {expected} to {actual}")]
    InvertedMask {
        drawable_index: usize,
        expected: bool,
        actual: bool,
    },
}

impl WgpuMeshBuffers {
    pub fn from_drawables(device: &wgpu::Device, meshes: &[Moc3DrawableMesh]) -> Option<Self> {
        let mut drawables = Vec::with_capacity(meshes.len());
        for mesh in meshes {
            drawables.push(create_wgpu_drawable_buffers(device, mesh)?);
        }
        let draw_order_indices = draw_order_indices_from(
            drawables.len(),
            |index| drawables[index].draw_order(),
            |index| drawables[index].render_order(),
        );

        Some(Self {
            render_order_seen: Vec::with_capacity(drawables.len()),
            vertex_upload_bytes: Vec::new(),
            drawables,
            draw_order_indices,
            packed_vertex_buffer: None,
            packed_index_buffer: None,
            static_geometry: false,
        })
    }

    /// Creates compact GPU buffers for immutable drawable geometry, such as the
    /// default-pose meshes produced by CubismV2. Geometry and indices are stored
    /// once in shared buffers, without per-vertex snapshots or duplicate CPU index
    /// arrays. These buffers cannot be updated after creation.
    pub fn from_static_drawables(
        device: &wgpu::Device,
        meshes: &[Moc3DrawableMesh],
    ) -> Option<Self> {
        let (vertex_byte_len, index_byte_len) =
            meshes
                .iter()
                .try_fold((0usize, 0usize), |(vertex_total, index_total), mesh| {
                    Some((
                        vertex_total.checked_add(
                            mesh.vertices().len().checked_mul(DrawableVertex::STRIDE)?,
                        )?,
                        index_total.checked_add(
                            mesh.indices()
                                .len()
                                .checked_mul(std::mem::size_of::<u16>())?,
                        )?,
                    ))
                })?;
        let padded_buffer_size = |size: usize| size.max(4).checked_add(3).map(|size| size & !3);
        let vertex_buffer_size = u64::try_from(padded_buffer_size(vertex_byte_len)?).ok()?;
        let index_buffer_size = u64::try_from(padded_buffer_size(index_byte_len)?).ok()?;
        let max_buffer_size = device.limits().max_buffer_size;
        if vertex_buffer_size > max_buffer_size || index_buffer_size > max_buffer_size {
            return None;
        }

        let mut vertex_bytes = Vec::new();
        let mut index_bytes = Vec::new();
        vertex_bytes.try_reserve_exact(vertex_byte_len).ok()?;
        index_bytes.try_reserve_exact(index_byte_len).ok()?;
        let mut layouts = Vec::with_capacity(meshes.len());

        for mesh in meshes {
            let vertex_count = u32::try_from(mesh.vertices().len()).ok()?;
            let index_count = u32::try_from(mesh.indices().len()).ok()?;
            let base_vertex = i32::try_from(vertex_bytes.len() / DrawableVertex::STRIDE).ok()?;
            let first_index = u32::try_from(index_bytes.len() / std::mem::size_of::<u16>()).ok()?;
            let vertex_start = u64::try_from(vertex_bytes.len()).ok()?;
            let index_start = u64::try_from(index_bytes.len()).ok()?;
            append_vertices_from_drawable(mesh, &mut vertex_bytes);
            for index in mesh.indices() {
                index_bytes.extend_from_slice(&index.to_ne_bytes());
            }
            first_index.checked_add(index_count)?;

            let vertex_end = u64::try_from(vertex_bytes.len()).ok()?;
            let index_end = u64::try_from(index_bytes.len()).ok()?;
            layouts.push((
                vertex_start..vertex_end,
                index_start..index_end,
                first_index,
                base_vertex,
                vertex_count,
                index_count,
                DrawableInfo::from_mesh(mesh),
            ));
        }

        let empty_contents = [0u8; 4];
        let vertex_contents: &[u8] = if vertex_bytes.is_empty() {
            &empty_contents
        } else {
            &vertex_bytes
        };
        let index_contents: &[u8] = if index_bytes.is_empty() {
            &empty_contents
        } else {
            &index_bytes
        };
        let vertex_buffer = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("live2d.static.vertices"),
                contents: vertex_contents,
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        let index_buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("live2d.static.indices"),
                contents: index_contents,
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        let drawables = layouts
            .into_iter()
            .map(
                |(
                    vertex_range,
                    index_range,
                    first_index,
                    base_vertex,
                    vertex_count,
                    index_count,
                    info,
                )| {
                    WgpuDrawableBuffers {
                        vertex_buffer: Arc::clone(&vertex_buffer),
                        index_buffer: Arc::clone(&index_buffer),
                        vertex_range,
                        index_range,
                        first_index,
                        base_vertex,
                        vertex_count,
                        index_count,
                        vertex_snapshot: None,
                        indices: None,
                        info,
                    }
                },
            )
            .collect::<Vec<_>>();
        let draw_order_indices = draw_order_indices_from(
            drawables.len(),
            |index| drawables[index].draw_order(),
            |index| drawables[index].render_order(),
        );

        Some(Self {
            render_order_seen: Vec::with_capacity(drawables.len()),
            vertex_upload_bytes: Vec::new(),
            drawables,
            draw_order_indices,
            packed_vertex_buffer: Some(vertex_buffer),
            packed_index_buffer: Some(index_buffer),
            static_geometry: true,
        })
    }

    pub(super) fn packed_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        Some((
            self.packed_vertex_buffer.as_deref()?,
            self.packed_index_buffer.as_deref()?,
        ))
    }

    pub fn drawables(&self) -> &[WgpuDrawableBuffers] {
        &self.drawables
    }

    pub fn drawable_infos(&self) -> Vec<DrawableInfo> {
        self.drawables.iter().map(|d| d.info.clone()).collect()
    }

    pub(crate) fn iter_drawable_infos(&self) -> impl Iterator<Item = &DrawableInfo> {
        self.drawables.iter().map(WgpuDrawableBuffers::info)
    }

    pub(crate) fn drawable_bounds(&self, drawable_index: usize) -> Option<ClippingRect> {
        self.drawables
            .get(drawable_index)
            .and_then(WgpuDrawableBuffers::bounds)
    }

    pub fn draw_order_indices(&self) -> &[usize] {
        &self.draw_order_indices
    }

    pub fn update_drawables(
        &mut self,
        queue: &wgpu::Queue,
        meshes: &[Moc3DrawableMesh],
    ) -> Result<WgpuMeshUpdate, WgpuMeshUpdateError> {
        if self.static_geometry {
            return Err(WgpuMeshUpdateError::StaticGeometry);
        }
        if self.drawables.len() != meshes.len() {
            return Err(WgpuMeshUpdateError::DrawableCount {
                expected: self.drawables.len(),
                actual: meshes.len(),
            });
        }

        for (drawable_index, (drawable, mesh)) in self.drawables.iter().zip(meshes).enumerate() {
            validate_drawable_update(drawable_index, drawable, mesh)?;
        }

        let mut uploads = 0;
        let mut bounds_changed = false;
        let mut visibility_changed = false;
        let mut order_changed = false;
        {
            let (drawables, vertex_upload_bytes) =
                (&mut self.drawables, &mut self.vertex_upload_bytes);
            for (drawable, mesh) in drawables.iter_mut().zip(meshes) {
                let changes = drawable
                    .vertex_snapshot
                    .as_ref()
                    .expect("dynamic drawable has a vertex snapshot")
                    .changes(mesh);
                if changes.vertex_data_changed {
                    encode_vertices_from_drawable(mesh, vertex_upload_bytes);
                    if !vertex_upload_bytes.is_empty() {
                        queue.write_buffer(
                            drawable.vertex_buffer.as_ref(),
                            drawable.vertex_range.start,
                            vertex_upload_bytes,
                        );
                        uploads += 1;
                    }
                    drawable
                        .vertex_snapshot
                        .as_mut()
                        .expect("dynamic drawable has a vertex snapshot")
                        .update(mesh);
                }
                let was_visible = drawable.is_visible();
                let old_bounds = drawable.info.bounds();
                let old_draw_order = drawable.draw_order().to_bits();
                let old_render_order = drawable.render_order();
                drawable.info.update_from_mesh(mesh, changes.bounds_changed);
                let is_visible = drawable.is_visible();
                bounds_changed |= old_bounds != drawable.info.bounds();
                visibility_changed |= was_visible != is_visible;
                order_changed |= old_draw_order != drawable.draw_order().to_bits()
                    || old_render_order != drawable.render_order();
            }
        }
        if order_changed {
            let drawables = &self.drawables;
            draw_order_indices_from_into(
                drawables.len(),
                |index| drawables[index].draw_order(),
                |index| drawables[index].render_order(),
                &mut self.draw_order_indices,
                &mut self.render_order_seen,
            );
        }

        Ok(WgpuMeshUpdate {
            uploaded_drawables: uploads,
            bounds_changed,
            visibility_changed,
        })
    }
}

fn validate_drawable_update(
    drawable_index: usize,
    drawable: &WgpuDrawableBuffers,
    mesh: &Moc3DrawableMesh,
) -> Result<(), WgpuMeshUpdateError> {
    validate_count(
        drawable.vertex_count as usize,
        mesh.vertices().len(),
        WgpuMeshUpdateError::VertexCount {
            drawable_index,
            expected: drawable.vertex_count as usize,
            actual: mesh.vertices().len(),
        },
    )?;
    validate_count(
        drawable.index_count as usize,
        mesh.indices().len(),
        WgpuMeshUpdateError::IndexCount {
            drawable_index,
            expected: drawable.index_count as usize,
            actual: mesh.indices().len(),
        },
    )?;
    validate_unchanged(
        drawable
            .indices
            .as_deref()
            .expect("dynamic drawable has an index snapshot"),
        mesh.indices(),
        WgpuMeshUpdateError::Indices { drawable_index },
    )?;
    validate_unchanged(
        &drawable.texture_index(),
        &mesh.texture_index(),
        WgpuMeshUpdateError::TextureIndex {
            drawable_index,
            expected: drawable.texture_index(),
            actual: mesh.texture_index(),
        },
    )?;
    validate_unchanged(
        &drawable.blend_mode(),
        &mesh.blend_mode(),
        WgpuMeshUpdateError::BlendMode {
            drawable_index,
            expected: drawable.blend_mode(),
            actual: mesh.blend_mode(),
        },
    )?;
    validate_unchanged(
        drawable.masks(),
        mesh.masks(),
        WgpuMeshUpdateError::Masks { drawable_index },
    )?;
    validate_unchanged(
        &drawable.inverted_mask(),
        &mesh.is_inverted_mask(),
        WgpuMeshUpdateError::InvertedMask {
            drawable_index,
            expected: drawable.inverted_mask(),
            actual: mesh.is_inverted_mask(),
        },
    )?;

    Ok(())
}

fn validate_count(
    expected: usize,
    actual: usize,
    error: WgpuMeshUpdateError,
) -> Result<(), WgpuMeshUpdateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(error)
    }
}

fn validate_unchanged<T: PartialEq + ?Sized>(
    expected: &T,
    actual: &T,
    error: WgpuMeshUpdateError,
) -> Result<(), WgpuMeshUpdateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(error)
    }
}

pub fn create_wgpu_drawable_buffers(
    device: &wgpu::Device,
    mesh: &Moc3DrawableMesh,
) -> Option<WgpuDrawableBuffers> {
    let mut vertex_bytes = Vec::new();
    encode_vertices_from_drawable(mesh, &mut vertex_bytes);
    let index_bytes = encode_indices(mesh.indices());
    let vertex_count = u32::try_from(mesh.vertices().len()).ok()?;
    let index_count = u32::try_from(mesh.indices().len()).ok()?;

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("live2d.drawable.vertices"),
        contents: &vertex_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("live2d.drawable.indices"),
        contents: &index_bytes,
        usage: wgpu::BufferUsages::INDEX,
    });

    Some(WgpuDrawableBuffers {
        vertex_range: 0..u64::try_from(vertex_bytes.len()).ok()?,
        index_range: 0..u64::try_from(index_bytes.len()).ok()?,
        vertex_buffer: Arc::new(vertex_buffer),
        index_buffer: Arc::new(index_buffer),
        first_index: 0,
        base_vertex: 0,
        vertex_count,
        index_count,
        vertex_snapshot: Some(DrawableVertexSnapshot::from_mesh(mesh)),
        indices: Some(mesh.indices().to_vec()),
        info: DrawableInfo::from_mesh(mesh),
    })
}

fn color_bits(values: [f32; 3]) -> [u32; 3] {
    values.map(f32::to_bits)
}
