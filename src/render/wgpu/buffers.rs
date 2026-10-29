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
    /// Returns the GPU resource containing this drawable's vertices. Mesh buffers
    /// share this resource with other drawables; use `vertex_buffer_slice()` for
    /// this drawable's range.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    /// Returns the vertex range occupied by this drawable.
    pub fn vertex_buffer_slice(&self) -> wgpu::BufferSlice<'_> {
        self.vertex_buffer.slice(self.vertex_range.clone())
    }

    /// Returns the shared GPU resource containing this drawable's indices. Use
    /// `index_buffer_slice()` for this drawable's range.
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
    upload_calls: usize,
    bounds_changed: bool,
    visibility_changed: bool,
}

impl WgpuMeshUpdate {
    pub fn uploaded_drawables(&self) -> usize {
        self.uploaded_drawables
    }

    /// Number of queue writes used for the changed drawable vertices.
    pub fn upload_calls(&self) -> usize {
        self.upload_calls
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
    /// Packs animated geometry into two shared allocations, retaining the same
    /// per-drawable snapshots and update validation as standalone buffers.
    pub fn from_drawables(device: &wgpu::Device, meshes: &[Moc3DrawableMesh]) -> Option<Self> {
        Self::from_packed_drawables(device, meshes, false).or_else(|| {
            // Very large models can exceed a single buffer's device limit even
            // though every individual drawable fits. Preserve that support.
            let drawables = meshes
                .iter()
                .map(|mesh| create_wgpu_drawable_buffers(device, mesh))
                .collect::<Option<Vec<_>>>()?;
            let draw_order_indices = draw_order_indices_from(
                drawables.len(),
                |i| drawables[i].draw_order(),
                |i| drawables[i].render_order(),
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
        Self::from_packed_drawables(device, meshes, true)
    }

    fn from_packed_drawables(
        device: &wgpu::Device,
        meshes: &[Moc3DrawableMesh],
        static_geometry: bool,
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
                label: Some("live2d.packed.vertices"),
                contents: vertex_contents,
                usage: if static_geometry {
                    wgpu::BufferUsages::VERTEX
                } else {
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
                },
            },
        ));
        let index_buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("live2d.packed.indices"),
                contents: index_contents,
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        let drawables = layouts
            .into_iter()
            .zip(meshes)
            .map(
                |(
                    (
                        vertex_range,
                        index_range,
                        first_index,
                        base_vertex,
                        vertex_count,
                        index_count,
                        info,
                    ),
                    mesh,
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
                        vertex_snapshot: (!static_geometry)
                            .then(|| DrawableVertexSnapshot::from_mesh(mesh)),
                        indices: (!static_geometry).then(|| mesh.indices().to_vec()),
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
            static_geometry,
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
        let mut upload_calls = 0;
        let mut upload_start = 0;
        let mut bounds_changed = false;
        let mut visibility_changed = false;
        let mut order_changed = false;
        {
            let (drawables, vertex_upload_bytes) =
                (&mut self.drawables, &mut self.vertex_upload_bytes);
            vertex_upload_bytes.clear();
            for (drawable, mesh) in drawables.iter_mut().zip(meshes) {
                let changes = drawable
                    .vertex_snapshot
                    .as_ref()
                    .expect("dynamic drawable has a vertex snapshot")
                    .changes(mesh);
                if changes.vertex_data_changed {
                    if !mesh.vertices().is_empty() {
                        if self.packed_vertex_buffer.is_some() {
                            // Adjacent dirty drawables occupy consecutive bytes
                            // in the shared buffer. Encode them into one reusable
                            // staging span and issue one queue write for the run.
                            if vertex_upload_bytes.is_empty() {
                                upload_start = drawable.vertex_range.start;
                            }
                            append_vertices_from_drawable(mesh, vertex_upload_bytes);
                        } else {
                            encode_vertices_from_drawable(mesh, vertex_upload_bytes);
                            queue.write_buffer(
                                drawable.vertex_buffer.as_ref(),
                                drawable.vertex_range.start,
                                vertex_upload_bytes,
                            );
                            upload_calls += 1;
                        }
                        uploads += 1;
                    }
                    drawable
                        .vertex_snapshot
                        .as_mut()
                        .expect("dynamic drawable has a vertex snapshot")
                        .update(mesh);
                } else if let Some(buffer) = self.packed_vertex_buffer.as_deref()
                    && !vertex_upload_bytes.is_empty()
                    && !mesh.vertices().is_empty()
                {
                    // Preserve sparse updates: never encode or upload an
                    // unchanged drawable just to bridge two dirty spans.
                    queue.write_buffer(buffer, upload_start, vertex_upload_bytes);
                    upload_calls += 1;
                    vertex_upload_bytes.clear();
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
            if let Some(buffer) = self.packed_vertex_buffer.as_deref()
                && !vertex_upload_bytes.is_empty()
            {
                queue.write_buffer(buffer, upload_start, vertex_upload_bytes);
                upload_calls += 1;
            }
            vertex_upload_bytes.clear();
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
            upload_calls,
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

#[cfg(test)]
mod packed_tests {
    use super::*;
    use crate::{core::Matrix44, moc3::Moc3DrawableVertex, render::wgpu::*};

    // Keep the old allocation path as an independent rendering reference.
    fn unpacked(device: &wgpu::Device, meshes: &[Moc3DrawableMesh]) -> WgpuMeshBuffers {
        let drawables = meshes
            .iter()
            .map(|mesh| create_wgpu_drawable_buffers(device, mesh).unwrap())
            .collect();
        WgpuMeshBuffers {
            drawables,
            draw_order_indices: draw_order_indices_from(
                meshes.len(),
                |i| meshes[i].draw_order(),
                |i| meshes[i].render_order(),
            ),
            render_order_seen: Vec::new(),
            vertex_upload_bytes: Vec::new(),
            packed_vertex_buffer: None,
            packed_index_buffer: None,
            static_geometry: false,
        }
    }

    fn scene(frame: usize) -> Vec<Moc3DrawableMesh> {
        (0..4)
            .map(|i| {
                let shift = frame as f32 * 0.07;
                let x = -0.8 + i as f32 * 0.35 + shift;
                Moc3DrawableMesh::from_parts(
                    i % 2,
                    0,
                    if i == 1 || (i == 3 && frame == 0) {
                        0.0
                    } else {
                        0.4 + shift
                    },
                    if frame == 2 { (4 - i) as f32 } else { i as f32 },
                    vec![
                        Moc3DrawableVertex::new([x, -0.8], [0.0, 1.0]),
                        Moc3DrawableVertex::new([x + 0.6, -0.8 + shift], [1.0, 1.0]),
                        Moc3DrawableVertex::new([x + 0.3, 0.9], [0.5, shift]),
                    ],
                    // Odd index counts exercise non-four-byte-aligned index ranges.
                    vec![0, 1, 2],
                    if i == 0 { vec![1] } else { vec![] },
                )
            })
            .collect()
    }

    #[test]
    fn oversized_packed_buffer_keeps_the_per_drawable_fallback() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_buffer_size: 256,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut buffers = WgpuMeshBuffers::from_drawables(&device, &scene(0)).unwrap();
        assert!(buffers.packed_buffers().is_none());
        let update = buffers.update_drawables(&queue, &scene(1)).unwrap();
        assert_eq!(update.uploaded_drawables(), 4);
        assert_eq!(update.upload_calls(), 4);
    }

    #[test]
    fn packed_uploads_coalesce_dirty_runs_and_skip_unchanged_meshes() {
        let (device, queue) = wgpu::Device::noop(&Default::default());
        let mut meshes = scene(0);
        let mut buffers = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
        let unchanged = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(unchanged.uploaded_drawables(), 0);
        assert_eq!(unchanged.upload_calls(), 0);

        meshes = scene(1);
        let all = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(all.uploaded_drawables(), 4);
        assert_eq!(all.upload_calls(), 1);

        // Separated dirty ranges must use their actual buffer offsets, without
        // uploading or corrupting the unchanged drawable between them.
        meshes[0].set_opacity(0.75);
        meshes[2].vertices_mut()[0].set_position([0.25, 0.5]);
        meshes[3].set_multiply_color([0.5, 0.8, 0.7]);
        let sparse = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(sparse.uploaded_drawables(), 3);
        assert_eq!(sparse.upload_calls(), 2);
        assert!(sparse.bounds_changed());
        assert_eq!(
            buffers
                .update_drawables(&queue, &meshes)
                .unwrap()
                .upload_calls(),
            0
        );
    }

    fn pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &WgpuLive2dRenderer,
        buffers: &WgpuMeshBuffers,
        textures: &[WgpuTexture],
    ) -> Vec<u8> {
        let mut plan = WgpuClippingPlan::from_mesh_buffers(buffers);
        plan.prepare_single_texture_masks(buffers).unwrap();
        let clipping = renderer.create_clipping_resources(device, &plan).unwrap();
        let mask = renderer.create_mask_render_target(device, 256).unwrap();
        let transform = renderer.create_transform(device, &Matrix44::identity());
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("packed-regression.target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed-regression.readback"),
            size: 64 * 64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        for mask_pass in [true, false] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if mask_pass { mask.view() } else { &view },
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if mask_pass {
                renderer.draw_masks_with_textures(&mut pass, buffers, &clipping, textures)
            } else {
                renderer.draw_with_textures_clipping_and_transform(
                    &mut pass, buffers, textures, &clipping, &mask, &transform,
                )
            }
            .unwrap();
        }
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let result = readback.slice(..).get_mapped_range().unwrap().to_vec();
        readback.unmap();
        result
    }

    #[test]
    #[ignore = "requires a real GPU; checks rendered bytes against separate drawable buffers"]
    fn packed_animation_matches_unpacked_pixels() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let renderer = WgpuLive2dRenderer::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let textures = [
            renderer
                .create_rgba8_texture(
                    &device,
                    &queue,
                    2,
                    2,
                    &[
                        255, 32, 16, 255, 32, 255, 16, 127, 16, 32, 255, 80, 255, 255, 255, 0,
                    ],
                )
                .unwrap(),
            renderer
                .create_rgba8_texture(&device, &queue, 1, 1, &[32, 192, 255, 200])
                .unwrap(),
        ];
        let meshes = scene(0);
        let mut packed = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
        let mut reference = unpacked(&device, &meshes);
        assert!(Arc::ptr_eq(
            &packed.drawables[0].vertex_buffer,
            &packed.drawables[3].vertex_buffer
        ));
        assert!(Arc::ptr_eq(
            &packed.drawables[0].index_buffer,
            &packed.drawables[3].index_buffer
        ));
        for frame in 0..6 {
            let mut meshes = scene(frame.min(2));
            if frame >= 3 {
                meshes[0].set_opacity(0.6);
                meshes[2].vertices_mut()[0].set_position([0.25, 0.5]);
                meshes[3].set_multiply_color([0.5, 0.8, 0.7]);
                meshes[3].set_screen_color([0.1, 0.3, 0.2]);
            }
            if frame == 5 {
                // A zero-vertex drawable has no buffer gap and must not break
                // the offsets of a dirty span on either side.
                meshes[1] = Moc3DrawableMesh::from_parts(1, 0, 0.0, 1.0, vec![], vec![], vec![]);
                packed = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
                reference = unpacked(&device, &meshes);
                meshes[0].set_opacity(0.9);
                meshes[2].set_opacity(0.9);
            }
            packed.update_drawables(&queue, &meshes).unwrap();
            reference.update_drawables(&queue, &meshes).unwrap();
            let expected = pixels(&device, &queue, &renderer, &reference, &textures);
            assert!(
                expected
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| pixel[3] != 0)
            );
            assert_eq!(
                pixels(&device, &queue, &renderer, &packed, &textures),
                expected,
                "rendered pixels differ at frame {frame}"
            );
        }
    }
}
