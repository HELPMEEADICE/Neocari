use std::{collections::HashMap, ops::Range, sync::Arc};

use wgpu::util::DeviceExt;

use crate::moc3::{Moc3DrawableBlendMode, Moc3DrawableMesh, Moc3DrawableVertex};
use crate::render::common::{
    ClippingRect, DrawableInfo, DrawableVertex, draw_order_indices_from,
    draw_order_indices_from_into, encode_indices, encode_vertices_from_drawable,
};

const GEOMETRY_STRIDE: usize = size_of::<Moc3DrawableVertex>();
const MATERIAL_STRIDE: usize = 28;
pub(super) const INDIRECT_STRIDE: u64 = size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
pub(super) const INDIRECT_FEATURES: wgpu::Features =
    wgpu::Features::INDIRECT_FIRST_INSTANCE.union(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

/// Position/UV stream used by packed mesh buffers (vertex slot 0).
pub fn packed_geometry_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2, 1 => Float32x2
    ];
    wgpu::VertexBufferLayout {
        array_stride: GEOMETRY_STRIDE as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

/// One opacity/multiply/screen record per drawable (instance slot 1).
pub fn packed_material_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        2 => Float32, 3 => Float32x3, 4 => Float32x3
    ];
    wgpu::VertexBufferLayout {
        array_stride: MATERIAL_STRIDE as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRIBUTES,
    }
}

fn geometry_bytes(mesh: &Moc3DrawableMesh) -> &[u8] {
    bytemuck::cast_slice(mesh.vertices())
}

fn material_record(mesh: &Moc3DrawableMesh) -> [u8; MATERIAL_STRIDE] {
    let [mr, mg, mb] = mesh.multiply_color();
    let [sr, sg, sb] = mesh.screen_color();
    float_record([mesh.opacity(), mr, mg, mb, sr, sg, sb])
}

fn float_record<const N: usize>(values: impl IntoIterator<Item = f32>) -> [u8; N] {
    let mut bytes = [0; N];
    for (word, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        *word = value.to_ne_bytes();
    }
    bytes
}

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
    indices: Option<Arc<[u16]>>,
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
    material_changed: bool,
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
                material_changed: true,
                bounds_changed: true,
            };
        }

        let material_changed = self.opacity_bits != mesh.opacity().to_bits()
            || self.multiply_bits != color_bits(mesh.multiply_color())
            || self.screen_bits != color_bits(mesh.screen_color());
        let mut vertex_data_changed = false;
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
            material_changed,
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
        self.update_material(mesh);
    }

    fn update_material(&mut self, mesh: &Moc3DrawableMesh) {
        self.opacity_bits = mesh.opacity().to_bits();
        self.multiply_bits = color_bits(mesh.multiply_color());
        self.screen_bits = color_bits(mesh.screen_color());
    }
}

impl WgpuDrawableBuffers {
    /// Returns the GPU resource containing this drawable's vertices. Mesh buffers
    /// share this resource with other drawables; use `vertex_buffer_slice()` for
    /// this drawable's range. Packed meshes store only position/UV here; use
    /// `WgpuMeshBuffers::material_buffer()` with the packed pipeline for colors.
    /// Standalone drawable buffers retain the interleaved 44-byte layout.
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
    packed_upload: Option<PackedUpload>,
    packed_vertex_buffer: Option<Arc<wgpu::Buffer>>,
    packed_index_buffer: Option<Arc<wgpu::Buffer>>,
    packed_material_buffer: Option<wgpu::Buffer>,
    indirect: Option<IndirectDraws>,
    static_geometry: bool,
    has_instance_candidates: bool,
}

#[derive(Debug)]
struct IndirectDraws {
    buffer: wgpu::Buffer,
    slots: Vec<u32>,
}

fn encode_indirect_draws(
    drawables: &[WgpuDrawableBuffers],
    order: &[usize],
    slots: &mut [u32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(order.len() * INDIRECT_STRIDE as usize);
    for (slot, &index) in order.iter().enumerate() {
        let drawable = &drawables[index];
        slots[index] = slot as u32;
        for word in [
            drawable.index_count,
            1,
            drawable.first_index,
            drawable.base_vertex as u32,
            index as u32,
        ] {
            bytes.extend_from_slice(&word.to_ne_bytes());
        }
    }
    bytes
}

#[derive(Debug)]
struct PackedUpload {
    geometry: Vec<u8>,
    materials: Vec<u8>,
}

struct PackedGeometryLayout {
    vertex_range: Range<u64>,
    index_range: Range<u64>,
    first_index: u32,
    base_vertex: i32,
    vertex_count: u32,
    index_count: u32,
}

#[derive(Default)]
struct UploadSpan(Range<usize>);

impl UploadSpan {
    fn include(
        &mut self,
        range: Range<usize>,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        bytes: &[u8],
        stats: &mut UploadStats,
    ) {
        if !self.0.is_empty() && self.0.end == range.start {
            self.0.end = range.end;
        } else {
            self.flush(queue, buffer, bytes, stats);
            self.0 = range;
        }
    }

    fn flush(
        &mut self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        bytes: &[u8],
        stats: &mut UploadStats,
    ) {
        write_upload(
            queue,
            buffer,
            self.0.start as u64,
            &bytes[self.0.clone()],
            stats,
        );
        self.0 = 0..0;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WgpuMeshUpdate {
    uploaded_drawables: usize,
    upload_calls: usize,
    uploaded_bytes: usize,
    geometry_uploads: usize,
    material_uploads: usize,
    bounds_changed: bool,
    visibility_changed: bool,
}

impl WgpuMeshUpdate {
    pub fn uploaded_drawables(&self) -> usize {
        self.uploaded_drawables
    }

    /// Number of queue writes used for changed geometry and material records.
    pub fn upload_calls(&self) -> usize {
        self.upload_calls
    }

    pub fn uploaded_bytes(&self) -> usize {
        self.uploaded_bytes
    }

    pub fn geometry_uploads(&self) -> usize {
        self.geometry_uploads
    }

    pub fn material_uploads(&self) -> usize {
        self.material_uploads
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
    /// Packs animated geometry and instance materials into three shared allocations, retaining the same
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
                packed_upload: None,
                drawables,
                draw_order_indices,
                packed_vertex_buffer: None,
                packed_index_buffer: None,
                packed_material_buffer: None,
                indirect: None,
                static_geometry: false,
                has_instance_candidates: false,
            })
        })
    }

    /// Creates compact GPU buffers for immutable drawable geometry, such as the
    /// default-pose meshes produced by CubismV2. Bit-identical geometry shares
    /// vertices and indices across drawables. These buffers do not retain CPU
    /// geometry snapshots and cannot be updated after creation.
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
        // Indices stay immutable across animation. Each animated drawable keeps
        // its own vertex range so a later pose can diverge without reallocating.
        let mut vertex_offsets = HashMap::new();
        let mut index_offsets = HashMap::new();
        let mut vertex_byte_len = 0usize;
        let mut index_byte_len = 0usize;
        let mut layouts = Vec::with_capacity(meshes.len());
        for mesh in meshes {
            let vertices = geometry_bytes(mesh);
            let vertex_start = if static_geometry {
                *vertex_offsets.entry(vertices).or_insert(vertex_byte_len)
            } else {
                vertex_byte_len
            };
            let index_start = *index_offsets
                .entry(mesh.indices())
                .or_insert(index_byte_len);
            let vertex_end = vertex_start.checked_add(vertices.len())?;
            let index_end = index_start.checked_add(size_of_val(mesh.indices()))?;
            let first_index = u32::try_from(index_start / size_of::<u16>()).ok()?;
            let index_count = u32::try_from(mesh.indices().len()).ok()?;
            first_index.checked_add(index_count)?;
            layouts.push(PackedGeometryLayout {
                vertex_range: vertex_start as u64..vertex_end as u64,
                index_range: index_start as u64..index_end as u64,
                base_vertex: i32::try_from(vertex_start / GEOMETRY_STRIDE).ok()?,
                first_index,
                vertex_count: u32::try_from(mesh.vertices().len()).ok()?,
                index_count,
            });
            vertex_byte_len = vertex_byte_len.max(vertex_end);
            index_byte_len = index_byte_len.max(index_end);
        }
        let padded_buffer_size = |size: usize| size.max(4).checked_add(3).map(|size| size & !3);
        let vertex_buffer_size = u64::try_from(padded_buffer_size(vertex_byte_len)?).ok()?;
        let index_buffer_size = u64::try_from(padded_buffer_size(index_byte_len)?).ok()?;
        let max_buffer_size = device.limits().max_buffer_size;
        let material_byte_len = meshes.len().checked_mul(MATERIAL_STRIDE)?;
        let material_buffer_size = u64::try_from(padded_buffer_size(material_byte_len)?).ok()?;
        u32::try_from(meshes.len()).ok()?;
        if vertex_buffer_size > max_buffer_size
            || index_buffer_size > max_buffer_size
            || material_buffer_size > max_buffer_size
        {
            return None;
        }

        let mut vertex_bytes = Vec::new();
        let mut index_bytes = Vec::new();
        let mut material_bytes = Vec::new();
        vertex_bytes.try_reserve_exact(vertex_byte_len).ok()?;
        index_bytes.try_reserve_exact(index_byte_len).ok()?;
        material_bytes.try_reserve_exact(material_byte_len).ok()?;

        for (mesh, layout) in meshes.iter().zip(&layouts) {
            material_bytes.extend_from_slice(&material_record(mesh));
            if layout.vertex_range.start == vertex_bytes.len() as u64 {
                vertex_bytes.extend_from_slice(geometry_bytes(mesh));
            }
            if layout.index_range.start == index_bytes.len() as u64 {
                index_bytes.extend_from_slice(bytemuck::cast_slice(mesh.indices()));
            }
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
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("live2d.packed.materials"),
            contents: if material_bytes.is_empty() {
                &empty_contents
            } else {
                &material_bytes
            },
            usage: if static_geometry {
                wgpu::BufferUsages::VERTEX
            } else {
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
            },
        });
        let drawables = layouts
            .into_iter()
            .zip(meshes)
            .map(|(layout, mesh)| WgpuDrawableBuffers {
                vertex_buffer: Arc::clone(&vertex_buffer),
                index_buffer: Arc::clone(&index_buffer),
                vertex_range: layout.vertex_range,
                index_range: layout.index_range,
                first_index: layout.first_index,
                base_vertex: layout.base_vertex,
                vertex_count: layout.vertex_count,
                index_count: layout.index_count,
                vertex_snapshot: None,
                indices: (!static_geometry).then(|| Arc::clone(mesh.shared_indices())),
                info: DrawableInfo::from_mesh(mesh),
            })
            .collect::<Vec<_>>();
        let draw_order_indices = draw_order_indices_from(
            drawables.len(),
            |index| drawables[index].draw_order(),
            |index| drawables[index].render_order(),
        );
        let indirect_size = drawables.len() as u64 * INDIRECT_STRIDE;
        let indirect = (drawables.len() > 1
            && indirect_size <= max_buffer_size
            && usize::try_from(indirect_size).is_ok()
            && device.features().contains(INDIRECT_FEATURES))
        .then(|| {
            let mut slots = vec![0; drawables.len()];
            let bytes = encode_indirect_draws(&drawables, &draw_order_indices, &mut slots);
            IndirectDraws {
                buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("live2d.packed.indirect"),
                    contents: &bytes,
                    usage: if static_geometry {
                        wgpu::BufferUsages::INDIRECT
                    } else {
                        wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST
                    },
                }),
                slots,
            }
        });

        let has_instance_candidates = drawables.windows(2).any(|pair| {
            !pair[0].is_empty() && pair[0].draw_index_range() == pair[1].draw_index_range()
        });
        Some(Self {
            has_instance_candidates,
            render_order_seen: Vec::with_capacity(drawables.len()),
            vertex_upload_bytes: Vec::new(),
            packed_upload: (!static_geometry).then_some(PackedUpload {
                geometry: vertex_bytes,
                materials: material_bytes,
            }),
            drawables,
            draw_order_indices,
            packed_vertex_buffer: Some(vertex_buffer),
            packed_index_buffer: Some(index_buffer),
            packed_material_buffer: Some(material_buffer),
            indirect,
            static_geometry,
        })
    }

    pub(super) fn packed_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        Some((
            self.packed_vertex_buffer.as_deref()?,
            self.packed_index_buffer.as_deref()?,
        ))
    }

    /// Packed opacity/color instance stream. `None` selects legacy interleaved buffers.
    pub fn material_buffer(&self) -> Option<&wgpu::Buffer> {
        self.packed_material_buffer.as_ref()
    }

    pub(super) fn instance_range(&self, index: usize) -> Range<u32> {
        if self.packed_material_buffer.is_some() {
            index as u32..index as u32 + 1
        } else {
            0..1
        }
    }

    pub(super) fn indirect_slot(&self, index: usize) -> Option<u32> {
        self.indirect.as_ref().map(|draws| draws.slots[index])
    }

    pub(super) fn indirect_buffer(&self) -> Option<&wgpu::Buffer> {
        self.indirect.as_ref().map(|draws| &draws.buffer)
    }

    pub(super) fn can_batch_draws(&self) -> bool {
        self.has_instance_candidates || self.indirect.is_some()
    }

    /// Called only for consecutive material slots with compatible pass state.
    /// Compare the uploaded pose bit-for-bit before reusing the first mesh.
    pub(super) fn same_instance_geometry(
        &self,
        first: &WgpuDrawableBuffers,
        next: &WgpuDrawableBuffers,
    ) -> bool {
        if first.draw_index_range() != next.draw_index_range()
            || first.vertex_count != next.vertex_count
        {
            return false;
        }
        if first.base_vertex == next.base_vertex {
            return true;
        }
        self.packed_upload.as_ref().is_some_and(|upload| {
            let vertices = |d: &WgpuDrawableBuffers| {
                &upload.geometry[d.vertex_range.start as usize..d.vertex_range.end as usize]
            };
            vertices(first) == vertices(next)
        })
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
        let mut upload_stats = UploadStats::default();
        let mut geometry_uploads = 0;
        let mut material_uploads = 0;
        let mut geometry_span = UploadSpan::default();
        let mut material_span = UploadSpan::default();
        let mut bounds_changed = false;
        let mut visibility_changed = false;
        let mut order_changed = false;
        for (index, (drawable, mesh)) in self.drawables.iter_mut().zip(meshes).enumerate() {
            let (geometry_dirty, material_dirty, update_bounds) =
                if let Some(upload) = &mut self.packed_upload {
                    let range =
                        drawable.vertex_range.start as usize..drawable.vertex_range.end as usize;
                    let next = geometry_bytes(mesh);
                    let cached = &mut upload.geometry[range.clone()];
                    let geometry_dirty = cached != next;
                    if geometry_dirty {
                        cached.copy_from_slice(next);
                        geometry_span.include(
                            range,
                            queue,
                            &drawable.vertex_buffer,
                            &upload.geometry,
                            &mut upload_stats,
                        );
                    }

                    let start = index * MATERIAL_STRIDE;
                    let range = start..start + MATERIAL_STRIDE;
                    let record = material_record(mesh);
                    let material_dirty =
                        !mesh.vertices().is_empty() && upload.materials[range.clone()] != record;
                    if material_dirty {
                        upload.materials[range.clone()].copy_from_slice(&record);
                        material_span.include(
                            range,
                            queue,
                            self.packed_material_buffer.as_ref().unwrap(),
                            &upload.materials,
                            &mut upload_stats,
                        );
                    }
                    (geometry_dirty, material_dirty, geometry_dirty)
                } else {
                    let snapshot = drawable.vertex_snapshot.as_mut().unwrap();
                    let changes = snapshot.changes(mesh);
                    let dirty = (changes.vertex_data_changed || changes.material_changed)
                        && !mesh.vertices().is_empty();
                    if dirty {
                        snapshot.update(mesh);
                        encode_vertices_from_drawable(mesh, &mut self.vertex_upload_bytes);
                        write_upload(
                            queue,
                            &drawable.vertex_buffer,
                            drawable.vertex_range.start,
                            &self.vertex_upload_bytes,
                            &mut upload_stats,
                        );
                    }
                    (dirty, false, changes.bounds_changed)
                };
            geometry_uploads += usize::from(geometry_dirty);
            material_uploads += usize::from(material_dirty);
            uploads += usize::from(geometry_dirty || material_dirty);

            let was_visible = drawable.is_visible();
            let old_bounds = drawable.info.bounds();
            let old_draw_order = crate::core::draw_order_from_raw(drawable.draw_order());
            let old_render_order = drawable.render_order();
            drawable.info.update_dynamic_fields(mesh, update_bounds);
            bounds_changed |= old_bounds != drawable.info.bounds();
            visibility_changed |= was_visible != drawable.is_visible();
            order_changed |= old_draw_order
                != crate::core::draw_order_from_raw(drawable.draw_order())
                || old_render_order != drawable.render_order();
        }
        if let Some(upload) = &self.packed_upload {
            geometry_span.flush(
                queue,
                self.packed_vertex_buffer.as_deref().unwrap(),
                &upload.geometry,
                &mut upload_stats,
            );
            material_span.flush(
                queue,
                self.packed_material_buffer.as_ref().unwrap(),
                &upload.materials,
                &mut upload_stats,
            );
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
            if let Some(draws) = &mut self.indirect
                && self
                    .draw_order_indices
                    .iter()
                    .enumerate()
                    .any(|(slot, &index)| draws.slots[index] as usize != slot)
            {
                let bytes =
                    encode_indirect_draws(drawables, &self.draw_order_indices, &mut draws.slots);
                queue.write_buffer(&draws.buffer, 0, &bytes);
            }
        }

        Ok(WgpuMeshUpdate {
            uploaded_drawables: uploads,
            upload_calls: upload_stats.calls,
            uploaded_bytes: upload_stats.bytes,
            geometry_uploads,
            material_uploads,
            bounds_changed,
            visibility_changed,
        })
    }
}

#[derive(Default)]
struct UploadStats {
    calls: usize,
    bytes: usize,
}

fn write_upload(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    bytes: &[u8],
    stats: &mut UploadStats,
) {
    if !bytes.is_empty() {
        queue.write_buffer(buffer, offset, bytes);
        stats.calls += 1;
        stats.bytes += bytes.len();
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
    let indices = drawable.indices.as_ref().unwrap();
    if !Arc::ptr_eq(indices, mesh.shared_indices()) && indices.as_ref() != mesh.indices() {
        return Err(WgpuMeshUpdateError::Indices { drawable_index });
    }
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
        indices: Some(Arc::clone(mesh.shared_indices())),
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
            packed_upload: None,
            packed_vertex_buffer: None,
            packed_index_buffer: None,
            packed_material_buffer: None,
            indirect: None,
            static_geometry: false,
            has_instance_candidates: false,
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
                max_buffer_size: 160,
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
        assert_eq!(all.upload_calls(), 3);
        assert_eq!(all.geometry_uploads(), 4);
        assert_eq!(all.material_uploads(), 3);
        assert_eq!(
            all.uploaded_bytes(),
            4 * 3 * GEOMETRY_STRIDE + 3 * MATERIAL_STRIDE
        );

        // Separated dirty ranges must use their actual buffer offsets, without
        // uploading or corrupting the unchanged drawable between them.
        meshes[0].set_opacity(0.75);
        meshes[2].vertices_mut()[0].set_position([0.25, 0.5]);
        meshes[3].set_multiply_color([0.5, 0.8, 0.7]);
        let sparse = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(sparse.uploaded_drawables(), 3);
        assert_eq!(sparse.upload_calls(), 3);
        assert_eq!(sparse.geometry_uploads(), 1);
        assert_eq!(sparse.material_uploads(), 2);
        assert_eq!(
            sparse.uploaded_bytes(),
            3 * GEOMETRY_STRIDE + 2 * MATERIAL_STRIDE
        );
        assert!(sparse.bounds_changed());
        assert_eq!(
            buffers
                .update_drawables(&queue, &meshes)
                .unwrap()
                .upload_calls(),
            0
        );
    }

    #[test]
    fn packed_material_changes_never_upload_geometry() {
        let (device, queue) = wgpu::Device::noop(&Default::default());
        let mut meshes = scene(0);
        let mut buffers = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
        assert_eq!(buffers.drawables[0].vertex_buffer().size(), 4 * 3 * 16);
        assert_eq!(buffers.material_buffer().unwrap().size(), 4 * 28);

        meshes[0].set_opacity(0.8);
        meshes[0].set_multiply_color([0.1, 0.2, 0.3]);
        meshes[0].set_screen_color([0.3, 0.2, 0.1]);
        let update = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(update.uploaded_drawables(), 1);
        assert_eq!(update.geometry_uploads(), 0);
        assert_eq!(update.material_uploads(), 1);
        assert_eq!(update.upload_calls(), 1);
        assert_eq!(update.uploaded_bytes(), 28);
        assert!(!update.bounds_changed());
        assert_eq!(
            buffers
                .update_drawables(&queue, &meshes)
                .unwrap()
                .uploaded_bytes(),
            0
        );

        meshes[0].vertices_mut()[0].set_position([0.5, 0.25]);
        let update = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(update.geometry_uploads(), 1);
        assert_eq!(update.material_uploads(), 0);
        assert_eq!(update.uploaded_bytes(), 3 * 16);
        assert_eq!(update.upload_calls(), 1);
        assert!(update.bounds_changed());
    }

    #[test]
    fn animated_instances_split_and_rejoin_as_the_pose_changes() {
        let (device, queue) = wgpu::Device::noop(&Default::default());
        let mut meshes = vec![scene(0).remove(2); 2];
        meshes[1].set_opacity(0.75);
        let mut buffers = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
        let can_instance =
            |b: &WgpuMeshBuffers| b.same_instance_geometry(&b.drawables[0], &b.drawables[1]);
        assert!(buffers.indirect.is_none());
        assert!(buffers.can_batch_draws());
        assert!(can_instance(&buffers));
        // Three u16 indices plus wgpu's alignment padding, shared by both meshes.
        assert_eq!(buffers.drawables[0].index_buffer().size(), 8);

        let original = meshes[1].vertices()[0];
        meshes[1].vertices_mut()[0] = Moc3DrawableVertex::new(original.position(), [0.5, 0.5]);
        let update = buffers.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(update.geometry_uploads(), 1);
        assert!(!update.bounds_changed());
        assert!(!can_instance(&buffers));
        meshes[1].vertices_mut()[0] = original;
        buffers.update_drawables(&queue, &meshes).unwrap();
        assert!(can_instance(&buffers));
    }

    #[test]
    fn static_meshes_share_geometry_even_when_the_combined_input_exceeds_the_buffer_limit() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_buffer_size: 64,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut meshes = vec![scene(0).remove(2); 2];
        meshes[1].set_opacity(0.75);
        let buffers = WgpuMeshBuffers::from_static_drawables(&device, &meshes).unwrap();
        let drawables = buffers.drawables();
        assert_eq!(
            drawables[0].vertex_buffer().size(),
            3 * GEOMETRY_STRIDE as u64
        );
        assert_eq!(drawables[0].base_vertex(), drawables[1].base_vertex());
        assert_eq!(
            drawables[0].draw_index_range(),
            drawables[1].draw_index_range()
        );
        assert_eq!(
            buffers.material_buffer().unwrap().size(),
            2 * MATERIAL_STRIDE as u64
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
        let indirect_supported = adapter.features().contains(INDIRECT_FEATURES);
        eprintln!(
            "packed test adapter: {:?}, indirect batching: {indirect_supported}",
            adapter.get_info()
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: if indirect_supported {
                INDIRECT_FEATURES
            } else {
                wgpu::Features::empty()
            },
            ..Default::default()
        }))
        .unwrap();
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

        let make_mesh = |x: f32, order: f32, multiply| {
            let mut mesh = Moc3DrawableMesh::from_parts(
                0,
                0,
                0.65,
                order,
                vec![
                    Moc3DrawableVertex::new([x, -0.8], [0.0, 1.0]),
                    Moc3DrawableVertex::new([x + 1.2, -0.8], [1.0, 1.0]),
                    Moc3DrawableVertex::new([x + 0.6, 0.9], [0.5, 0.0]),
                ],
                vec![0, 1, 2],
                vec![],
            );
            mesh.set_multiply_color(multiply);
            mesh
        };
        let mut meshes = vec![
            make_mesh(-0.7, 10.0, [1.0, 0.6, 0.3]),
            make_mesh(-0.7, 20.0, [0.3, 0.6, 1.0]),
            make_mesh(-0.5, 30.0, [0.4, 1.0, 0.6]),
        ];
        let static_buffers = WgpuMeshBuffers::from_static_drawables(&device, &meshes).unwrap();
        let mut packed = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
        let mut reference = unpacked(&device, &meshes);
        assert_eq!(packed.indirect.is_some(), indirect_supported);
        assert_eq!(
            static_buffers.drawables()[0].draw_index_range(),
            static_buffers.drawables()[1].draw_index_range()
        );
        let original = pixels(&device, &queue, &renderer, &reference, &textures);
        assert_eq!(
            pixels(&device, &queue, &renderer, &static_buffers, &textures),
            original
        );
        assert_eq!(
            pixels(&device, &queue, &renderer, &packed, &textures),
            original
        );

        let vertex = meshes[1].vertices()[0];
        meshes[1].vertices_mut()[0].set_position([0.5, 0.25]);
        packed.update_drawables(&queue, &meshes).unwrap();
        reference.update_drawables(&queue, &meshes).unwrap();
        let deformed = pixels(&device, &queue, &renderer, &reference, &textures);
        assert_ne!(deformed, original);
        assert_eq!(
            pixels(&device, &queue, &renderer, &packed, &textures),
            deformed
        );
        meshes[1].vertices_mut()[0] = vertex;
        packed.update_drawables(&queue, &meshes).unwrap();
        assert_eq!(
            pixels(&device, &queue, &renderer, &packed, &textures),
            original
        );

        // Reordering the overlapping meshes changes the pixels and requires
        // rebuilding the indirect arguments with the new drawable order.
        meshes[2].set_draw_order(5.0);
        packed.update_drawables(&queue, &meshes).unwrap();
        reference.update_drawables(&queue, &meshes).unwrap();
        let reordered = pixels(&device, &queue, &renderer, &reference, &textures);
        assert_ne!(reordered, original);
        assert_eq!(
            pixels(&device, &queue, &renderer, &packed, &textures),
            reordered
        );
    }
}
