//! Ordered model composition, including nested Cubism 5.3 offscreen groups.
use super::{
    WgpuClippingLayout, WgpuClippingRect, WgpuClippingResources, WgpuLive2dRenderer,
    WgpuMaskChannel, WgpuMaskParams, WgpuMaskRenderTarget, WgpuMeshBuffers, WgpuRenderError,
    WgpuTexture, WgpuTransform,
};
use crate::moc3::{Moc3BlendConfig, Moc3RenderNode, Moc3RenderPlan};
use wgpu::util::DeviceExt;

#[derive(Debug, thiserror::Error)]
pub enum WgpuModelRenderError {
    #[error(transparent)]
    Draw(#[from] WgpuRenderError),
    #[error("invalid model target size {width}x{height}")]
    InvalidSize { width: u32, height: u32 },
    #[error("model composition resources belong to another output format")]
    FormatMismatch,
    #[error("invalid or cyclic model composition plan")]
    InvalidPlan,
    #[error("model composition uniforms exceed device buffer limits")]
    UniformLimit,
}

/// Prepared mesh, clipping, and material inputs for one model frame.
#[derive(Clone, Copy)]
pub struct WgpuModelFrame<'a> {
    pub plan: &'a Moc3RenderPlan,
    pub meshes: &'a WgpuMeshBuffers,
    pub textures: &'a [WgpuTexture],
    pub clipping: &'a WgpuClippingResources,
    pub mask_target: &'a WgpuMaskRenderTarget,
    pub transform: &'a WgpuTransform,
}

pub struct WgpuModelTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub load: wgpu::LoadOp<wgpu::Color>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WgpuModelRenderStats {
    pub drawable_draws: u32,
    pub composite_draws: u32,
    pub allocated_layers: usize,
}

#[derive(Debug)]
struct Surface {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}
#[derive(Debug)]
struct Layer {
    color: Surface,
    backdrop: Surface,
    mask: Surface,
}
#[derive(Debug)]
struct Uniforms {
    bind_group: wgpu::BindGroup,
    stride: u32,
    identity: u32,
}

/// Reusable layers are allocated by nesting depth, not by group count.
/// Resizing drops the old pool. A frame uploads one aligned uniform batch.
#[derive(Debug)]
pub struct WgpuModelResources {
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    layers: Vec<Layer>,
    uniform_layout: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    mask_params: WgpuMaskParams,
}

impl WgpuLive2dRenderer {
    pub fn create_model_resources(&self, device: &wgpu::Device) -> WgpuModelResources {
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("live2d.model.uniforms.layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("live2d.model.composite.layout"),
            bind_group_layouts: &[
                Some(&self.texture_bind_group_layout),
                Some(&self.texture_bind_group_layout),
                Some(&self.texture_bind_group_layout),
                Some(&uniform_layout),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("live2d.model.composite.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });
        let pipeline = |entry, blend| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("live2d.model.composite.pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        WgpuModelResources {
            format: self.color_format,
            width: 0,
            height: 0,
            layers: Vec::new(),
            uniform_layout,
            composite_pipeline: pipeline("fs_composite", None),
            present_pipeline: pipeline(
                "fs_present",
                Some(super::live2d_blend_state(
                    crate::moc3::Moc3DrawableBlendMode::Normal,
                )),
            ),
            mask_params: self.create_mask_params(
                device,
                WgpuClippingLayout::new(
                    WgpuMaskChannel::Red,
                    WgpuClippingRect::new(0.0, 0.0, 1.0, 1.0),
                ),
            ),
        }
    }

    /// Encodes masks, nested groups, advanced blending, and final presentation.
    /// Call after updating the runtime, mesh buffers, and clipping resources.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_model(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut WgpuModelResources,
        frame: WgpuModelFrame<'_>,
        target: WgpuModelTarget<'_>,
    ) -> Result<WgpuModelRenderStats, WgpuModelRenderError> {
        if self.color_format != resources.format {
            return Err(WgpuModelRenderError::FormatMismatch);
        }
        let limit = device.limits().max_texture_dimension_2d;
        if target.width == 0 || target.height == 0 || target.width > limit || target.height > limit
        {
            return Err(WgpuModelRenderError::InvalidSize {
                width: target.width,
                height: target.height,
            });
        }
        if resources.width != target.width || resources.height != target.height {
            resources.layers.clear();
            resources.width = target.width;
            resources.height = target.height;
        }
        if !frame.clipping.contexts().is_empty() {
            let mut pass = begin_pass(
                encoder,
                frame.mask_target.view(),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            self.draw_masks_with_textures(&mut pass, frame.meshes, frame.clipping, frame.textures)?;
        }
        if frame.plan.offscreens.is_empty()
            && frame
                .plan
                .drawable_blends
                .iter()
                .all(|blend| !blend.is_advanced())
        {
            let mut pass = begin_pass(encoder, target.view, target.load);
            let mut stats = WgpuModelRenderStats::default();
            for node in &frame.plan.nodes {
                let Moc3RenderNode::Drawable(index) = *node else {
                    return Err(WgpuModelRenderError::InvalidPlan);
                };
                let blend = *frame
                    .plan
                    .drawable_blends
                    .get(index)
                    .ok_or(WgpuModelRenderError::InvalidPlan)?;
                stats.drawable_draws += u32::from(self.draw_model_drawable(
                    &mut pass,
                    frame.meshes,
                    index,
                    frame.textures,
                    frame.clipping,
                    frame.mask_target,
                    frame.transform,
                    simple_blend_mode(blend),
                )?);
            }
            stats.allocated_layers = resources.layers.len();
            return Ok(stats);
        }
        resources.ensure_layer(device, self, 0);
        let uniforms = resources.uniforms(device, frame.plan)?;
        clear(encoder, &resources.layers[0].color.view);
        let mut stats = WgpuModelRenderStats::default();
        resources.render_nodes(
            device,
            self,
            encoder,
            frame,
            &uniforms,
            &frame.plan.nodes,
            0,
            &mut stats,
        )?;
        {
            let mut pass = begin_pass(encoder, target.view, target.load);
            pass.set_pipeline(&resources.present_pipeline);
            let layer = &resources.layers[0];
            pass.set_bind_group(0, &layer.color.bind_group, &[]);
            pass.set_bind_group(1, &layer.backdrop.bind_group, &[]);
            pass.set_bind_group(2, &layer.mask.bind_group, &[]);
            pass.set_bind_group(3, &uniforms.bind_group, &[uniforms.identity]);
            pass.draw(0..3, 0..1);
        }
        stats.allocated_layers = resources.layers.len();
        Ok(stats)
    }
}

impl WgpuModelResources {
    fn ensure_layer(&mut self, device: &wgpu::Device, renderer: &WgpuLive2dRenderer, depth: usize) {
        while self.layers.len() <= depth {
            let surface = |format| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("live2d.model.layer"),
                    size: wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                let bind_group = renderer.create_texture_bind_group(device, &view);
                Surface {
                    texture,
                    view,
                    bind_group,
                }
            };
            self.layers.push(Layer {
                color: surface(self.format),
                backdrop: surface(self.format),
                mask: surface(wgpu::TextureFormat::Rgba8Unorm),
            });
        }
    }

    fn uniforms(
        &self,
        device: &wgpu::Device,
        plan: &Moc3RenderPlan,
    ) -> Result<Uniforms, WgpuModelRenderError> {
        let stride = 64u32.div_ceil(device.limits().min_uniform_buffer_offset_alignment)
            * device.limits().min_uniform_buffer_offset_alignment;
        let count = plan
            .offscreens
            .len()
            .checked_add(plan.drawable_blends.len())
            .and_then(|n| n.checked_add(1))
            .ok_or(WgpuModelRenderError::UniformLimit)?;
        let len = (stride as usize)
            .checked_mul(count)
            .ok_or(WgpuModelRenderError::UniformLimit)?;
        if len as u64 > device.limits().max_buffer_size || len > u32::MAX as usize {
            return Err(WgpuModelRenderError::UniformLimit);
        }
        let mut bytes = vec![0; len];
        let mut encode = |index: usize,
                          blend: Moc3BlendConfig,
                          opacity: f32,
                          multiply: [f32; 3],
                          screen: [f32; 3],
                          masked: bool,
                          inverted: bool| {
            let row = &mut bytes[index * stride as usize..index * stride as usize + 64];
            for (i, v) in multiply
                .into_iter()
                .chain([0.0])
                .chain(screen)
                .chain([0.0])
                .enumerate()
            {
                row[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
            }
            for (i, v) in [
                blend.color as u32,
                blend.alpha as u32,
                u32::from(masked),
                u32::from(inverted),
            ]
            .into_iter()
            .enumerate()
            {
                row[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_ne_bytes());
            }
            row[48..52].copy_from_slice(&opacity.to_ne_bytes());
        };
        for (i, s) in plan.offscreens.iter().enumerate() {
            encode(
                i,
                s.blend,
                s.opacity,
                s.multiply_color,
                s.screen_color,
                !s.masks.is_empty(),
                s.inverted_mask,
            );
        }
        for (i, &blend) in plan.drawable_blends.iter().enumerate() {
            encode(
                plan.offscreens.len() + i,
                blend,
                1.0,
                [1.0; 3],
                [0.0; 3],
                false,
                false,
            );
        }
        encode(
            count - 1,
            Default::default(),
            1.0,
            [1.0; 3],
            [0.0; 3],
            false,
            false,
        );
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("live2d.model.uniforms"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("live2d.model.uniforms.bind_group"),
            layout: &self.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });
        Ok(Uniforms {
            bind_group,
            stride,
            identity: (count as u32 - 1) * stride,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn render_nodes(
        &mut self,
        device: &wgpu::Device,
        renderer: &WgpuLive2dRenderer,
        encoder: &mut wgpu::CommandEncoder,
        frame: WgpuModelFrame<'_>,
        uniforms: &Uniforms,
        nodes: &[Moc3RenderNode],
        depth: usize,
        stats: &mut WgpuModelRenderStats,
    ) -> Result<(), WgpuModelRenderError> {
        if depth > frame.plan.offscreens.len() + 1 {
            return Err(WgpuModelRenderError::InvalidPlan);
        }
        let mut index = 0;
        while index < nodes.len() {
            if let Moc3RenderNode::Drawable(i) = nodes[index] {
                let blend = *frame
                    .plan
                    .drawable_blends
                    .get(i)
                    .ok_or(WgpuModelRenderError::InvalidPlan)?;
                if !blend.is_advanced() {
                    let mut pass =
                        begin_pass(encoder, &self.layers[depth].color.view, wgpu::LoadOp::Load);
                    while let Some(Moc3RenderNode::Drawable(i)) = nodes.get(index) {
                        if frame
                            .plan
                            .drawable_blends
                            .get(*i)
                            .ok_or(WgpuModelRenderError::InvalidPlan)?
                            .is_advanced()
                        {
                            break;
                        }
                        let blend = simple_blend_mode(frame.plan.drawable_blends[*i]);
                        stats.drawable_draws += u32::from(renderer.draw_model_drawable(
                            &mut pass,
                            frame.meshes,
                            *i,
                            frame.textures,
                            frame.clipping,
                            frame.mask_target,
                            frame.transform,
                            blend,
                        )?);
                        index += 1;
                    }
                    continue;
                }
                self.ensure_layer(device, renderer, depth + 1);
                clear(encoder, &self.layers[depth + 1].color.view);
                {
                    let mut pass = begin_pass(
                        encoder,
                        &self.layers[depth + 1].color.view,
                        wgpu::LoadOp::Load,
                    );
                    stats.drawable_draws += u32::from(renderer.draw_model_drawable(
                        &mut pass,
                        frame.meshes,
                        i,
                        frame.textures,
                        frame.clipping,
                        frame.mask_target,
                        frame.transform,
                        crate::moc3::Moc3DrawableBlendMode::Normal,
                    )?);
                }
                self.composite(
                    encoder,
                    uniforms,
                    depth,
                    (frame.plan.offscreens.len() + i) as u32 * uniforms.stride,
                );
                stats.composite_draws += 1;
            } else if let Moc3RenderNode::Offscreen(i) = nodes[index] {
                let group = frame
                    .plan
                    .offscreens
                    .get(i)
                    .ok_or(WgpuModelRenderError::InvalidPlan)?;
                if group.enabled && group.opacity > 0.0 {
                    self.ensure_layer(device, renderer, depth + 1);
                    clear(encoder, &self.layers[depth + 1].color.view);
                    self.render_nodes(
                        device,
                        renderer,
                        encoder,
                        frame,
                        uniforms,
                        &group.nodes,
                        depth + 1,
                        stats,
                    )?;
                    if !group.masks.is_empty() {
                        self.group_mask(renderer, encoder, frame, &group.masks, depth + 1)?;
                    }
                    self.composite(encoder, uniforms, depth, i as u32 * uniforms.stride);
                    stats.composite_draws += 1;
                }
            }
            index += 1;
        }
        Ok(())
    }

    fn group_mask(
        &self,
        renderer: &WgpuLive2dRenderer,
        encoder: &mut wgpu::CommandEncoder,
        frame: WgpuModelFrame<'_>,
        masks: &[i32],
        depth: usize,
    ) -> Result<(), WgpuModelRenderError> {
        let mut pass = begin_pass(
            encoder,
            &self.layers[depth].mask.view,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        pass.set_pipeline(&renderer.mask_pipeline);
        pass.set_bind_group(1, frame.transform.bind_group(), &[]);
        pass.set_bind_group(2, self.mask_params.bind_group(), &[]);
        for &raw in masks {
            let index = usize::try_from(raw).map_err(|_| WgpuModelRenderError::InvalidPlan)?;
            let d = frame
                .meshes
                .drawables()
                .get(index)
                .ok_or(WgpuModelRenderError::InvalidPlan)?;
            if !d.info().is_enabled() || d.is_empty() {
                continue;
            }
            let t = usize::try_from(d.texture_index())
                .ok()
                .and_then(|i| frame.textures.get(i))
                .ok_or(WgpuRenderError::MissingTexture {
                    texture_index: d.texture_index(),
                })?;
            pass.set_bind_group(0, t.bind_group(), &[]);
            if let Some((v, i)) = frame.meshes.packed_buffers() {
                pass.set_vertex_buffer(0, v.slice(..));
                pass.set_index_buffer(i.slice(..), wgpu::IndexFormat::Uint16);
            } else {
                pass.set_vertex_buffer(0, d.vertex_buffer_slice());
                pass.set_index_buffer(d.index_buffer_slice(), wgpu::IndexFormat::Uint16);
            }
            pass.draw_indexed(d.draw_index_range(), d.base_vertex(), 0..1);
        }
        Ok(())
    }

    fn composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        uniforms: &Uniforms,
        depth: usize,
        offset: u32,
    ) {
        let parent = &self.layers[depth];
        let child = &self.layers[depth + 1];
        encoder.copy_texture_to_texture(
            parent.color.texture.as_image_copy(),
            parent.backdrop.texture.as_image_copy(),
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let mut pass = begin_pass(encoder, &parent.color.view, wgpu::LoadOp::Load);
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &child.color.bind_group, &[]);
        pass.set_bind_group(1, &parent.backdrop.bind_group, &[]);
        pass.set_bind_group(2, &child.mask.bind_group, &[]);
        pass.set_bind_group(3, &uniforms.bind_group, &[offset]);
        pass.draw(0..3, 0..1);
    }
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("live2d.model.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
fn clear(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
    let _ = begin_pass(encoder, view, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT));
}

fn simple_blend_mode(blend: Moc3BlendConfig) -> crate::moc3::Moc3DrawableBlendMode {
    use crate::moc3::{Moc3ColorBlendMode as Color, Moc3DrawableBlendMode as Mode};
    match blend.color {
        Color::PremultipliedAdd => Mode::Additive,
        Color::PremultipliedMultiply => Mode::Multiplicative,
        _ => Mode::Normal,
    }
}
