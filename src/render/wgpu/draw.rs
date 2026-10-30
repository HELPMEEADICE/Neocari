use super::{
    WgpuLive2dRenderer,
    buffers::WgpuMeshBuffers,
    clipping::{WgpuClippingResources, WgpuMaskRenderTarget},
    texture::{WgpuTexture, WgpuTransform},
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WgpuRenderError {
    #[error("invalid texture index {texture_index}")]
    InvalidTextureIndex { texture_index: i32 },
    #[error("missing texture bind group {texture_index}")]
    MissingTexture { texture_index: i32 },
    #[error("missing drawable {drawable_index}")]
    MissingDrawable { drawable_index: usize },
    #[error("drawable {drawable_index} has clipping masks but no prepared clipping context")]
    MissingClippingContext { drawable_index: usize },
    #[error(
        "drawable {drawable_index} uses {mask_count} clipping masks, but clipping is not implemented"
    )]
    UnsupportedClippingMasks {
        drawable_index: usize,
        mask_count: usize,
    },
}

impl WgpuLive2dRenderer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_model_drawable(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        buffers: &WgpuMeshBuffers,
        index: usize,
        textures: &[WgpuTexture],
        clipping: &WgpuClippingResources,
        mask: &WgpuMaskRenderTarget,
        transform: &WgpuTransform,
        blend: crate::moc3::Moc3DrawableBlendMode,
    ) -> Result<bool, WgpuRenderError> {
        let drawable = buffers
            .drawables()
            .get(index)
            .ok_or(WgpuRenderError::MissingDrawable {
                drawable_index: index,
            })?;
        if !drawable.is_visible() {
            return Ok(false);
        }
        let texture = texture_bind_group_from_textures(textures, drawable.texture_index())?;
        pass.set_bind_group(0, texture, &[]);
        pass.set_bind_group(1, transform.bind_group(), &[]);
        if drawable.masks().is_empty() {
            pass.set_pipeline(self.pipeline_for_blend_mode(blend));
        } else {
            let context = clipping.context_for_drawable(index).ok_or(
                WgpuRenderError::MissingClippingContext {
                    drawable_index: index,
                },
            )?;
            pass.set_pipeline(self.masked_pipeline_for_blend_mode(blend));
            pass.set_bind_group(2, mask.bind_group(), &[]);
            pass.set_bind_group(3, context.clip_params().bind_group(), &[]);
        }
        if let Some((vertices, indices)) = buffers.packed_buffers() {
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
        } else {
            pass.set_vertex_buffer(0, drawable.vertex_buffer_slice());
            pass.set_index_buffer(drawable.index_buffer_slice(), wgpu::IndexFormat::Uint16);
        }
        pass.draw_indexed(drawable.draw_index_range(), drawable.base_vertex(), 0..1);
        Ok(true)
    }
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        texture_bind_groups: &[wgpu::BindGroup],
    ) -> Result<u32, WgpuRenderError> {
        self.draw_with_bind_groups_and_transform(
            render_pass,
            mesh_buffers,
            texture_bind_groups,
            &self.identity_transform,
        )
    }

    pub fn draw_with_bind_groups_and_transform(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        texture_bind_groups: &[wgpu::BindGroup],
        transform: &WgpuTransform,
    ) -> Result<u32, WgpuRenderError> {
        self.draw_with_bind_group_provider(render_pass, mesh_buffers, transform, |texture_index| {
            texture_bind_group_at(texture_bind_groups, texture_index)
        })
    }

    pub fn draw_with_textures(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        textures: &[WgpuTexture],
    ) -> Result<u32, WgpuRenderError> {
        self.draw_with_textures_and_transform(
            render_pass,
            mesh_buffers,
            textures,
            &self.identity_transform,
        )
    }

    pub fn draw_with_textures_and_transform(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        textures: &[WgpuTexture],
        transform: &WgpuTransform,
    ) -> Result<u32, WgpuRenderError> {
        self.draw_with_bind_group_provider(render_pass, mesh_buffers, transform, |texture_index| {
            texture_bind_group_from_textures(textures, texture_index)
        })
    }

    pub fn draw_with_textures_and_clipping(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        textures: &[WgpuTexture],
        clipping_resources: &WgpuClippingResources,
        mask_target: &WgpuMaskRenderTarget,
    ) -> Result<u32, WgpuRenderError> {
        self.draw_with_textures_clipping_and_transform(
            render_pass,
            mesh_buffers,
            textures,
            clipping_resources,
            mask_target,
            &self.identity_transform,
        )
    }

    pub fn draw_with_textures_clipping_and_transform(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        textures: &[WgpuTexture],
        clipping_resources: &WgpuClippingResources,
        mask_target: &WgpuMaskRenderTarget,
        transform: &WgpuTransform,
    ) -> Result<u32, WgpuRenderError> {
        self.draw_with_clipping_bind_group_provider(
            render_pass,
            mesh_buffers,
            clipping_resources,
            mask_target.bind_group(),
            transform,
            |texture_index| texture_bind_group_from_textures(textures, texture_index),
        )
    }

    pub fn draw_masks(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        clipping_resources: &WgpuClippingResources,
        texture_bind_groups: &[wgpu::BindGroup],
    ) -> Result<u32, WgpuRenderError> {
        self.draw_masks_with_bind_group_provider(
            render_pass,
            mesh_buffers,
            clipping_resources,
            |texture_index| texture_bind_group_at(texture_bind_groups, texture_index),
        )
    }

    pub fn draw_masks_with_textures(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        clipping_resources: &WgpuClippingResources,
        textures: &[WgpuTexture],
    ) -> Result<u32, WgpuRenderError> {
        self.draw_masks_with_bind_group_provider(
            render_pass,
            mesh_buffers,
            clipping_resources,
            |texture_index| texture_bind_group_from_textures(textures, texture_index),
        )
    }

    fn draw_with_bind_group_provider<'a>(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        transform: &WgpuTransform,
        mut bind_group_for_texture: impl FnMut(i32) -> Result<&'a wgpu::BindGroup, WgpuRenderError>,
    ) -> Result<u32, WgpuRenderError> {
        let mut drawn = 0;
        let packed_buffers = mesh_buffers.packed_buffers();
        if let Some((vertex_buffer, index_buffer)) = packed_buffers {
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        }
        for &drawable_index in mesh_buffers.draw_order_indices() {
            let drawable = mesh_buffers
                .drawables()
                .get(drawable_index)
                .ok_or(WgpuRenderError::MissingDrawable { drawable_index })?;
            if !drawable.is_visible() {
                continue;
            }
            if !drawable.masks().is_empty() {
                return Err(WgpuRenderError::UnsupportedClippingMasks {
                    drawable_index,
                    mask_count: drawable.masks().len(),
                });
            }
            let texture_bind_group = bind_group_for_texture(drawable.texture_index())?;

            render_pass.set_pipeline(self.pipeline_for_blend_mode(drawable.blend_mode()));
            render_pass.set_bind_group(0, texture_bind_group, &[]);
            render_pass.set_bind_group(1, transform.bind_group(), &[]);
            if packed_buffers.is_none() {
                render_pass.set_vertex_buffer(0, drawable.vertex_buffer_slice());
                render_pass
                    .set_index_buffer(drawable.index_buffer_slice(), wgpu::IndexFormat::Uint16);
            }
            render_pass.draw_indexed(drawable.draw_index_range(), drawable.base_vertex(), 0..1);
            drawn += 1;
        }

        Ok(drawn)
    }

    fn draw_masks_with_bind_group_provider<'a>(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        clipping_resources: &WgpuClippingResources,
        mut bind_group_for_texture: impl FnMut(i32) -> Result<&'a wgpu::BindGroup, WgpuRenderError>,
    ) -> Result<u32, WgpuRenderError> {
        let mut drawn = 0;
        let packed_buffers = mesh_buffers.packed_buffers();
        if let Some((vertex_buffer, index_buffer)) = packed_buffers {
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        }
        for context in clipping_resources.contexts() {
            for &drawable_index in context.mask_drawable_indices() {
                let drawable = mesh_buffers
                    .drawables()
                    .get(drawable_index)
                    .ok_or(WgpuRenderError::MissingDrawable { drawable_index })?;
                if drawable.is_empty() || !drawable.info().is_enabled() {
                    continue;
                }
                let texture_bind_group = bind_group_for_texture(drawable.texture_index())?;

                render_pass.set_pipeline(&self.mask_pipeline);
                render_pass.set_bind_group(0, texture_bind_group, &[]);
                render_pass.set_bind_group(1, context.mask_transform().bind_group(), &[]);
                render_pass.set_bind_group(2, context.mask_params().bind_group(), &[]);
                if packed_buffers.is_none() {
                    render_pass.set_vertex_buffer(0, drawable.vertex_buffer_slice());
                    render_pass
                        .set_index_buffer(drawable.index_buffer_slice(), wgpu::IndexFormat::Uint16);
                }
                render_pass.draw_indexed(drawable.draw_index_range(), drawable.base_vertex(), 0..1);
                drawn += 1;
            }
        }

        Ok(drawn)
    }

    fn draw_with_clipping_bind_group_provider<'a>(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        mesh_buffers: &WgpuMeshBuffers,
        clipping_resources: &WgpuClippingResources,
        mask_bind_group: &wgpu::BindGroup,
        transform: &WgpuTransform,
        mut bind_group_for_texture: impl FnMut(i32) -> Result<&'a wgpu::BindGroup, WgpuRenderError>,
    ) -> Result<u32, WgpuRenderError> {
        let mut drawn = 0;
        let packed_buffers = mesh_buffers.packed_buffers();
        if let Some((vertex_buffer, index_buffer)) = packed_buffers {
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        }
        for &drawable_index in mesh_buffers.draw_order_indices() {
            let drawable = mesh_buffers
                .drawables()
                .get(drawable_index)
                .ok_or(WgpuRenderError::MissingDrawable { drawable_index })?;
            if !drawable.is_visible() {
                continue;
            }
            let texture_bind_group = bind_group_for_texture(drawable.texture_index())?;

            if drawable.masks().is_empty() {
                render_pass.set_pipeline(self.pipeline_for_blend_mode(drawable.blend_mode()));
                render_pass.set_bind_group(0, texture_bind_group, &[]);
                render_pass.set_bind_group(1, transform.bind_group(), &[]);
            } else {
                let context = clipping_resources
                    .context_for_drawable(drawable_index)
                    .ok_or(WgpuRenderError::MissingClippingContext { drawable_index })?;
                render_pass
                    .set_pipeline(self.masked_pipeline_for_blend_mode(drawable.blend_mode()));
                render_pass.set_bind_group(0, texture_bind_group, &[]);
                render_pass.set_bind_group(1, transform.bind_group(), &[]);
                render_pass.set_bind_group(2, mask_bind_group, &[]);
                render_pass.set_bind_group(3, context.clip_params().bind_group(), &[]);
            }

            if packed_buffers.is_none() {
                render_pass.set_vertex_buffer(0, drawable.vertex_buffer_slice());
                render_pass
                    .set_index_buffer(drawable.index_buffer_slice(), wgpu::IndexFormat::Uint16);
            }
            render_pass.draw_indexed(drawable.draw_index_range(), drawable.base_vertex(), 0..1);
            drawn += 1;
        }

        Ok(drawn)
    }
}

fn texture_bind_group_at(
    texture_bind_groups: &[wgpu::BindGroup],
    texture_index: i32,
) -> Result<&wgpu::BindGroup, WgpuRenderError> {
    let texture_index_usize = usize::try_from(texture_index)
        .map_err(|_| WgpuRenderError::InvalidTextureIndex { texture_index })?;
    texture_bind_groups
        .get(texture_index_usize)
        .ok_or(WgpuRenderError::MissingTexture { texture_index })
}

fn texture_bind_group_from_textures(
    textures: &[WgpuTexture],
    texture_index: i32,
) -> Result<&wgpu::BindGroup, WgpuRenderError> {
    let texture_index_usize = usize::try_from(texture_index)
        .map_err(|_| WgpuRenderError::InvalidTextureIndex { texture_index })?;
    textures
        .get(texture_index_usize)
        .map(WgpuTexture::bind_group)
        .ok_or(WgpuRenderError::MissingTexture { texture_index })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        moc3::{Moc3DrawableMesh, Moc3DrawableVertex},
        render::wgpu::WgpuClippingPlan,
    };

    #[test]
    fn disabled_masks_are_skipped_while_transparent_enabled_masks_are_drawn() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let renderer = WgpuLive2dRenderer::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let textures = vec![
            renderer
                .create_rgba8_texture(&device, &queue, 1, 1, &[255; 4])
                .unwrap(),
        ];
        let mesh = |opacity, masks| {
            Moc3DrawableMesh::from_parts(
                0,
                0,
                opacity,
                0.0,
                vec![
                    Moc3DrawableVertex::new([-0.5, -0.5], [0.0, 0.0]),
                    Moc3DrawableVertex::new([0.5, -0.5], [1.0, 0.0]),
                    Moc3DrawableVertex::new([0.0, 0.5], [0.5, 1.0]),
                ],
                vec![0, 1, 2],
                masks,
            )
        };
        let mut meshes = vec![mesh(0.0, vec![]), mesh(1.0, vec![0])];
        let mut buffers = WgpuMeshBuffers::from_drawables(&device, &meshes).unwrap();
        let mut plan = WgpuClippingPlan::from_mesh_buffers(&buffers);
        plan.prepare_single_texture_masks(&buffers).unwrap();
        let resources = renderer.create_clipping_resources(&device, &plan).unwrap();
        let target = renderer.create_mask_render_target(&device, 16).unwrap();
        for (enabled, expected) in [(true, 1), (false, 0), (true, 1)] {
            meshes[0].set_enabled(enabled);
            buffers.update_drawables(&queue, &meshes).unwrap();
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view(),
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
                assert_eq!(
                    renderer
                        .draw_masks_with_textures(&mut pass, &buffers, &resources, &textures)
                        .unwrap(),
                    expected
                );
            }
            let _ = encoder.finish();
        }
    }
}
