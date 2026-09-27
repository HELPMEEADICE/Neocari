use std::{error::Error, path::PathBuf, sync::mpsc, time::Duration};

use neocari::{
    assets::load_moc2_model,
    core::Matrix44,
    render::wgpu::{WgpuClippingPlan, WgpuLive2dRenderer, WgpuMaskRenderTarget, WgpuMeshBuffers},
};

const WIDTH: u32 = 400;
const HEIGHT: u32 = 650;
const MASK_SIZE: u32 = 2048;

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(render())
}

async fn render() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_path = args.next().map(PathBuf::from).ok_or(
        "usage: cargo run --features wgpu --example render_moc2 -- <model2.json> [output.png]",
    )?;
    let output_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("moc2-render.png"));

    eprintln!("loading V2 model");
    let model = load_moc2_model(&model_path)?;
    eprintln!(
        "loaded {} drawables, {} textures",
        model.drawables().len(),
        model.textures().len()
    );
    let drawables = model.drawables();
    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for vertex in drawables.iter().flat_map(|drawable| drawable.vertices()) {
        let [x, y] = vertex.position();
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].min(y);
        bounds[2] = bounds[2].max(x);
        bounds[3] = bounds[3].max(y);
    }
    let bounds_width = bounds[2] - bounds[0];
    let bounds_height = bounds[3] - bounds[1];
    if !bounds_width.is_finite()
        || !bounds_height.is_finite()
        || bounds_width <= 0.0
        || bounds_height <= 0.0
    {
        return Err("model has no drawable bounds".into());
    }

    eprintln!("requesting GPU adapter");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await?;
    eprintln!("adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("render_moc2.device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await?;

    eprintln!("device ready");
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = WgpuLive2dRenderer::new(&device, format);
    let textures = model
        .textures()
        .iter()
        .map(|texture| {
            renderer.create_rgba8_texture(
                &device,
                &queue,
                texture.width(),
                texture.height(),
                texture.rgba(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!("textures uploaded");
    let mesh_buffers = WgpuMeshBuffers::from_static_drawables(&device, drawables)
        .ok_or("failed to create GPU mesh buffers")?;
    eprintln!("mesh buffers ready");
    let mut clipping_plan = WgpuClippingPlan::from_mesh_buffers(&mesh_buffers);
    clipping_plan.prepare_single_texture_masks(&mesh_buffers)?;
    let clipping_resources = renderer.create_clipping_resources(&device, &clipping_plan)?;
    let mask_target: WgpuMaskRenderTarget =
        renderer.create_mask_render_target(&device, MASK_SIZE)?;

    let scale_y = 1.5 / bounds_height;
    let scale_x = scale_y * HEIGHT as f32 / WIDTH as f32;
    let mut matrix = Matrix44::identity();
    matrix.scale(scale_x, -scale_y);
    matrix.translate(
        -(bounds[0] + bounds[2]) * 0.5 * scale_x,
        (bounds[1] + bounds[3]) * 0.5 * scale_y,
    );
    let transform = renderer.create_transform(&device, &matrix);

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_moc2.output"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let padded_bytes_per_row = (WIDTH * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_moc2.readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render_moc2.encoder"),
    });
    if !clipping_resources.contexts().is_empty() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_moc2.mask_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: mask_target.view(),
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
        renderer.draw_masks_with_textures(
            &mut pass,
            &mesh_buffers,
            &clipping_resources,
            &textures,
        )?;
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_moc2.model_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.92,
                        g: 0.94,
                        b: 0.98,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.draw_with_textures_clipping_and_transform(
            &mut pass,
            &mesh_buffers,
            &textures,
            &clipping_resources,
            &mask_target,
            &transform,
        )?;
    }
    encoder.copy_texture_to_buffer(
        output_texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    eprintln!("submitting render commands");
    queue.submit([encoder.finish()]);

    eprintln!("submitted, mapping readback");
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })?;
    receiver.recv()??;
    eprintln!("readback complete");
    let mapped = slice.get_mapped_range()?;
    let mut pixels = vec![0; (WIDTH * HEIGHT * 4) as usize];
    let mut background = [0; 3];
    let mut non_background_pixels = 0usize;
    let mut min_x = WIDTH;
    let mut min_y = HEIGHT;
    let mut max_x = 0;
    let mut max_y = 0;
    for y in 0..HEIGHT {
        let source_row = (y * padded_bytes_per_row) as usize;
        let target_row = (y * WIDTH * 4) as usize;
        let row_len = (WIDTH * 4) as usize;
        pixels[target_row..target_row + row_len]
            .copy_from_slice(&mapped[source_row..source_row + row_len]);
        if y == 0 {
            background.copy_from_slice(&pixels[0..3]);
        }
        for x in 0..WIDTH {
            let offset = target_row + (x * 4) as usize;
            let [r, g, b] = [pixels[offset], pixels[offset + 1], pixels[offset + 2]];
            if [r, g, b] != background {
                non_background_pixels += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    drop(mapped);
    readback.unmap();
    image::save_buffer(
        &output_path,
        &pixels,
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
    )?;
    let adapter_info = adapter.get_info();
    println!(
        "Rendered {} drawables ({} visible), {} clipping contexts on {}.",
        drawables.len(),
        mesh_buffers
            .drawables()
            .iter()
            .filter(|item| item.is_visible())
            .count(),
        clipping_resources.contexts().len(),
        adapter_info.name
    );
    println!(
        "{} non-background pixels; model bounds in image: ({min_x},{min_y})-({max_x},{max_y}).",
        non_background_pixels
    );
    println!("Saved {}", output_path.display());
    if non_background_pixels < 100 {
        return Err("render output contains too few model pixels".into());
    }
    Ok(())
}
