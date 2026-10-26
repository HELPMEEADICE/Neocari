//! Renders Cubism V2 animation frames with the built-in WGPU renderer.
//!
//! ```text
//! cargo run --features wgpu --example render_moc2_animation -- \
//!     path/to/model.model.json out.png --motion idle01 --expression smile01 --frames 30
//! ```

use std::{error::Error, path::PathBuf, sync::mpsc, time::Duration};

use neocari::{
    assets::load_moc2_runtime,
    core::Matrix44,
    render::wgpu::{WgpuClippingPlan, WgpuLive2dRenderer, WgpuMaskRenderTarget, WgpuMeshBuffers},
};

const DEFAULT_WIDTH: u32 = 400;
const DEFAULT_HEIGHT: u32 = 650;
const MASK_SIZE: u32 = 2048;

struct Options {
    model: PathBuf,
    output: PathBuf,
    motion: Option<(String, usize)>,
    expression: Option<String>,
    frames: u32,
    delta_seconds: f32,
    drag: Option<(f32, f32)>,
    auto_behaviours: bool,
    transparent: bool,
    width: u32,
    height: u32,
}

fn parse_args() -> Result<Options, Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let model = args
        .next()
        .ok_or("usage: render_moc2_animation <model2.json> [output.png] [options]")?;
    let mut output = None;
    let mut motion = None;
    let mut expression = None;
    let mut frames = 1u32;
    let mut delta_seconds = 1.0 / 60.0;
    let mut drag = None;
    let mut auto_behaviours = true;
    let mut transparent = false;
    let mut width = DEFAULT_WIDTH;
    let mut height = DEFAULT_HEIGHT;

    let mut rest = args.peekable();
    while let Some(argument) = rest.next() {
        match argument.to_str().unwrap_or_default() {
            "--motion" => {
                let value = rest.next().ok_or("--motion needs a group name")?;
                let value = value.to_string_lossy().to_string();
                let (group, index) = match value.split_once(':') {
                    Some((group, index)) => (group.to_owned(), index.parse::<usize>()?),
                    None => (value, 0),
                };
                motion = Some((group, index));
            }
            "--expression" => {
                let value = rest.next().ok_or("--expression needs a name")?;
                expression = Some(value.to_string_lossy().to_string());
            }
            "--frames" => {
                let value = rest.next().ok_or("--frames needs a count")?;
                frames = value.to_string_lossy().parse()?;
            }
            "--dt" => {
                let value = rest.next().ok_or("--dt needs a duration")?;
                delta_seconds = value.to_string_lossy().parse()?;
            }
            "--drag" => {
                let x = rest.next().ok_or("--drag needs x and y")?;
                let y = rest.next().ok_or("--drag needs x and y")?;
                drag = Some((x.to_string_lossy().parse()?, y.to_string_lossy().parse()?));
            }
            "--no-auto" => auto_behaviours = false,
            "--transparent" => transparent = true,
            "--width" => {
                let value = rest.next().ok_or("--width needs a pixel count")?;
                width = value.to_string_lossy().parse()?;
            }
            "--height" => {
                let value = rest.next().ok_or("--height needs a pixel count")?;
                height = value.to_string_lossy().parse()?;
            }
            "--help" | "-h" => {
                return Err("usage: render_moc2_animation <model2.json> [output.png] \
                    [--motion group[:index]] [--expression name] [--frames n] [--dt seconds] \
                    [--drag x y] [--no-auto] [--width px] [--height px]"
                    .into());
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}").into());
            }
            _ => output = Some(argument),
        }
    }

    Ok(Options {
        model,
        output: output.unwrap_or_else(|| PathBuf::from("moc2-animation.png")),
        motion,
        expression,
        frames: frames.max(1),
        delta_seconds,
        drag,
        auto_behaviours,
        transparent,
        width,
        height,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(render(parse_args()?))
}

fn fitted_matrix(model: &neocari::assets::Moc2RuntimeModel, width: u32, height: u32) -> Matrix44 {
    // Reproduce the Cubism 2 framework projection: the model matrix maps the
    // canvas to clip space with a width of two units, then the viewport scales
    // the shorter screen axis so the model keeps its aspect ratio.
    let mut projection = Matrix44::identity();
    if height > width {
        projection.scale(1.0, width as f32 / height as f32);
    } else {
        projection.scale(height as f32 / width as f32, 1.0);
    }
    Matrix44::multiply_matrices(&projection, &model.model_matrix())
}

fn output_path(options: &Options, frame: u32) -> PathBuf {
    if options.frames <= 1 {
        return options.output.clone();
    }
    let stem = options
        .output
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "frame".to_owned());
    let parent = options.output.parent().unwrap_or_else(|| ".".as_ref());
    parent.join(format!("{stem}_{frame:03}.png"))
}

async fn render(options: Options) -> Result<(), Box<dyn Error>> {
    eprintln!("loading Cubism V2 model");
    let mut model = load_moc2_runtime(&options.model)?;
    model.runtime_mut().set_auto_blink(options.auto_behaviours);
    model.runtime_mut().set_auto_breath(options.auto_behaviours);
    if let Some((x, y)) = options.drag {
        model.runtime_mut().set_drag(x, y);
    }
    if let Some((group, index)) = &options.motion
        && !model.play_motion(group, *index)?
    {
        eprintln!("motion group {group}[{index}] has no file");
    }
    if let Some(name) = &options.expression
        && !model.play_expression(name)?
    {
        eprintln!("expression {name} is not declared by the model");
    }
    model
        .update(0.0)
        .ok_or("failed to build the initial mesh set")?;
    eprintln!(
        "loaded {} drawables, {} textures, canvas {}x{}",
        model.meshes().len(),
        model.textures().len(),
        model.runtime().canvas().width(),
        model.runtime().canvas().height()
    );

    let width = options.width;
    let height = options.height;
    let format = wgpu::TextureFormat::Rgba8Unorm;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("render_moc2_animation.device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await?;

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
    let mut mesh_buffers =
        WgpuMeshBuffers::from_drawables(&device, model.meshes()).ok_or("mesh buffers")?;

    let matrix = fitted_matrix(&model, width, height);
    let clear_color = if options.transparent {
        wgpu::Color::TRANSPARENT
    } else {
        wgpu::Color {
            r: 0.92,
            g: 0.94,
            b: 0.98,
            a: 1.0,
        }
    };
    let transform = renderer.create_transform(&device, &matrix);
    let mask_target: WgpuMaskRenderTarget =
        renderer.create_mask_render_target(&device, MASK_SIZE)?;

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_moc2_animation.output"),
        size: wgpu::Extent3d {
            width,
            height,
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
    let padded_bytes_per_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_moc2_animation.readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    for frame in 0..options.frames {
        if frame > 0 {
            model
                .update(options.delta_seconds)
                .ok_or("animation update failed")?;
            mesh_buffers
                .update_drawables(&queue, model.meshes())
                .map_err(|error| error.to_string())?;
        }

        let mut clipping_plan = WgpuClippingPlan::from_mesh_buffers(&mesh_buffers);
        clipping_plan.prepare_single_texture_masks(&mesh_buffers)?;
        let clipping_resources = renderer.create_clipping_resources(&device, &clipping_plan)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_moc2_animation.encoder"),
        });
        if !clipping_resources.contexts().is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_moc2_animation.mask_pass"),
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
                label: Some("render_moc2_animation.model_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
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
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

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
        let mapped = slice.get_mapped_range()?;
        let mut pixels = vec![0; (width * height * 4) as usize];
        for y in 0..height {
            let source_row = (y * padded_bytes_per_row) as usize;
            let target_row = (y * width * 4) as usize;
            let row_len = (width * 4) as usize;
            pixels[target_row..target_row + row_len]
                .copy_from_slice(&mapped[source_row..source_row + row_len]);
        }
        drop(mapped);
        readback.unmap();

        let path = output_path(&options, frame);
        image::save_buffer(&path, &pixels, width, height, image::ColorType::Rgba8)?;
        println!("frame {frame} -> {}", path.display());
    }

    println!(
        "Rendered {} frame(s) with {} drawables on {}.",
        options.frames,
        model.meshes().len(),
        adapter.get_info().name
    );
    Ok(())
}
