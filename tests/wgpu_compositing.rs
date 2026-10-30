#![cfg(feature = "wgpu")]

use neocari::{
    core::Matrix44,
    moc3::{
        Moc3AlphaBlendMode as Alpha, Moc3BlendConfig, Moc3ColorBlendMode as Color,
        Moc3DrawableMesh, Moc3DrawableVertex, Moc3OffscreenState, Moc3RenderNode as Node,
        Moc3RenderPlan,
    },
    render::wgpu::{
        WgpuClippingPlan, WgpuLive2dRenderer, WgpuMeshBuffers, WgpuModelFrame,
        WgpuModelRenderStats, WgpuModelResources, WgpuModelTarget,
    },
};

fn quad(texture: i32, opacity: f32, right: f32) -> Moc3DrawableMesh {
    Moc3DrawableMesh::from_parts(
        texture,
        0,
        opacity,
        0.0,
        vec![
            Moc3DrawableVertex::new([-1.0, -1.0], [0.0, 1.0]),
            Moc3DrawableVertex::new([right, -1.0], [1.0, 1.0]),
            Moc3DrawableVertex::new([right, 1.0], [1.0, 0.0]),
            Moc3DrawableVertex::new([-1.0, 1.0], [0.0, 0.0]),
        ],
        vec![0, 1, 2, 0, 2, 3],
        vec![],
    )
}

fn group(nodes: Vec<Node>, opacity: f32) -> Moc3OffscreenState {
    Moc3OffscreenState {
        part_index: 0,
        enabled: true,
        opacity,
        multiply_color: [1.0; 3],
        screen_color: [0.0; 3],
        blend: Default::default(),
        masks: vec![],
        inverted_mask: false,
        nodes,
    }
}

// Both validation and real GPU tests use packed meshes, including nonzero base vertices.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &WgpuLive2dRenderer,
    resources: &mut WgpuModelResources,
    meshes: &[Moc3DrawableMesh],
    plan: &Moc3RenderPlan,
    read: bool,
) -> (Vec<u8>, WgpuModelRenderStats) {
    let textures = [
        [0, 0, 255, 255],
        [255, 0, 0, 255],
        [255, 255, 255, 255],
        [204, 204, 204, 255],
    ]
    .map(|rgba| {
        renderer
            .create_rgba8_texture(device, queue, 1, 1, &rgba)
            .unwrap()
    });
    let meshes = WgpuMeshBuffers::from_drawables(device, meshes).unwrap();
    let mut clipping = WgpuClippingPlan::from_mesh_buffers(&meshes);
    clipping.prepare_single_texture_masks(&meshes).unwrap();
    let clipping = renderer
        .create_clipping_resources(device, &clipping)
        .unwrap();
    let mask = renderer.create_mask_render_target(device, 16).unwrap();
    let transform = renderer.create_transform(device, &Matrix44::identity());
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
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
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let stats = renderer
        .encode_model(
            device,
            &mut encoder,
            resources,
            WgpuModelFrame {
                plan,
                meshes: &meshes,
                textures: &textures,
                clipping: &clipping,
                mask_target: &mask,
                transform: &transform,
            },
            WgpuModelTarget {
                view: &view,
                width: 8,
                height: 8,
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            },
        )
        .unwrap();
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(8),
            },
        },
        wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    if !read {
        return (vec![], stats);
    }
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let pixels = (0..8)
        .flat_map(|y| mapped[y * 256..y * 256 + 32].iter().copied())
        .collect();
    (pixels, stats)
}

fn cases(device: &wgpu::Device, queue: &wgpu::Queue, read: bool) {
    let renderer = WgpuLive2dRenderer::new(device, wgpu::TextureFormat::Rgba8Unorm);
    let mut resources = renderer.create_model_resources(device);
    let meshes = [
        quad(0, 0.5, 1.0),
        quad(1, 0.5, 1.0),
        quad(2, 1.0, 1.0),
        quad(2, 0.0, 0.0),
        quad(3, 0.5, 1.0),
    ];
    let mut plan = Moc3RenderPlan {
        nodes: vec![Node::Drawable(0), Node::Drawable(1)],
        offscreens: vec![],
        drawable_blends: vec![Default::default(); 5],
    };
    let assert_pixel = |pixels: &[u8], x: usize, expected: [f32; 4]| {
        if !read {
            return;
        }
        let actual = &pixels[(4 * 8 + x) * 4..(4 * 8 + x) * 4 + 4];
        for (c, &value) in expected.iter().enumerate() {
            assert!(
                (actual[c] as f32 - value * 255.0).abs() <= 3.0,
                "pixel {x}: {actual:?}, expected {expected:?}"
            );
        }
    };
    for (alpha, expected) in [
        (Alpha::Over, [0.5, 0.0, 0.25, 0.75]),
        (Alpha::Atop, [0.25, 0.0, 0.25, 0.5]),
        (Alpha::Out, [0.0, 0.0, 0.25, 0.25]),
        (Alpha::Conjoint, [0.5, 0.0, 0.0, 0.5]),
        (Alpha::Disjoint, [0.5, 0.0, 0.5, 1.0]),
    ] {
        plan.drawable_blends[1] = Moc3BlendConfig {
            color: Color::Normal,
            alpha,
        };
        let (pixels, _) = render(
            device,
            queue,
            &renderer,
            &mut resources,
            &meshes,
            &plan,
            read,
        );
        assert_pixel(&pixels, 4, expected);
    }
    plan.drawable_blends[1] = Moc3BlendConfig {
        color: Color::Lighten,
        alpha: Alpha::Over,
    };
    let (pixels, _) = render(
        device,
        queue,
        &renderer,
        &mut resources,
        &meshes,
        &plan,
        read,
    );
    assert_pixel(&pixels, 4, [0.5, 0.0, 0.5, 0.75]);
    // Distinguish unclamped AddGlow from Add while the final output stays below one.
    for (color, value) in [
        (Color::Add, 0.65),
        (Color::AddGlow, 0.8),
        (Color::Multiply, 0.56),
    ] {
        // The backdrop must be normal; give the second draw its own configuration.
        let mut pair = meshes.to_vec();
        pair.push(meshes[4].clone());
        plan.nodes = vec![Node::Drawable(4), Node::Drawable(5)];
        plan.drawable_blends[4] = Default::default();
        plan.drawable_blends.truncate(5);
        plan.drawable_blends.push(Moc3BlendConfig {
            color,
            alpha: Alpha::Over,
        });
        let (pixels, _) = render(device, queue, &renderer, &mut resources, &pair, &plan, read);
        assert_pixel(&pixels, 4, [value, value, value, 0.75]);
    }
    plan.nodes = vec![Node::Offscreen(0)];
    plan.offscreens = vec![
        group(vec![Node::Offscreen(1)], 0.5),
        group(vec![Node::Drawable(2), Node::Drawable(2)], 0.5),
    ];
    let (pixels, stats) = render(
        device,
        queue,
        &renderer,
        &mut resources,
        &meshes,
        &plan,
        read,
    );
    assert_pixel(&pixels, 4, [0.25; 4]);
    assert_eq!(stats.allocated_layers, 3);
    plan.offscreens = vec![group(vec![Node::Drawable(2)], 0.5)];
    plan.offscreens[0].multiply_color = [0.5, 0.25, 1.0];
    plan.offscreens[0].screen_color = [0.5, 0.0, 0.0];
    let (pixels, _) = render(
        device,
        queue,
        &renderer,
        &mut resources,
        &meshes,
        &plan,
        read,
    );
    assert_pixel(&pixels, 4, [0.375, 0.125, 0.5, 0.5]);
    // Transparent mask meshes still contribute their texture alpha. Inversion must
    // be applied after sampling; group opacity must be applied only once.
    plan.offscreens = vec![group(vec![Node::Drawable(2)], 0.5)];
    plan.offscreens[0].masks = vec![3];
    for inverted in [false, true] {
        plan.offscreens[0].inverted_mask = inverted;
        let (pixels, stats) = render(
            device,
            queue,
            &renderer,
            &mut resources,
            &meshes,
            &plan,
            read,
        );
        assert_pixel(&pixels, 1, [if inverted { 0.0 } else { 0.5 }; 4]);
        assert_pixel(&pixels, 6, [if inverted { 0.5 } else { 0.0 }; 4]);
        assert_eq!(stats.allocated_layers, 3);
    }
}

#[test]
fn validates_nested_composition_and_all_alpha_pipelines() {
    let (device, queue) = wgpu::Device::noop(&Default::default());
    cases(&device, &queue, false);
}

#[test]
#[ignore = "requires a native GPU adapter; checks pixel output rather than the noop backend"]
fn gpu_composition_pixels_match_porter_duff_and_nested_masks() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        eprintln!("composition adapter: {}", adapter.get_info().name);
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        cases(&device, &queue, true);
    });
}
