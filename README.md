# Neocari

A pure Rust Live2D/Cubism runtime derived from Mocari, with the goal of adding native Cubism V2 model support to Rust.

## Lineage

Neocari is a derivative project based on [Mocari](https://github.com/Eatgrapes/Mocari), originally created by [Eatgrapes](https://github.com/Eatgrapes). This fork focuses on enabling Rust applications to load and render Cubism V2 models natively, without requiring Live2D Cubism Core binaries.

Neocari is an unofficial and independent community project. It is not affiliated with, endorsed by, sponsored by, or certified by Live2D Inc. This repository does not include Live2D Cubism Core, SDK binaries, official source code, or proprietary Live2D files. “Live2D” and “Cubism” are trademarks of their respective owners. Users are responsible for complying with the licenses of Live2D Inc., model creators, asset owners, and applicable laws. Rights holders can contact the maintainer or open an issue to request a review of repository content.

## Features

- Full Cubism V2 (`.moc`) runtime:
  - parameter pivot tables with multi-axis interpolation and outside-range handling
  - warp and rotation deformers with parent hierarchies, scale, opacity, and extrapolation
  - part visibilities, part opacity, draw order, clipping masks, and blend modes
  - `.mtn` motion playback with fade in/out, priorities, and smoothing
  - `.exp` expressions with add, multiply, and overwrite blending
  - `.physics` hair chains, `.pose` part fading, automatic eye blinking, and breathing
- Parse and render Cubism V2 `.moc` models at their default parameters.
- Load Cubism V2 model settings JSON and referenced PNG textures.
- Render Cubism V3/V4/V5 `.moc3` models and use the parameter, motion, expression, and physics runtime.
- Evaluate MOC3 Blend Shapes, inherited colors, nested rotations, and Cubism 5.3 offscreen groups.
- Use the built-in `wgpu` renderer or provide a custom backend through `render::common`.

## Build from source

Requirements: Rust and Cargo.

```bash
git clone https://github.com/HELPMEEADICE/Neocari.git
cd Neocari
cargo build
```

The default build has no renderer dependency. Enable the built-in WGPU renderer with:

```bash
cargo build --features wgpu
```

To use the library from another project:

```toml
[dependencies]
neocari = { git = "https://github.com/HELPMEEADICE/Neocari", features = ["wgpu"] }
```

## Render a Cubism V2 model

`assets::load_moc2_model` loads a Cubism V2 model settings JSON with its `.moc`
file and PNG textures and produces default-pose drawables. For animation use
`assets::load_moc2_runtime`, which also loads the physics and pose files declared
by the settings and returns a `Moc2RuntimeModel`:

```rust
use neocari::assets::load_moc2_runtime;

fn animate() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = load_moc2_runtime("path/to/model.model.json")?;
    model.play_motion("idle", 0)?;      // .mtn motion declared by the settings file
    model.play_expression("smile")?;    // .exp expression declared by the settings file
    model.runtime_mut().set_drag(0.25, -0.1);

    for _ in 0..60 {
        model.update(1.0 / 60.0);
        for mesh in model.meshes() {
            // Upload mesh.vertices()/indices() and draw with the shared drawable renderer.
        }
    }
    Ok(())
}
```

The runtime reproduces the Cubism V2 update pipeline: part/pose state, motions
with fade windows and priorities, automatic blinking, expressions, drag/breath
offsets, hair physics, pose fading, then deformer and drawable evaluation.

`Moc2Runtime` can also be driven directly when motion and expression files are
loaded elsewhere:

```rust
use std::sync::Arc;
use neocari::moc2::{Moc2Motion, Moc2Runtime};

fn play(runtime: &mut Moc2Runtime, motion: Moc2Motion) {
    runtime.start_motion(Arc::new(motion), 500.0, 500.0);
    runtime.update(1.0 / 60.0);
}
```

The `assets::load_moc2_model` loader can still be used for previews and import
tools. For those fixed default-pose meshes, `WgpuMeshBuffers::from_static_drawables`
packs geometry into shared GPU buffers and avoids update snapshots.

Render a static frame:

```bash
cargo run --features wgpu --example render_moc2 -- path/to/model.model.json output.png
```

Render animation frames (motion, expression, physics, blinking, and breathing):

```bash
cargo run --features wgpu --example render_moc2_animation -- \
    path/to/model.model.json frame.png --motion idle01 --expression smile01 --frames 30
```

## Render a MOC3 model with offscreen composition

The `show_model` example uses `ModelRuntime::render_plan()` and
`WgpuLive2dRenderer::encode_model()` to preserve nested offscreen groups, their
opacity, colors, masks, and separate color/alpha blend modes. Custom renderers
must follow that tree: flattening its meshes changes the result for models such
as Ren. The existing `draw_with_*` methods remain useful for flat drawable lists
and UI overlays.

After updating mesh buffers and clipping resources, build the current render
plan and pass it in a `WgpuModelFrame` to `encode_model`, with a
`WgpuModelTarget` describing the output view, dimensions, and load operation.
Create `WgpuModelResources` once with `create_model_resources` and reuse it:
scratch textures are pooled by nesting depth and rebuilt when dimensions change.
Models with ordinary blend modes and no offscreen groups draw directly to the
target.

See [alignment validation](docs/ayagami-alignment.md) for the models, regression
coverage, and reference discrepancies checked during this work.

## License

Neocari is distributed under the GNU General Public License, version 3 only. See [LICENSE](LICENSE). Neocari is derived from MIT-licensed Mocari; the original MIT license and Eatgrapes copyright notice are preserved in [LICENSES/MIT.txt](LICENSES/MIT.txt).
