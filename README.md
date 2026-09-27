# Neocari

A pure Rust Live2D/Cubism runtime derived from Mocari, with the goal of adding native Cubism V2 model support to Rust.

## Lineage

Neocari is a derivative project based on [Mocari](https://github.com/Eatgrapes/Mocari), originally created by [Eatgrapes](https://github.com/Eatgrapes). This fork focuses on enabling Rust applications to load and render Cubism V2 models natively, without requiring Live2D Cubism Core binaries.

Neocari is an unofficial and independent community project. It is not affiliated with, endorsed by, sponsored by, or certified by Live2D Inc. This repository does not include Live2D Cubism Core, SDK binaries, official source code, or proprietary Live2D files. “Live2D” and “Cubism” are trademarks of their respective owners. Users are responsible for complying with the licenses of Live2D Inc., model creators, asset owners, and applicable laws. Rights holders can contact the maintainer or open an issue to request a review of repository content.

## Features

- Parse and render Cubism V2 `.moc` models at their declared default parameter values.
- Load Cubism V2 model settings JSON and referenced PNG textures.
- Render Cubism V3/V4/V5 `.moc3` models and use the parameter, motion, expression, and physics runtime.
- Use the built-in `wgpu` renderer or provide a custom backend through `render::common`.

Cubism V2 motion playback and interactive V2 parameter updates are not implemented yet.

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

The `assets::load_moc2_model` loader accepts a Cubism V2 model settings JSON and loads its `.moc` file and PNG textures. The generated meshes can be passed to the same WGPU renderer used for Cubism V3.

```bash
cargo run --features wgpu --example render_moc2 -- path/to/model.model.json output.png
```

## License

Neocari is distributed under the GNU General Public License, version 3 only. See [LICENSE](LICENSE). Neocari is derived from MIT-licensed Mocari; the original MIT license and Eatgrapes copyright notice are preserved in [LICENSES/MIT.txt](LICENSES/MIT.txt).
