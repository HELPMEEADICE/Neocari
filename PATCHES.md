# Local renderer changes

## Packed animated mesh buffers (2026-10-27)

`WgpuMeshBuffers::from_drawables` now shares a vertex and index allocation among
animated drawables, using the existing packed rendering path. Geometry bytes,
per-drawable change detection, upload offsets, masks and draw ordering are
unchanged. Static and animated geometry share the packing implementation;
animated vertices additionally allow `COPY_DST` and retain update snapshots.
Models exceeding one buffer's device limit retain the separate allocation path.

The opt-in `packed_animation_matches_unpacked_pixels` test compares rendered
RGBA bytes against the previous separate-buffer path on a real GPU, before and
after animation updates. `oversized_packed_buffer_keeps_the_per_drawable_fallback`
checks the device-limit fallback with the noop backend.

```powershell
cargo test --manifest-path vendor/neocari/Cargo.toml --features wgpu --lib packed_tests -- --include-ignored --nocapture
cargo test --manifest-path vendor/neocari/Cargo.toml --features wgpu --test wgpu_renderer
```

This file records this round of changes; the checked-in submodule history also
contains earlier application-specific runtime fixes.
