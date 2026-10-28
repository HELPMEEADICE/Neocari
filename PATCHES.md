# Local renderer changes

## Cubism 2 null object references (2026-10-28)

The MOC reader now retains a reference-table slot for every null object, matching
`Live2D-v2-Lua`'s `live2d/core/io/binary_reader.lua`. Object references can resolve
to null without allocating another slot. Previously, a null shifted subsequent
references onto unrelated objects. In Mashiro's `school_summer-2023` costume,
`B_ARM_R.18` consequently used `B_ARM_R_c.01` instead of `B_ARM_R_c.02`, placing
part of the right arm beside the ear.

Two parser unit tests cover parent-ID aliasing after a null and references to a
null slot. Both fail with the previous reader. Local validation used standalone
`rustc` harnesses importing the parser, scene and deformer sources; no local
`cargo` commands were run. All 10 unit tests passed. The costume's 4,327 vertices
also matched Lua geometry at default parameters and across 99 sampled poses from
all 33 motions (428,373 animated vertices, maximum error 0.020 model pixels,
within the existing reference-comparison tolerance).

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
