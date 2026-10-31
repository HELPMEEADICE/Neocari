# Local renderer changes

## Cubism 2 deformers with an empty ID (2026-10-31)

An empty string is now an ordinary deformer ID, matching `Live2D-v2-Lua`'s
`ModelContext:init()` and `getDeformerIndex()`, which compare IDs as-is.
Previously, `convert_deformers` and the drawable lookup skipped empty IDs, so a
deformer or drawable whose target is `""` never resolved. The whole model then
failed with `deformer hierarchy is cyclic or references a missing parent`, or
with `missing drawable deformer` when a drawable hung directly from the unnamed
deformer. When several deformers share an ID, the first one wins, as in the Lua
lookup.

72 BanG Dream! costumes contain exactly one unnamed deformer: 70 of Yukina's
100 costumes, where it sits between `B_FACE.07` (once `B_FACE.13`) and the
front hair, plus Mashiro's `vocal_limited_ssr` and Tomoe's `live_event_279_ur`.
Treating `""` as the root instead loads these models but detaches the front
hair from the face, so it is drawn in the wrong place.

The parent-first ordering moved into `order_deformers()`. Three parser unit
tests cover an unnamed parent listed after its child, duplicate IDs, and
missing or cyclic parents, which are still rejected. Loading every costume of
64 characters (2,954 models, moc2 and moc3) through BandoriPet's archive cache
went from 2,880 to 2,952 successes with no model newly failing. The remaining
two failures are unrelated: Yukina's `live_sr_01` targets a nonexistent
`B_FACE.08`, and Ritsu's `school_winter_hs` reports an invalid blend shape
weight limit. Yukina's `dream_festival_4_ur` renders with its front hair in the
same place as the Lua renderer.

## Coalesced animated uploads and late Cubism 2 overrides (2026-10-29)

Animated packed buffers now combine consecutive changed drawables into one queue
write, reusing the encoding scratch allocation. Unchanged drawables split dirty
runs, so sparse animation does not upload extra geometry. Empty drawables do not
introduce gaps. Oversized models retain separate buffers and individual writes.
`WgpuMeshUpdate::upload_calls` reports the number of vertex queue writes while
`uploaded_drawables` retains its previous meaning.

`Moc2Runtime::update_with_parameters` runs final overrides after physics and pose
but before geometry evaluation. Applications can apply lip sync at the same
point in the animation pipeline without evaluating every deformer and drawable
twice. `update` preserves its existing behavior through a no-op callback.

Unit tests cover coalesced and sparse writes, unchanged frames, the device-limit
fallback, and matching parameter/saved-parameter/mesh state against two geometry
evaluations. The real-GPU packed/unpacked pixel regression additionally exercises
sparse changes, colors, visibility, draw order, masks and an empty drawable.

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
