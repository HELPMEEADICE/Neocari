# MOC3 alignment validation

The local ayagami checkout was used as an independent reference for MOC3
evaluation. The comparison feeds identical parameter and part opacity values
to each runtime, then compares geometry (with the Y-axis convention adjusted),
opacity, multiply/screen colors, quantized draw order, visibility, and the
ordered offscreen tree. Physics and pose playback are excluded from this
evaluator comparison. Non-renderable helper meshes are excluded from tree
ordering comparisons because their order has no effect on rendering.

The audit covers hotaru_live_01 and the bundled Haru, Hiyori, Mao, Mark, Natori,
Ren, Rice, and Wanko models. Cases include default parameters, every declared
expression at five fade times, every motion at six times, five values per
parameter, and 100 deterministic combined parameter states per model. The
comparison tolerance is 5e-5. All 5,860 states pass when using the reference
control described below for Mao.

Corrections include Blend Shape deltas for meshes, warps, rotations, parts and
glue; global color selectors; triangular/bilinear warp selection; nested
rotation reflection; pivot snapping and out-of-range visibility; local part
opacity inheritance; and excluding meshes without triangles from glue.

Cubism 5.3 rendering now reads the extended count/offset tables and retains
offscreen groups in the render plan. The WGPU encoder composites nested groups
with authored opacity and colors, normal/inverted masks, and independent color
and alpha blend modes. Transparent mask sources still supply coverage.

## Reference discrepancies

Mao exposes a reference issue in `ayagami/src/driver/deformer.rs`: calculating a
rotation below a warp applies the parent's visual state again while probing
geometry. This repeatedly adds screen color. A control run against a temporary
copy, changing only the probe to use a throwaway visual state, removes the color
differences. The original reference checkout is unmodified; Neocari does not
reproduce the duplicated color propagation.

The reference shader also implements Lighten with `min`. Neocari uses `max`.
Add and AddGlow are distinct: only Add clamps the intermediate blended color.

For GPU comparison, the reference renderer uses mipmapped, premultiplied sRGB
textures, while Neocari's existing texture API uses straight RGBA8 with one mip.
A temporary reference copy was configured to use that same sampling path to
isolate composition from texture filtering. Ren's default, hologram,
transparency, and night frames then differed by at most 3/255 in any channel
at 800 x 800 on an NVIDIA RTX 5080 Laptop GPU. The original renderer sources
were not changed.

## Regression checks

```powershell
cargo test
cargo test --features wgpu
cargo clippy --all-targets --features wgpu -- -D warnings

# Native GPU pixel readback: alpha composition, nested opacity, inverted masks,
# transparent mask sources, packed draw offsets, Add/AddGlow and Multiply.
cargo test --features wgpu --test wgpu_compositing -- --ignored --nocapture

# Optional test against the external hotaru model (all 46 expressions).
$env:NEOCARI_TEST_HOTARU_MODEL = 'C:/oss/hotaru_live_01/adv_live2d_hotaru_016_live_01.model3.json'
cargo test --test hotaru_expressions -- --ignored
```

These checks establish parity for the audited inputs, not full conformance of
every Cubism feature or pixel identity across texture sampling implementations.
