//! Opt-in regression coverage; the third-party model is kept outside the repo.
use neocari::{
    assets::load_model_runtime,
    expression::{ExpressionPlayer, load_expression},
    render::common::{ClippingPlan, DrawableInfo},
};

#[test]
#[ignore = "set NEOCARI_TEST_HOTARU_MODEL to hotaru_live_01's .model3.json"]
fn hotaru_expressions_preserve_blend_shape_meshes() {
    let path = std::path::PathBuf::from(
        std::env::var_os("NEOCARI_TEST_HOTARU_MODEL").expect("set NEOCARI_TEST_HOTARU_MODEL"),
    );
    let dir = path.parent().unwrap();
    let mut model = load_model_runtime(&path).unwrap();
    let expressions = model.runtime().model().expressions().to_vec();
    assert_eq!(expressions.len(), 46);
    for reference in expressions {
        let mut player =
            ExpressionPlayer::new(load_expression(dir.join(reference.file())).unwrap());
        for _ in 0..4 {
            let runtime = model.runtime_mut();
            runtime.reset_parameters();
            player.tick(0.125);
            player.apply(runtime);
            runtime.update_meshes().unwrap();
            let infos: Vec<_> = runtime
                .meshes()
                .iter()
                .map(DrawableInfo::from_mesh)
                .collect();
            let mut clipping = ClippingPlan::from_drawables(&infos);
            clipping.prepare_single_texture_masks(&infos).unwrap();
            assert!(
                runtime.meshes().iter().all(|m| {
                    m.opacity().is_finite()
                        && m.vertices()
                            .iter()
                            .all(|v| v.position().iter().all(|p| p.is_finite()))
                }),
                "{}",
                reference.name()
            );
        }
        let required_meshes: &[&str] = match reference.name() {
            "exp_heart01" => &["ArtMesh99", "ArtMesh142"],
            "exp_kirakira01" => &["ArtMesh100", "ArtMesh149"],
            "exp_spin01" => &["ArtMesh101", "ArtMesh152"],
            "exp_cry03" | "exp_cry04" | "exp_shy01" => {
                &["ArtMesh6", "ArtMesh7", "ArtMesh66", "ArtMesh60"]
            }
            _ => &[],
        };
        for id in required_meshes {
            let runtime = model.runtime();
            let index = runtime.drawable_index(id).unwrap();
            assert!(
                runtime.meshes()[index].opacity() > 0.0,
                "{}: {id} disappeared",
                reference.name()
            );
        }
    }
}
