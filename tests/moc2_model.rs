//! End-to-end Cubism 2 checks against a model supplied by the environment.
//!
//! Proprietary `.moc` samples are not part of this repository. Point
//! `NEOCARI_MOC2_MODEL` at a `.model.json` file to exercise these tests:
//!
//! ```text
//! NEOCARI_MOC2_MODEL=/path/to/model.model.json cargo test --test moc2_model
//! ```

use std::path::PathBuf;

use neocari::assets::{load_moc2_model, load_moc2_runtime};

fn model_path() -> Option<PathBuf> {
    std::env::var_os("NEOCARI_MOC2_MODEL").map(PathBuf::from)
}

fn mesh_positions(model: &neocari::assets::Moc2RuntimeModel) -> Vec<[f32; 2]> {
    model
        .meshes()
        .iter()
        .flat_map(|mesh| mesh.vertices().iter().map(|vertex| vertex.position()))
        .collect()
}

#[test]
fn loads_cubism2_model_when_configured() {
    let Some(path) = model_path() else {
        eprintln!("NEOCARI_MOC2_MODEL is not set; skipping Cubism 2 model test");
        return;
    };

    let model = load_moc2_model(&path).expect("load Cubism 2 model");
    assert!(model.model().canvas_width() > 0);
    assert!(model.model().canvas_height() > 0);
    assert!(!model.drawables().is_empty());
    assert!(!model.model().scene().parameters().is_empty());
    assert!(!model.textures().is_empty());

    for drawable in model.drawables() {
        for vertex in drawable.vertices() {
            assert!(vertex.position().iter().all(|value| value.is_finite()));
            assert!(vertex.uv().iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn animates_cubism2_model_when_configured() {
    let Some(path) = model_path() else {
        return;
    };

    let mut model = load_moc2_runtime(&path).expect("load Cubism 2 runtime");
    assert!(model.runtime().has_physics());
    model.runtime_mut().set_auto_breath(false);
    model.runtime_mut().set_auto_blink(false);

    model.update(1.0 / 60.0).expect("first update");
    let initial = mesh_positions(&model);

    let started = model
        .play_motion_group("idle01")
        .expect("load idle01 motion");
    assert!(started);

    for _ in 0..30 {
        model.update(1.0 / 60.0).expect("update");
    }
    let animated = mesh_positions(&model);
    assert_ne!(initial, animated, "motion should deform the model");
    assert!(
        model
            .runtime()
            .parameter_value("PARAM_ANGLE_X")
            .is_some_and(|value| value != 0.0),
        "motion should drive PARAM_ANGLE_X"
    );
    assert!(
        model
            .runtime()
            .physics()
            .is_some_and(|physics| physics.hair_count() > 0)
    );
}

#[test]
fn applies_cubism2_expression_when_configured() {
    let Some(path) = model_path() else {
        return;
    };

    let mut model = load_moc2_runtime(&path).expect("load Cubism 2 runtime");
    model.runtime_mut().set_auto_breath(false);
    model.runtime_mut().set_auto_blink(false);
    model.update(0.0).expect("update");

    let started = model.play_expression("smile01").expect("load expression");
    assert!(started);

    for _ in 0..30 {
        model.update(1.0 / 60.0).expect("update");
    }
    assert!(
        model
            .runtime()
            .parameter_value("PARAM_EYE_L_SMILE")
            .is_some_and(|value| value > 0.5),
        "smile01 should raise PARAM_EYE_L_SMILE"
    );
}
