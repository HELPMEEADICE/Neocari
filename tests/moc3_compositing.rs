use neocari::{
    assets::load_model_runtime,
    moc3::{
        Moc3AlphaBlendMode as Alpha, Moc3BlendConfig, Moc3ColorBlendMode as Color, Moc3CountInfo,
        Moc3OffscreenInfo, Moc3RenderNode as Node, Moc3RenderPlan, Moc3SectionOffsets,
    },
};

fn walk(nodes: &[Node], plan: &Moc3RenderPlan, meshes: &mut Vec<usize>, groups: &mut Vec<usize>) {
    for node in nodes {
        match *node {
            Node::Drawable(index) => meshes.push(index),
            Node::Offscreen(index) => {
                assert!(!groups.contains(&index));
                groups.push(index);
                walk(&plan.offscreens[index].nodes, plan, meshes, groups);
            }
        }
    }
}

#[test]
fn ren_preserves_nested_groups_and_authored_composition() {
    let mut model = load_model_runtime("assets/models/Ren/Ren.model3.json").unwrap();
    let plan = model.runtime().render_plan().unwrap();
    let mut meshes = vec![];
    let mut groups = vec![];
    walk(&plan.nodes, &plan, &mut meshes, &mut groups);
    meshes.sort_unstable();
    groups.sort_unstable();
    assert_eq!(meshes, (0..198).collect::<Vec<_>>());
    assert_eq!(groups, (0..24).collect::<Vec<_>>());
    assert!(
        plan.offscreens
            .iter()
            .any(|g| g.nodes.iter().any(|n| matches!(n, Node::Offscreen(_))))
    );
    assert!(plan.offscreens.iter().any(|g| g.opacity == 0.0));
    assert!(
        plan.offscreens
            .iter()
            .any(|g| g.opacity > 0.0 && g.opacity < 1.0)
    );
    assert!(plan.offscreens.iter().any(|g| !g.masks.is_empty()));
    for alpha in [Alpha::Atop, Alpha::Out, Alpha::Disjoint] {
        assert!(plan.offscreens.iter().any(|g| g.blend.alpha == alpha));
    }
    assert!(
        plan.offscreens
            .iter()
            .any(|g| g.blend.color == Color::Multiply)
    );
    model.runtime_mut().set_parameter("ParamHologram", 1.0);
    model.runtime_mut().update_meshes().unwrap();
    for index in [94, 95, 98] {
        assert_eq!(
            model.runtime().meshes()[index].opacity(),
            1.0,
            "effect mesh {index} must reach its offscreen group"
        );
    }
}

#[test]
fn v53_counts_and_offscreen_ownership_are_validated() {
    let mut bytes = std::fs::read("assets/models/Ren/Ren.moc3").unwrap();
    let counts = Moc3CountInfo::parse(&bytes).unwrap();
    assert_eq!(counts.offscreens(), 24);
    assert_eq!(counts.offscreen_keyforms(), 27);
    let offsets = Moc3SectionOffsets::parse(&bytes).unwrap();
    let owner = offsets.section_offset(155).unwrap() as usize;
    bytes[owner..owner + 4].copy_from_slice(&(-1i32).to_le_bytes());
    assert!(Moc3OffscreenInfo::parse(&bytes).is_err());
}

#[test]
fn blend_configuration_validates_modes_and_normalizes_legacy_alpha() {
    assert_eq!(
        Moc3BlendConfig::from_raw(0x106).unwrap(),
        Moc3BlendConfig {
            color: Color::Multiply,
            alpha: Alpha::Atop
        }
    );
    assert_eq!(Moc3BlendConfig::from_raw(0x401).unwrap().alpha, Alpha::Over);
    for raw in [18, 0x500, 0x10000] {
        assert!(Moc3BlendConfig::from_raw(raw).is_err());
    }
}
