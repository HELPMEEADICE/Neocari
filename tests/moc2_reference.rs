//! Compares the Neocari Cubism 2 runtime against reference geometry dumps.
//!
//! The dumps are produced by the reference Lua implementation. Each frame uses:
//!
//! ```text
//! PARAMS <n>
//! P <id> <value>
//! DRAWABLES <n>
//! D <id> <interpolatedOpacity> <baseOpacity> <drawOrder> <partsIndex> <outside> <available>
//! V <count>
//! <x/y coordinates, one per line>
//! ```
//!
//! Single-frame dumps start directly with `PARAMS`; animation dumps repeat the
//! block after a `FRAME <index> <timeMs>` header.
//!
//! Environment variables:
//!
//! - `NEOCARI_MOC2_MOC` and `NEOCARI_MOC2_DUMP`: static geometry comparison.
//! - `NEOCARI_MOC2_MODEL` and `NEOCARI_MOC2_ANIMATION_DUMP`: animated comparison.

use std::{collections::HashMap, path::PathBuf};

use neocari::{
    assets::load_moc2_runtime,
    moc2::{Moc2Model, Moc2Runtime},
};

struct ReferenceDrawable {
    id: String,
    interpolated_opacity: f32,
    base_opacity: f32,
    draw_order: i32,
    part_index: usize,
    outside: bool,
    points: Vec<f32>,
}

struct ReferenceFrame {
    parameters: Vec<(String, f32)>,
    drawables: Vec<ReferenceDrawable>,
}

fn parse_frame(lines: &mut impl Iterator<Item = &'static str>) -> Option<ReferenceFrame> {
    let parameter_count = lines
        .next()?
        .strip_prefix("PARAMS ")?
        .trim()
        .parse::<usize>()
        .expect("parameter count");

    let mut parameters = Vec::with_capacity(parameter_count);
    for _ in 0..parameter_count {
        let line = lines.next().expect("parameter line");
        let mut parts = line.splitn(3, ' ');
        assert_eq!(parts.next(), Some("P"));
        let id = parts.next().expect("parameter id").to_owned();
        let value = parts
            .next()
            .expect("parameter value")
            .trim()
            .parse::<f32>()
            .expect("parameter value");
        parameters.push((id, value));
    }

    let drawable_count = lines
        .next()
        .expect("drawable header")
        .strip_prefix("DRAWABLES ")
        .expect("drawable header")
        .trim()
        .parse::<usize>()
        .expect("drawable count");

    let mut drawables = Vec::with_capacity(drawable_count);
    for _ in 0..drawable_count {
        let line = lines.next().expect("drawable line");
        let fields = line.split_whitespace().collect::<Vec<_>>();
        assert_eq!(fields[0], "D");
        let interpolated_opacity = fields[2].parse().unwrap();
        let base_opacity = fields[3].parse().unwrap();
        let draw_order = fields[4].parse().unwrap();
        let part_index = fields[5].parse().unwrap();
        let outside = fields[6] == "1";

        let value_count = lines
            .next()
            .and_then(|line| line.strip_prefix("V "))
            .and_then(|value| value.trim().parse::<usize>().ok())
            .expect("vertex value count");
        let mut points = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            points.push(lines.next().unwrap().trim().parse::<f32>().unwrap());
        }

        drawables.push(ReferenceDrawable {
            id: fields[1].to_owned(),
            interpolated_opacity,
            base_opacity,
            draw_order,
            part_index,
            outside,
            points,
        });
    }

    Some(ReferenceFrame {
        parameters,
        drawables,
    })
}

fn parse_static_dump(source: &'static str) -> ReferenceFrame {
    let mut lines = source.lines();
    parse_frame(&mut lines).expect("first frame")
}

fn parse_animation_dump(source: &'static str) -> Vec<(usize, f32, ReferenceFrame)> {
    let mut lines = source.lines().peekable();
    let mut frames = Vec::new();

    while let Some(line) = lines.next() {
        let Some(header) = line.strip_prefix("FRAME ") else {
            continue;
        };
        let mut fields = header.split_whitespace();
        let index = fields.next().unwrap().parse::<usize>().unwrap();
        let time_ms = fields.next().unwrap().parse::<f32>().unwrap();
        let frame = parse_frame(&mut lines).expect("frame payload");
        frames.push((index, time_ms, frame));
    }

    frames
}

fn static_paths() -> Option<(PathBuf, PathBuf)> {
    let moc = std::env::var_os("NEOCARI_MOC2_MOC")?;
    let dump = std::env::var_os("NEOCARI_MOC2_DUMP")?;
    Some((PathBuf::from(moc), PathBuf::from(dump)))
}

fn animation_paths() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("NEOCARI_MOC2_MODEL")?;
    let dump = std::env::var_os("NEOCARI_MOC2_ANIMATION_DUMP")?;
    Some((PathBuf::from(model), PathBuf::from(dump)))
}

fn compare_drawables(
    runtime: &Moc2Runtime,
    reference: &ReferenceFrame,
    frame_label: &str,
    compared_points: &mut usize,
) {
    assert_eq!(
        runtime.meshes().len(),
        reference.drawables.len(),
        "{frame_label}: drawable count"
    );

    for (index, (mesh, expected)) in runtime
        .meshes()
        .iter()
        .zip(&reference.drawables)
        .enumerate()
    {
        let drawable = &runtime.scene().drawables()[index];
        assert_eq!(
            drawable.id(),
            expected.id,
            "{frame_label}: drawable {index} id"
        );
        assert_eq!(
            mesh.vertices().len() * 2,
            expected.points.len(),
            "{frame_label}: drawable {index} vertex count"
        );
        assert_eq!(
            mesh.draw_order() as i32,
            expected.draw_order,
            "{frame_label}: drawable {index} draw order"
        );
        assert_eq!(
            drawable.part_index(),
            expected.part_index,
            "{frame_label}: drawable {index} part"
        );

        if expected.outside {
            assert_eq!(
                mesh.opacity(),
                0.0,
                "{frame_label}: outside drawable {index}"
            );
            continue;
        }

        let part_opacity = if runtime.scene().parts()[expected.part_index].is_visible() {
            1.0
        } else {
            0.0
        };
        let expected_opacity = expected.interpolated_opacity * part_opacity * expected.base_opacity;
        assert!(
            (mesh.opacity() - expected_opacity).abs() <= 1e-4,
            "{frame_label}: drawable {index} opacity: ours {} reference {expected_opacity}",
            mesh.opacity()
        );

        for (vertex_index, vertex) in mesh.vertices().iter().enumerate() {
            let actual = vertex.position();
            let expected_position = [
                expected.points[vertex_index * 2],
                expected.points[vertex_index * 2 + 1],
            ];
            // The reference evaluates in f64 (Lua numbers); Neocari runs in f32.
            let tolerance = |expected: f32| 1e-2 + 1e-5 * expected.abs();
            assert!(
                (actual[0] - expected_position[0]).abs() <= tolerance(expected_position[0])
                    && (actual[1] - expected_position[1]).abs() <= tolerance(expected_position[1]),
                "{frame_label}: drawable {index} vertex {vertex_index}: ours {actual:?} reference {expected_position:?}"
            );
            *compared_points += 1;
        }
    }
}

fn compare_parameters(
    runtime: &Moc2Runtime,
    frame_label: &str,
    reference_values: &HashMap<String, f32>,
) {
    for (index, parameter) in runtime.scene().parameters().iter().enumerate() {
        let Some(expected) = reference_values.get(parameter.id()) else {
            continue;
        };
        let actual = runtime.parameter_values()[index];
        assert!(
            (actual - expected).abs() <= 1e-4 + 1e-5 * expected.abs(),
            "{frame_label}: parameter {} ours {actual} reference {expected}",
            parameter.id()
        );
    }
}

#[test]
fn geometry_matches_the_reference_runtime() {
    let Some((moc_path, dump_path)) = static_paths() else {
        eprintln!("NEOCARI_MOC2_MOC/NEOCARI_MOC2_DUMP are not set; skipping comparison");
        return;
    };

    let bytes = std::fs::read(moc_path).expect("read moc");
    let model = Moc2Model::from_bytes(&bytes).expect("parse moc");
    let mut runtime = model.into_runtime().expect("build runtime");
    runtime.update_meshes().expect("update meshes");

    let source = std::fs::read_to_string(dump_path).expect("read dump");
    let frame = parse_static_dump(Box::leak(source.into_boxed_str()));

    assert_eq!(
        runtime.parameter_values().len(),
        frame.parameters.len(),
        "parameter count"
    );
    let reference_values = frame.parameters.iter().cloned().collect::<HashMap<_, _>>();
    let mut compared_points = 0;
    compare_parameters(&runtime, "static", &reference_values);
    compare_drawables(&runtime, &frame, "static", &mut compared_points);
    assert!(compared_points > 0);
}

#[test]
fn animation_matches_the_reference_runtime() {
    let Some((model_path, dump_path)) = animation_paths() else {
        eprintln!("NEOCARI_MOC2_MODEL/NEOCARI_MOC2_ANIMATION_DUMP are not set; skipping");
        return;
    };

    let source = std::fs::read_to_string(dump_path).expect("read dump");
    let frames = parse_animation_dump(Box::leak(source.into_boxed_str()));
    assert!(!frames.is_empty());

    let mut model = load_moc2_runtime(model_path).expect("load runtime model");
    model.runtime_mut().set_auto_blink(false);
    model.runtime_mut().set_auto_breath(false);

    // The reference dump starts `idle01` with zero fade times.
    let reference = model
        .settings()
        .motions("idle01")
        .and_then(|motions| motions.first())
        .and_then(|motion| motion.file())
        .expect("idle01 motion reference")
        .to_owned();
    let motion_path = model
        .model_dir()
        .expect("model directory")
        .join(reference.replace('\\', "/"));
    let motion = std::sync::Arc::new(
        neocari::moc2::Moc2Motion::from_bytes(&std::fs::read(motion_path).expect("read motion"))
            .expect("parse motion"),
    );
    model.runtime_mut().start_motion(motion, 0.0, 0.0);

    let mut compared_points = 0;
    for (frame_index, _time_ms, frame) in &frames {
        let delta = if *frame_index == 0 { 0.0 } else { 1.0 / 60.0 };
        model.update(delta).expect("update");

        let reference_values = frame.parameters.iter().cloned().collect::<HashMap<_, _>>();
        let label = format!("frame {frame_index}");
        compare_parameters(model.runtime(), &label, &reference_values);
        compare_drawables(model.runtime(), frame, &label, &mut compared_points);
    }

    assert!(compared_points > 0);
}

#[test]
fn expression_matches_the_reference_runtime() {
    let Some(model_path) = std::env::var_os("NEOCARI_MOC2_MODEL") else {
        return;
    };
    let Some(dump_path) = std::env::var_os("NEOCARI_MOC2_EXPRESSION_DUMP") else {
        eprintln!("NEOCARI_MOC2_EXPRESSION_DUMP is not set; skipping expression comparison");
        return;
    };

    let source = std::fs::read_to_string(dump_path).expect("read dump");
    let frames = parse_animation_dump(Box::leak(source.into_boxed_str()));
    assert!(!frames.is_empty());

    let mut model = load_moc2_runtime(model_path).expect("load runtime model");
    model.runtime_mut().set_auto_blink(false);
    model.runtime_mut().set_auto_breath(false);
    model.runtime_mut().clear_physics();

    let reference = model
        .settings()
        .expression_file("smile01")
        .expect("smile01 expression reference")
        .to_owned();
    let expression_path = model
        .model_dir()
        .expect("model directory")
        .join(reference.replace('\\', "/"));
    let expression = std::sync::Arc::new(
        neocari::moc2::Moc2Expression::from_json_str(
            &std::fs::read_to_string(expression_path).expect("read expression"),
        )
        .expect("parse expression"),
    );
    model.runtime_mut().start_expression(expression);

    let mut compared_points = 0;
    for (frame_index, _time_ms, frame) in &frames {
        let delta = if *frame_index == 0 { 0.0 } else { 1.0 / 60.0 };
        model.update(delta).expect("update");

        let reference_values = frame.parameters.iter().cloned().collect::<HashMap<_, _>>();
        let label = format!("expression frame {frame_index}");
        compare_parameters(model.runtime(), &label, &reference_values);
        compare_drawables(model.runtime(), frame, &label, &mut compared_points);
    }

    assert!(compared_points > 0);
}
