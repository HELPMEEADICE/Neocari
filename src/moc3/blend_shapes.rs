use crate::{Result, core::compute_keyform_axis_interval};

use super::{
    Moc3Header, Moc3SectionOffsets, Moc3Version,
    keyform_bindings::Moc3KeyformSlot,
    parse::{invalid_moc3, read_f32_section, read_i32_section, read_u32},
};

#[derive(Debug, Clone, PartialEq)]
struct ParameterMap {
    parameter: usize,
    keys: Vec<f32>,
    neutral: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FormMap {
    parameter_map: usize,
    begin: usize,
    count: usize,
    limits: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct WeightLimit {
    parameter: usize,
    points: Vec<(f32, f32)>,
}

/// Blend Shape forms contain additive deltas, in the same global keyform
/// arrays as the ordinary forms. The neutral key contributes no delta.
#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct BlendShapes {
    parameters: Vec<ParameterMap>,
    forms: Vec<FormMap>,
    warps: Vec<Vec<usize>>,
    meshes: Vec<Vec<usize>>,
    limits: Vec<WeightLimit>,
}

impl BlendShapes {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let header = Moc3Header::parse(bytes)?;
        if matches!(
            header.version(),
            Moc3Version::V3_0_0 | Moc3Version::V3_3_0 | Moc3Version::V4_0_0
        ) {
            return Ok(Self::default());
        }
        let offsets = Moc3SectionOffsets::parse(bytes)?;
        super::Moc3CountInfo::parse(bytes)?;
        let endian = header.endianness();
        let count = |index: usize| {
            read_u32(
                bytes,
                offsets.count_info_offset() as usize + index * 4,
                endian,
            ) as usize
        };
        let ints = |slot, len| read_i32_section(bytes, &offsets, slot, len, endian);
        let floats = |slot, len| read_f32_section(bytes, &offsets, slot, len, endian);
        let map_count = count(25);
        let form_count = count(26);
        if map_count == 0 && form_count == 0 {
            return Ok(Self::default());
        }
        // Every item needs at least one word in its section. Reject corrupt
        // counts before allocating target and ownership tables.
        for index in [2, 4, 5, 14, 25, 26, 27, 28, 29, 30, 31] {
            if count(index) > bytes.len() / 4 {
                return Err(invalid_moc3("Blend Shape count exceeds the file size"));
            }
        }
        let parameter_count = count(5);
        let parameter_begins = ints(115, parameter_count)?;
        let parameter_counts = ints(116, parameter_count)?;
        let mut owners = vec![None; map_count];
        for (parameter, (&begin, &len)) in
            parameter_begins.iter().zip(&parameter_counts).enumerate()
        {
            for owner in owners
                .get_mut(range(begin, len)?)
                .ok_or_else(|| invalid_moc3("Blend Shape parameter map range is invalid"))?
            {
                if owner.replace(parameter).is_some() {
                    return Err(invalid_moc3(
                        "Blend Shape parameter map has multiple owners",
                    ));
                }
            }
        }
        let key_begins = ints(117, map_count)?;
        let key_counts = ints(118, map_count)?;
        let neutrals = ints(119, map_count)?;
        let keys = floats(77, count(14))?;
        let parameters = (0..map_count)
            .map(|i| {
                let keys = keys
                    .get(range(key_begins[i], key_counts[i])?)
                    .ok_or_else(|| invalid_moc3("Blend Shape keys are outside key values"))?
                    .to_vec();
                let neutral = to_usize(neutrals[i], "Blend Shape neutral index")?;
                if neutral >= keys.len()
                    || keys.iter().any(|v| !v.is_finite())
                    || keys.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(invalid_moc3("invalid Blend Shape keys or neutral index"));
                }
                Ok(ParameterMap {
                    parameter: owners[i]
                        .ok_or_else(|| invalid_moc3("Blend Shape parameter map has no owner"))?,
                    keys,
                    neutral,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let limit_count = count(30);
        let limit_parameters = ints(132, limit_count)?;
        let point_begins = ints(133, limit_count)?;
        let point_counts = ints(134, limit_count)?;
        let values = floats(135, count(31))?;
        let weights = floats(136, count(31))?;
        let limits = (0..limit_count)
            .map(|i| {
                let parameter = to_usize(limit_parameters[i], "Blend Shape limit parameter")?;
                let points = range(point_begins[i], point_counts[i])?
                    .map(|j| Some((*values.get(j)?, *weights.get(j)?)))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        invalid_moc3("Blend Shape limit points are outside their section")
                    })?;
                if parameter >= parameter_count
                    || points.is_empty()
                    || points.iter().any(|(v, w)| !v.is_finite() || !w.is_finite())
                    || points.windows(2).any(|w| w[0].0 >= w[1].0)
                {
                    return Err(invalid_moc3("invalid Blend Shape weight limit"));
                }
                Ok(WeightLimit { parameter, points })
            })
            .collect::<Result<Vec<_>>>()?;
        let maps = ints(120, form_count)?;
        let begins = ints(121, form_count)?;
        let counts = ints(122, form_count)?;
        let limit_begins = ints(123, form_count)?;
        let limit_counts = ints(124, form_count)?;
        let limit_refs = ints(131, count(29))?;
        let forms = (0..form_count)
            .map(|i| {
                let parameter_map = to_usize(maps[i], "Blend Shape form parameter map")?;
                let begin = to_usize(begins[i], "Blend Shape form begin")?;
                let count = to_usize(counts[i], "Blend Shape form count")?;
                if parameters
                    .get(parameter_map)
                    .is_none_or(|p| p.keys.len() != count)
                {
                    return Err(invalid_moc3(
                        "Blend Shape form count does not match its keys",
                    ));
                }
                let limits = limit_refs
                    .get(range(limit_begins[i], limit_counts[i])?)
                    .ok_or_else(|| invalid_moc3("Blend Shape limit reference range is invalid"))?
                    .iter()
                    .map(|&r| {
                        let index = to_usize(r, "Blend Shape limit reference")?;
                        (index < limit_count)
                            .then_some(index)
                            .ok_or_else(|| invalid_moc3("invalid Blend Shape limit reference"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(FormMap {
                    parameter_map,
                    begin,
                    count,
                    limits,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let groups = |slot, group_count, owner_count| -> Result<Vec<Vec<usize>>> {
            let owners = ints(slot, group_count)?;
            let begins = ints(slot + 1, group_count)?;
            let counts = ints(slot + 2, group_count)?;
            let mut groups = vec![Vec::new(); owner_count];
            for i in 0..group_count {
                let owner = to_usize(owners[i], "Blend Shape target")?;
                let indices = range(begins[i], counts[i])?;
                if indices.end > form_count {
                    return Err(invalid_moc3("Blend Shape form map range is invalid"));
                }
                groups
                    .get_mut(owner)
                    .ok_or_else(|| invalid_moc3("invalid Blend Shape target"))?
                    .extend(indices);
            }
            Ok(groups)
        };
        Ok(Self {
            parameters,
            forms,
            warps: groups(125, count(27), count(2))?,
            meshes: groups(128, count(28), count(4))?,
            limits,
        })
    }

    pub(super) fn mesh_slots(&self, index: usize, values: &[f32]) -> Option<Vec<Moc3KeyformSlot>> {
        self.slots(
            self.meshes
                .get(index)
                .map(Vec::as_slice)
                .unwrap_or_default(),
            values,
        )
    }

    pub(super) fn warp_slots(&self, index: usize, values: &[f32]) -> Option<Vec<Moc3KeyformSlot>> {
        self.slots(
            self.warps.get(index).map(Vec::as_slice).unwrap_or_default(),
            values,
        )
    }

    fn slots(&self, forms: &[usize], values: &[f32]) -> Option<Vec<Moc3KeyformSlot>> {
        let mut slots = Vec::new();
        for &index in forms {
            let form = self.forms.get(index)?;
            let parameter = self.parameters.get(form.parameter_map)?;
            let interval =
                compute_keyform_axis_interval(&parameter.keys, *values.get(parameter.parameter)?)?;
            let mut limit = 1.0f32;
            for &index in &form.limits {
                let source = self.limits.get(index)?;
                let value = *values.get(source.parameter)?;
                let mut weight = source.points.last()?.1;
                if value <= source.points.first()?.0 {
                    weight = source.points.first()?.1;
                } else if let Some(pair) = source.points.windows(2).find(|p| value < p[1].0) {
                    let t = (value - pair[0].0) / (pair[1].0 - pair[0].0);
                    weight = pair[0].1 + (pair[1].1 - pair[0].1) * t;
                }
                limit = limit.min(weight);
            }
            for (key, weight) in [
                (interval.left_index(), 1.0 - interval.t()),
                (interval.left_index() + 1, interval.t()),
            ] {
                if key != parameter.neutral && weight * limit > 0.0 {
                    if key >= form.count {
                        return None;
                    }
                    slots.push(Moc3KeyformSlot {
                        local_index: 0,
                        global_index: Some(form.begin.checked_add(key)?),
                        weight: weight * limit,
                    });
                }
            }
        }
        Some(slots)
    }
}

fn range(begin: i32, count: i32) -> Result<std::ops::Range<usize>> {
    if count == 0 {
        return Ok(0..0);
    }
    let begin = to_usize(begin, "Blend Shape range begin")?;
    let count = to_usize(count, "Blend Shape range count")?;
    Ok(begin
        ..begin
            .checked_add(count)
            .ok_or_else(|| invalid_moc3("Blend Shape range overflows"))?)
}

fn to_usize(value: i32, name: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid_moc3(format!("{name} is negative")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moc3::{
        Moc3ArtMeshInfo, Moc3ArtMeshKeyformInfo, Moc3ArtMeshKeyforms, Moc3ArtMeshes, Moc3Deformers,
        Moc3KeyformBindings, build_moc3_drawable_meshes_with_parameters,
    };

    fn write_word(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn count(bytes: &mut [u8], index: usize, value: u32) {
        write_word(bytes, 0x2c0 + index * 4, value);
    }

    fn section(bytes: &mut Vec<u8>, slot: usize, values: &[i32]) {
        let offset = bytes.len() as u32;
        write_word(bytes, 0x40 + slot * 4, offset);
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = vec![0; 0x380];
        bytes[..4].copy_from_slice(b"MOC3");
        bytes[4] = 4;
        write_word(&mut bytes, 0x40, 0x2c0);
        write_word(&mut bytes, 0x44, 0x340);
        for (index, value) in [
            (2, 1),
            (4, 1),
            (5, 2),
            (14, 3),
            (25, 1),
            (26, 2),
            (27, 1),
            (28, 1),
            (29, 1),
            (30, 1),
            (31, 2),
        ] {
            count(&mut bytes, index, value);
        }
        for (slot, values) in [
            (51, vec![1.0f32.to_bits() as i32; 2]),
            (52, vec![(-1.0f32).to_bits() as i32; 2]),
            (53, vec![0, 0]),
            (56, vec![0, 0]),
            (
                77,
                vec![(-1.0f32).to_bits() as i32, 0, 1.0f32.to_bits() as i32],
            ),
            (115, vec![0, -1]),
            (116, vec![1, 0]),
            (117, vec![0]),
            (118, vec![3]),
            (119, vec![1]),
            (120, vec![0, 0]),
            (121, vec![1, 1]),
            (122, vec![3, 3]),
            (123, vec![0, -1]),
            (124, vec![1, 0]),
            (125, vec![0]),
            (126, vec![0]),
            (127, vec![1]),
            (128, vec![0]),
            (129, vec![1]),
            (130, vec![1]),
            (131, vec![0]),
            (132, vec![1]),
            (133, vec![0]),
            (134, vec![2]),
            (135, vec![0, 1.0f32.to_bits() as i32]),
            (136, vec![1.0f32.to_bits() as i32, 0.25f32.to_bits() as i32]),
        ] {
            section(&mut bytes, slot, &values);
        }
        bytes
    }

    #[test]
    fn neutral_key_contributes_no_delta() {
        let shapes = BlendShapes::parse(&fixture()).unwrap();
        assert!(shapes.mesh_slots(0, &[0.0, 0.0]).unwrap().is_empty());
        assert!(shapes.warp_slots(0, &[0.0, 0.0]).unwrap().is_empty());
    }

    #[test]
    fn interpolates_both_sides_of_neutral_and_applies_weight_limits() {
        let shapes = BlendShapes::parse(&fixture()).unwrap();
        for (value, index) in [(-0.5, 1), (0.5, 3)] {
            let mesh = shapes.mesh_slots(0, &[value, 0.5]).unwrap();
            let warp = shapes.warp_slots(0, &[value, 0.5]).unwrap();
            assert_eq!(mesh.len(), 1);
            assert_eq!(mesh[0].global_index, Some(index));
            assert_eq!(mesh[0].weight, 0.5);
            assert_eq!(warp[0].weight, 0.3125);
        }
    }

    #[test]
    fn additive_forms_keep_their_weights_when_combined() {
        let mut shapes = BlendShapes::parse(&fixture()).unwrap();
        shapes.meshes[0] = vec![0, 1];
        let slots = shapes.mesh_slots(0, &[1.0, 0.0]).unwrap();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots.iter().map(|s| s.weight).sum::<f32>(), 2.0);
    }

    #[test]
    fn rejects_invalid_blend_shape_references_and_ranges() {
        for (slot, values) in [
            (115, vec![5, -1]),
            (119, vec![3]),
            (120, vec![1, 0]),
            (122, vec![2, 3]),
            (128, vec![1]),
            (131, vec![1]),
            (133, vec![2]),
            (136, vec![f32::NAN.to_bits() as i32, 0]),
        ] {
            let mut bytes = fixture();
            section(&mut bytes, slot, &values);
            assert!(BlendShapes::parse(&bytes).is_err(), "slot {slot}");
        }
    }

    #[test]
    fn blend_shape_deltas_activate_mesh_opacity_and_move_vertices() {
        let mut bytes = fixture();
        count(&mut bytes, 2, 0);
        count(&mut bytes, 27, 0);
        let bindings = Moc3KeyformBindings::parse(&bytes).unwrap();
        let deformers = Moc3Deformers::parse(&bytes).unwrap();
        let meshes = Moc3ArtMeshes::from_parts(
            vec![Moc3ArtMeshInfo::new(0, 0, 3, 0, 0, 3, 0, 0)],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0, 1, 2],
            vec![],
        )
        .unwrap();
        let keyforms = Moc3ArtMeshKeyforms::from_parts(
            vec![0],
            vec![1],
            vec![3],
            vec![
                Moc3ArtMeshKeyformInfo::new(0.0, 10.0, 0),
                Moc3ArtMeshKeyformInfo::new(-1.0, -2.0, 6),
                Moc3ArtMeshKeyformInfo::new(0.0, 0.0, 12),
                Moc3ArtMeshKeyformInfo::new(1.0, 2.0, 18),
            ],
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -0.2, -0.4, -0.2, -0.4, -0.2, -0.4, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4,
            ],
        )
        .unwrap();
        let neutral = build_moc3_drawable_meshes_with_parameters(
            &meshes,
            &keyforms,
            &deformers,
            &bindings,
            &[0.0, 0.0],
        )
        .unwrap();
        assert_eq!(neutral[0].opacity(), 0.0);
        let active = build_moc3_drawable_meshes_with_parameters(
            &meshes,
            &keyforms,
            &deformers,
            &bindings,
            &[0.5, 0.0],
        )
        .unwrap();
        assert_eq!(active[0].opacity(), 0.5);
        assert_eq!(active[0].draw_order(), 11.0);
        assert_eq!(active[0].vertices()[0].position(), [0.1, -0.2]);
        assert_eq!(active[0].multiply_color(), [1.0, 1.0, 1.0]);
        assert_eq!(active[0].vertices()[0].uv(), neutral[0].vertices()[0].uv());
        let negative = build_moc3_drawable_meshes_with_parameters(
            &meshes,
            &keyforms,
            &deformers,
            &bindings,
            &[-1.0, 0.0],
        )
        .unwrap();
        assert_eq!(negative[0].opacity(), 0.0);
    }

    #[test]
    fn warp_deltas_affect_geometry_and_inherited_opacity() {
        let mut bytes = fixture();
        for (index, value) in [(1, 1), (7, 4), (10, 32)] {
            count(&mut bytes, index, value);
        }
        for (slot, values) in [
            (16, vec![-1]),
            (17, vec![0]),
            (18, vec![0]),
            (19, vec![-1]),
            (20, vec![0]),
            (21, vec![1]),
            (22, vec![4]),
            (23, vec![1]),
            (24, vec![1]),
            (
                59,
                vec![1.0f32.to_bits() as i32, 0, 0, (-1.0f32).to_bits() as i32],
            ),
            (60, vec![0, 8, 16, 24]),
        ] {
            section(&mut bytes, slot, &values);
        }
        let positions = [
            0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4,
        ]
        .map(|v| v.to_bits() as i32);
        section(&mut bytes, 71, &positions);
        let bindings = Moc3KeyformBindings::parse(&bytes).unwrap();
        let deformers = Moc3Deformers::parse(&bytes).unwrap();
        let composed = deformers.compose(&bindings, &[0.5, 0.0]).unwrap();
        assert_eq!(composed.deformer_opacity(0), 0.5);
        let mut vertices = vec![crate::core::Vector2::new(0.25, 0.25)];
        composed.transform_vertices(0, &mut vertices).unwrap();
        assert!((vertices[0].x() - 0.35).abs() < 0.00001);
        assert!((vertices[0].y() - 0.45).abs() < 0.00001);
    }
}
