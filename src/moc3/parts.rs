use crate::Result;

use super::{
    Moc3CountInfo, Moc3Header, Moc3KeyformBindings, Moc3SectionOffsets,
    parse::{read_f32_section, read_i32_section, to_usize},
};

const PART_KEYFORM_BINDING_BAND_INDICES_SLOT: usize = 4;
const PART_KEYFORM_BEGIN_INDICES_SLOT: usize = 5;
const PART_KEYFORM_COUNTS_SLOT: usize = 6;
const PART_PARENT_PART_INDICES_SLOT: usize = 9;
const PART_KEYFORM_DRAW_ORDERS_SLOT: usize = 58;
const PART_KEYFORM_OPACITIES_SLOT: usize = 59;

#[derive(Debug, Clone, PartialEq)]
pub struct Moc3Parts {
    visible: Vec<bool>,
    parent_part_indices: Vec<i32>,
    keyform_binding_band_indices: Vec<i32>,
    keyform_begin_indices: Vec<i32>,
    keyform_counts: Vec<i32>,
    keyform_draw_orders: Vec<f32>,
    keyform_opacities: Vec<f32>,
}

impl Moc3Parts {
    #[cfg(test)]
    pub(crate) fn from_parts(
        parent_part_indices: Vec<i32>,
        keyform_binding_band_indices: Vec<i32>,
        keyform_begin_indices: Vec<i32>,
        keyform_counts: Vec<i32>,
        keyform_draw_orders: Vec<f32>,
        keyform_opacities: Vec<f32>,
    ) -> Self {
        Self {
            visible: vec![true; parent_part_indices.len()],
            parent_part_indices,
            keyform_binding_band_indices,
            keyform_begin_indices,
            keyform_counts,
            keyform_draw_orders,
            keyform_opacities,
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let header = Moc3Header::parse(bytes)?;
        let offsets = Moc3SectionOffsets::parse(bytes)?;
        let counts = Moc3CountInfo::parse(bytes)?;
        let endianness = header.endianness();
        let part_count = to_usize(counts.parts(), "part count")?;
        let part_keyform_count = to_usize(counts.part_keyforms(), "part keyform count")?;

        Ok(Self {
            visible: super::parse::read_i32_section_or_default(
                bytes, &offsets, 7, part_count, endianness, 1,
            )?
            .into_iter()
            .map(|f| f != 0)
            .collect(),
            parent_part_indices: read_i32_section(
                bytes,
                &offsets,
                PART_PARENT_PART_INDICES_SLOT,
                part_count,
                endianness,
            )?,
            keyform_binding_band_indices: read_i32_section(
                bytes,
                &offsets,
                PART_KEYFORM_BINDING_BAND_INDICES_SLOT,
                part_count,
                endianness,
            )?,
            keyform_begin_indices: read_i32_section(
                bytes,
                &offsets,
                PART_KEYFORM_BEGIN_INDICES_SLOT,
                part_count,
                endianness,
            )?,
            keyform_counts: read_i32_section(
                bytes,
                &offsets,
                PART_KEYFORM_COUNTS_SLOT,
                part_count,
                endianness,
            )?,
            keyform_draw_orders: read_f32_section(
                bytes,
                &offsets,
                PART_KEYFORM_DRAW_ORDERS_SLOT,
                part_keyform_count,
                endianness,
            )?,
            keyform_opacities: read_f32_section(
                bytes,
                &offsets,
                PART_KEYFORM_OPACITIES_SLOT,
                part_keyform_count,
                endianness,
            )?,
        })
    }

    pub fn part_count(&self) -> usize {
        self.parent_part_indices.len()
    }

    pub(super) fn is_visible(&self, index: usize) -> bool {
        let mut current = Some(index);
        for _ in 0..self.part_count() + 1 {
            let Some(i) = current else {
                return true;
            };
            if !self.visible.get(i).copied().unwrap_or(false) {
                return false;
            }
            current = self
                .parent_part_index(i)
                .and_then(|p| usize::try_from(p).ok());
        }
        false
    }

    pub(crate) fn slots(
        &self,
        index: usize,
        bindings: &Moc3KeyformBindings,
        values: &[f32],
    ) -> Option<(usize, Vec<super::keyform_bindings::Moc3KeyformSlot>, bool)> {
        let band = *self.keyform_binding_band_indices.get(index)?;
        let count = usize::try_from(*self.keyform_counts.get(index)?).ok()?;
        let begin = usize::try_from(*self.keyform_begin_indices.get(index)?).ok()?;
        let mut slots = bindings.keyform_slots(band, count, values)?;
        slots.extend(bindings.blend_shapes.part_slots(index, values)?);
        Some((
            begin,
            slots,
            self.is_visible(index) && bindings.band_is_in_range(band, values)?,
        ))
    }

    pub fn parent_part_index(&self, part_index: usize) -> Option<i32> {
        self.parent_part_indices.get(part_index).copied()
    }

    pub fn interpolate_opacity(
        &self,
        part_index: usize,
        bindings: &Moc3KeyformBindings,
        parameter_values: &[f32],
    ) -> Option<f32> {
        let keyform_count = usize::try_from(*self.keyform_counts.get(part_index)?).ok()?;
        if keyform_count == 0 {
            return Some(1.0);
        }
        let begin = usize::try_from(*self.keyform_begin_indices.get(part_index)?).ok()?;
        let slots = bindings.keyform_slots(
            *self.keyform_binding_band_indices.get(part_index)?,
            keyform_count,
            parameter_values,
        )?;

        let mut opacity = 0.0f32;
        for slot in slots {
            let keyform_index = begin.checked_add(slot.local_index)?;
            opacity += *self.keyform_opacities.get(keyform_index)? * slot.weight;
        }
        Some(opacity)
    }

    pub fn interpolate_draw_order(
        &self,
        part_index: usize,
        bindings: &Moc3KeyformBindings,
        parameter_values: &[f32],
    ) -> Option<f32> {
        let keyform_count = usize::try_from(*self.keyform_counts.get(part_index)?).ok()?;
        if keyform_count == 0 {
            return None;
        }
        let begin = usize::try_from(*self.keyform_begin_indices.get(part_index)?).ok()?;
        let mut slots = bindings.keyform_slots(
            *self.keyform_binding_band_indices.get(part_index)?,
            keyform_count,
            parameter_values,
        )?;

        let mut draw_order = 0.0f32;
        slots.extend(
            bindings
                .blend_shapes
                .part_slots(part_index, parameter_values)?,
        );
        for slot in slots {
            let keyform_index = slot.index(begin)?;
            draw_order += *self.keyform_draw_orders.get(keyform_index)? * slot.weight;
        }
        Some(draw_order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hiyori_parts_with_default_opacity() {
        let bytes = std::fs::read("assets/models/Hiyori/Hiyori.moc3").unwrap();
        let parts = Moc3Parts::parse(&bytes).unwrap();
        let bindings = Moc3KeyformBindings::parse(&bytes).unwrap();
        let parameters = bindings.parameter_default_values();

        assert_eq!(parts.part_count(), 24);
        for part_index in 0..parts.part_count() {
            let opacity = parts
                .interpolate_opacity(part_index, &bindings, parameters)
                .unwrap();
            assert!((opacity - 1.0).abs() < 1e-6);
        }
    }
}
