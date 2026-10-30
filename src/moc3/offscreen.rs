use crate::Result;

use super::{
    Moc3CountInfo, Moc3Header, Moc3SectionOffsets, Moc3Version,
    parse::{invalid_moc3, read_i32_section, to_usize},
};

const PART_PARENT_PART_INDICES_SLOT: usize = 9;
const DRAWABLE_PARENT_PART_INDICES_SLOT: usize = 39;
const PART_OFFSCREEN_INDICES_SLOT: usize = 152;
const OFFSCREEN_OWNER_PART_INDICES_SLOT: usize = 155;

#[derive(Debug, Clone, PartialEq)]
pub struct Moc3OffscreenInfo {
    pub(crate) composition: super::compositing::CompositionInfo,
    part_parent_indices: Vec<i32>,
    drawable_parent_part_indices: Vec<i32>,
    part_offscreen_indices: Vec<i32>,
    offscreen_owner_part_indices: Vec<i32>,
}

impl Moc3OffscreenInfo {
    #[cfg(test)]
    pub(crate) fn from_parts(
        part_parent_indices: Vec<i32>,
        drawable_parent_part_indices: Vec<i32>,
        part_offscreen_indices: Vec<i32>,
        offscreen_owner_part_indices: Vec<i32>,
    ) -> Self {
        Self {
            composition: Default::default(),
            part_parent_indices,
            drawable_parent_part_indices,
            part_offscreen_indices,
            offscreen_owner_part_indices,
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let header = Moc3Header::parse(bytes)?;
        let offsets = Moc3SectionOffsets::parse(bytes)?;
        let counts = Moc3CountInfo::parse(bytes)?;
        let endianness = header.endianness();
        let part_count = to_usize(counts.parts(), "part count")?;
        let drawable_count = to_usize(counts.art_meshes(), "art mesh count")?;

        let part_parent_indices = read_i32_section(
            bytes,
            &offsets,
            PART_PARENT_PART_INDICES_SLOT,
            part_count,
            endianness,
        )?;
        let drawable_parent_part_indices = read_i32_section(
            bytes,
            &offsets,
            DRAWABLE_PARENT_PART_INDICES_SLOT,
            drawable_count,
            endianness,
        )?;
        let part_offscreen_indices = if header.version() == Moc3Version::V5_3_0 {
            read_i32_section(
                bytes,
                &offsets,
                PART_OFFSCREEN_INDICES_SLOT,
                part_count,
                endianness,
            )?
        } else {
            vec![-1; part_count]
        };
        let offscreen_count = to_usize(counts.offscreens(), "offscreen count")?;
        if part_offscreen_indices
            .iter()
            .any(|&index| index < -1 || index >= offscreen_count as i32)
        {
            return Err(invalid_moc3("invalid part offscreen reference"));
        }
        let offscreen_owner_part_indices = if offscreen_count == 0 {
            Vec::new()
        } else {
            read_i32_section(
                bytes,
                &offsets,
                OFFSCREEN_OWNER_PART_INDICES_SLOT,
                offscreen_count,
                endianness,
            )?
        };
        for (offscreen, &part) in offscreen_owner_part_indices.iter().enumerate() {
            if usize::try_from(part)
                .ok()
                .and_then(|part| part_offscreen_indices.get(part))
                .copied()
                != Some(offscreen as i32)
            {
                return Err(invalid_moc3("offscreen owner does not reference its group"));
            }
        }

        Ok(Self {
            composition: super::compositing::CompositionInfo::parse(bytes)?,
            part_parent_indices,
            drawable_parent_part_indices,
            part_offscreen_indices,
            offscreen_owner_part_indices,
        })
    }

    pub fn offscreen_count(&self) -> usize {
        self.offscreen_owner_part_indices.len()
    }

    pub fn part_offscreen_index(&self, part_index: usize) -> Option<i32> {
        self.part_offscreen_indices.get(part_index).copied()
    }

    pub fn part_offscreen_indices(&self) -> &[i32] {
        &self.part_offscreen_indices
    }

    pub fn offscreen_owner_part_index(&self, offscreen_index: usize) -> Option<i32> {
        self.offscreen_owner_part_indices
            .get(offscreen_index)
            .copied()
    }

    pub fn drawable_parent_part_index(&self, drawable_index: usize) -> Option<i32> {
        self.drawable_parent_part_indices
            .get(drawable_index)
            .copied()
    }
}
