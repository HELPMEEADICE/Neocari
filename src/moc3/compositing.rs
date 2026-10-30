use super::{
    Moc3CountInfo, Moc3DrawableBlendMode, Moc3Header, Moc3SectionOffsets, Moc3Version,
    keyform_bindings::Moc3KeyformSlot,
    parse::{invalid_moc3, read_f32_section, read_i32_section, read_u8_section, to_usize},
};
use crate::Result;

/// Cubism 5.3 color composition, independent of alpha composition.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Moc3ColorBlendMode {
    #[default]
    Normal = 0,
    PremultipliedAdd = 1,
    PremultipliedMultiply = 2,
    Add = 3,
    AddGlow = 4,
    Darken = 5,
    Multiply = 6,
    ColorBurn = 7,
    LinearBurn = 8,
    Lighten = 9,
    Screen = 10,
    ColorDodge = 11,
    Overlay = 12,
    SoftLight = 13,
    HardLight = 14,
    LinearLight = 15,
    Hue = 16,
    Color = 17,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Moc3AlphaBlendMode {
    #[default]
    Over = 0,
    Atop = 1,
    Out = 2,
    Conjoint = 3,
    Disjoint = 4,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Moc3BlendConfig {
    pub color: Moc3ColorBlendMode,
    pub alpha: Moc3AlphaBlendMode,
}

impl Moc3BlendConfig {
    pub fn from_raw(raw: u32) -> Result<Self> {
        use Moc3ColorBlendMode::*;
        let color = match raw & 255 {
            0 => Normal,
            1 => PremultipliedAdd,
            2 => PremultipliedMultiply,
            3 => Add,
            4 => AddGlow,
            5 => Darken,
            6 => Multiply,
            7 => ColorBurn,
            8 => LinearBurn,
            9 => Lighten,
            10 => Screen,
            11 => ColorDodge,
            12 => Overlay,
            13 => SoftLight,
            14 => HardLight,
            15 => LinearLight,
            16 => Hue,
            17 => Color,
            _ => return Err(invalid_moc3("invalid color blend mode")),
        };
        let alpha = match (raw >> 8) & 255 {
            0 => Moc3AlphaBlendMode::Over,
            1 => Moc3AlphaBlendMode::Atop,
            2 => Moc3AlphaBlendMode::Out,
            3 => Moc3AlphaBlendMode::Conjoint,
            4 => Moc3AlphaBlendMode::Disjoint,
            _ => return Err(invalid_moc3("invalid alpha blend mode")),
        };
        if raw >> 16 != 0 {
            return Err(invalid_moc3("invalid blend configuration bits"));
        }
        Ok(Self {
            color,
            alpha: if matches!(color, PremultipliedAdd | PremultipliedMultiply) {
                Moc3AlphaBlendMode::Over
            } else {
                alpha
            },
        })
    }
    pub fn is_advanced(self) -> bool {
        self.alpha != Moc3AlphaBlendMode::Over
            || !matches!(
                self.color,
                Moc3ColorBlendMode::Normal
                    | Moc3ColorBlendMode::PremultipliedAdd
                    | Moc3ColorBlendMode::PremultipliedMultiply
            )
    }
    pub(crate) fn from_legacy(mode: Moc3DrawableBlendMode) -> Self {
        Self {
            color: match mode {
                Moc3DrawableBlendMode::Normal => Moc3ColorBlendMode::Normal,
                Moc3DrawableBlendMode::Additive => Moc3ColorBlendMode::PremultipliedAdd,
                Moc3DrawableBlendMode::Multiplicative => Moc3ColorBlendMode::PremultipliedMultiply,
                Moc3DrawableBlendMode::Screen => Moc3ColorBlendMode::Screen,
            },
            alpha: Moc3AlphaBlendMode::Over,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moc3RenderNode {
    Drawable(usize),
    Offscreen(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Moc3OffscreenState {
    pub part_index: usize,
    pub enabled: bool,
    pub opacity: f32,
    pub multiply_color: [f32; 3],
    pub screen_color: [f32; 3],
    pub blend: Moc3BlendConfig,
    pub masks: Vec<i32>,
    pub inverted_mask: bool,
    pub nodes: Vec<Moc3RenderNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Moc3RenderPlan {
    pub nodes: Vec<Moc3RenderNode>,
    pub offscreens: Vec<Moc3OffscreenState>,
    pub drawable_blends: Vec<Moc3BlendConfig>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct CompositionInfo {
    pub drawable_blends: Vec<Moc3BlendConfig>,
    pub blends: Vec<Moc3BlendConfig>,
    pub masks: Vec<Vec<i32>>,
    pub inverted: Vec<bool>,
    part_keyform_indices: Vec<i32>,
    opacities: Vec<f32>,
    multiply: Vec<[f32; 3]>,
    screen: Vec<[f32; 3]>,
}

impl CompositionInfo {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let header = Moc3Header::parse(bytes)?;
        if header.version() != Moc3Version::V5_3_0 {
            return Ok(Self::default());
        }
        let counts = Moc3CountInfo::parse(bytes)?;
        let offsets = Moc3SectionOffsets::parse(bytes)?;
        let endian = header.endianness();
        let integers = |slot, count| read_i32_section(bytes, &offsets, slot, count, endian);
        let floats = |slot, count| read_f32_section(bytes, &offsets, slot, count, endian);
        let n = to_usize(counts.offscreens(), "offscreen count")?;
        let drawable_blends = integers(153, counts.art_meshes() as usize)?
            .into_iter()
            .map(|raw| Moc3BlendConfig::from_raw(raw as u32))
            .collect::<Result<Vec<_>>>()?;
        let blends = integers(157, n)?
            .into_iter()
            .map(|raw| Moc3BlendConfig::from_raw(raw as u32))
            .collect::<Result<Vec<_>>>()?;
        let flags = read_u8_section(bytes, &offsets, 156, n)?;
        let begins = integers(158, n)?;
        let lengths = integers(159, n)?;
        let refs = integers(80, counts.drawable_masks() as usize)?;
        let masks = (0..n)
            .map(|i| {
                let len = usize::try_from(lengths[i])
                    .map_err(|_| invalid_moc3("negative offscreen mask count"))?;
                if len == 0 {
                    return Ok(Vec::new());
                }
                let begin = usize::try_from(begins[i])
                    .map_err(|_| invalid_moc3("negative offscreen mask begin"))?;
                let end = begin
                    .checked_add(len)
                    .ok_or_else(|| invalid_moc3("offscreen mask range overflow"))?;
                let refs = refs
                    .get(begin..end)
                    .ok_or_else(|| invalid_moc3("offscreen masks outside reference section"))?;
                if refs.iter().any(|&v| v >= counts.art_meshes() as i32) {
                    return Err(invalid_moc3("invalid offscreen mask drawable"));
                }
                Ok(refs.iter().copied().filter(|&v| v >= 0).collect())
            })
            .collect::<Result<Vec<_>>>()?;
        let k = counts.offscreen_keyforms() as usize;
        let part_keyform_indices = integers(160, counts.part_keyforms() as usize)?;
        let opacities = floats(161, k)?;
        let multiply_indices = integers(162, k)?;
        let screen_indices = integers(163, k)?;
        let channel_colors = |first, count, indices: &[i32]| -> Result<Vec<[f32; 3]>> {
            let channels = [
                floats(first, count)?,
                floats(first + 1, count)?,
                floats(first + 2, count)?,
            ];
            indices
                .iter()
                .map(|&index| {
                    let i = usize::try_from(index)
                        .map_err(|_| invalid_moc3("negative offscreen color index"))?;
                    Ok([
                        *channels[0]
                            .get(i)
                            .ok_or_else(|| invalid_moc3("offscreen color outside section"))?,
                        *channels[1]
                            .get(i)
                            .ok_or_else(|| invalid_moc3("offscreen color outside section"))?,
                        *channels[2]
                            .get(i)
                            .ok_or_else(|| invalid_moc3("offscreen color outside section"))?,
                    ])
                })
                .collect()
        };
        let multiply = channel_colors(
            108,
            counts.keyform_multiply_colors() as usize,
            &multiply_indices,
        )?;
        let screen = channel_colors(
            111,
            counts.keyform_screen_colors() as usize,
            &screen_indices,
        )?;
        if part_keyform_indices.iter().any(|&v| v >= k as i32) {
            return Err(invalid_moc3("invalid part offscreen keyform index"));
        }
        Ok(Self {
            drawable_blends,
            blends,
            masks,
            inverted: flags.into_iter().map(|f| f & 8 != 0).collect(),
            part_keyform_indices,
            opacities,
            multiply,
            screen,
        })
    }

    pub fn state(
        &self,
        index: usize,
        part_index: usize,
        enabled: bool,
        begin: usize,
        slots: &[Moc3KeyformSlot],
    ) -> Option<Moc3OffscreenState> {
        let mut opacity = 0.0;
        let mut multiply = [0.0; 3];
        let mut screen = [0.0; 3];
        for slot in slots {
            let keyform =
                usize::try_from(*self.part_keyform_indices.get(slot.index(begin)?)?).ok()?;
            opacity += *self.opacities.get(keyform)? * slot.weight;
            for c in 0..3 {
                multiply[c] += self.multiply.get(keyform)?[c] * slot.weight;
                screen[c] += self.screen.get(keyform)?[c] * slot.weight;
            }
        }
        Some(Moc3OffscreenState {
            part_index,
            enabled,
            opacity: opacity.clamp(0.0, 1.0),
            multiply_color: multiply.map(|v| v.clamp(0.0, 1.0)),
            screen_color: screen.map(|v| v.clamp(0.0, 1.0)),
            blend: *self.blends.get(index)?,
            masks: self.masks.get(index)?.clone(),
            inverted_mask: *self.inverted.get(index)?,
            nodes: Vec::new(),
        })
    }
}
