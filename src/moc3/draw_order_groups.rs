use crate::Result;

use super::{
    Moc3CountInfo, Moc3Header, Moc3SectionOffsets,
    parse::{read_i32_section, to_usize},
};

const GROUP_OBJECT_BEGIN_INDICES_SLOT: usize = 81;
const GROUP_OBJECT_COUNTS_SLOT: usize = 82;
const GROUP_SUBTREE_COUNTS_SLOT: usize = 83;
const GROUP_MAX_DRAW_ORDERS_SLOT: usize = 84;
const GROUP_BASE_DRAW_ORDERS_SLOT: usize = 85;
const OBJECT_TYPES_SLOT: usize = 86;
const OBJECT_INDICES_SLOT: usize = 87;
const OBJECT_SELF_GROUP_IDX_SLOT: usize = 88;

const OBJECT_TYPE_DRAWABLE: i32 = 0;
const OBJECT_TYPE_PART: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GroupObject {
    pub object_type: i32,
    pub object_idx: usize,
    pub self_group_idx: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Group {
    pub object_begin: usize,
    pub object_count: usize,
    pub subtree_drawable_count: usize,
    pub base_draw_order: i32,
    pub max_draw_order: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Moc3DrawOrderGroups {
    pub(super) groups: Vec<Group>,
    pub(super) objects: Vec<GroupObject>,
    pub(super) drawable_count: usize,
}

impl Moc3DrawOrderGroups {
    pub(crate) fn render_nodes(
        &self,
        drawable_orders: &[i32],
        part_orders: &[i32],
        part_enable: &[bool],
        part_offscreens: &[i32],
        offscreens: &mut [super::Moc3OffscreenState],
    ) -> Option<Vec<super::Moc3RenderNode>> {
        self.collect_nodes(
            0,
            0,
            drawable_orders,
            part_orders,
            part_enable,
            part_offscreens,
            offscreens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_nodes(
        &self,
        index: usize,
        depth: usize,
        drawable_orders: &[i32],
        part_orders: &[i32],
        part_enable: &[bool],
        part_offscreens: &[i32],
        offscreens: &mut [super::Moc3OffscreenState],
    ) -> Option<Vec<super::Moc3RenderNode>> {
        use super::Moc3RenderNode;
        if depth > self.groups.len() {
            return None;
        }
        let mut nodes = Vec::new();
        for object in self.ordered_members(index, drawable_orders, part_orders, part_enable)? {
            if object.object_type == OBJECT_TYPE_DRAWABLE {
                nodes.push(Moc3RenderNode::Drawable(object.object_idx));
            } else if object.object_type == OBJECT_TYPE_PART {
                if object.self_group_idx == index {
                    return None;
                }
                let child = self.collect_nodes(
                    object.self_group_idx,
                    depth + 1,
                    drawable_orders,
                    part_orders,
                    part_enable,
                    part_offscreens,
                    offscreens,
                )?;
                if let Some(id) = part_offscreens
                    .get(object.object_idx)
                    .and_then(|&v| usize::try_from(v).ok())
                {
                    offscreens.get_mut(id)?.nodes = child;
                    nodes.push(Moc3RenderNode::Offscreen(id));
                } else {
                    nodes.extend(child);
                }
            }
        }
        Some(nodes)
    }
    pub fn parse(bytes: &[u8]) -> Result<Option<Self>> {
        let header = Moc3Header::parse(bytes)?;
        let offsets = Moc3SectionOffsets::parse(bytes)?;
        let counts = Moc3CountInfo::parse(bytes)?;
        let endianness = header.endianness();

        let group_count = to_usize(counts.draw_order_groups(), "draw order group count")?;
        let object_count = to_usize(
            counts.draw_order_group_objects(),
            "draw order group object count",
        )?;
        let drawable_count = to_usize(counts.art_meshes(), "art mesh count")?;

        if group_count == 0 || object_count == 0 {
            return Ok(None);
        }

        let begin = read_i32_section(
            bytes,
            &offsets,
            GROUP_OBJECT_BEGIN_INDICES_SLOT,
            group_count,
            endianness,
        )?;
        let count = read_i32_section(
            bytes,
            &offsets,
            GROUP_OBJECT_COUNTS_SLOT,
            group_count,
            endianness,
        )?;
        let subtree = read_i32_section(
            bytes,
            &offsets,
            GROUP_SUBTREE_COUNTS_SLOT,
            group_count,
            endianness,
        )?;
        let max = read_i32_section(
            bytes,
            &offsets,
            GROUP_MAX_DRAW_ORDERS_SLOT,
            group_count,
            endianness,
        )?;
        let base = read_i32_section(
            bytes,
            &offsets,
            GROUP_BASE_DRAW_ORDERS_SLOT,
            group_count,
            endianness,
        )?;
        let types = read_i32_section(bytes, &offsets, OBJECT_TYPES_SLOT, object_count, endianness)?;
        let indices = read_i32_section(
            bytes,
            &offsets,
            OBJECT_INDICES_SLOT,
            object_count,
            endianness,
        )?;
        let self_group_idx = read_i32_section(
            bytes,
            &offsets,
            OBJECT_SELF_GROUP_IDX_SLOT,
            object_count,
            endianness,
        )?;

        let mut groups = Vec::with_capacity(group_count);
        for group_index in 0..group_count {
            groups.push(Group {
                object_begin: to_usize(begin[group_index].max(0) as u32, "group object begin")?,
                object_count: to_usize(count[group_index].max(0) as u32, "group object count")?,
                subtree_drawable_count: to_usize(
                    subtree[group_index].max(0) as u32,
                    "group subtree count",
                )?,
                base_draw_order: base[group_index],
                max_draw_order: max[group_index],
            });
        }

        let mut objects = Vec::with_capacity(object_count);
        for object_index in 0..object_count {
            objects.push(GroupObject {
                object_type: types[object_index],
                object_idx: indices[object_index].max(0) as usize,
                self_group_idx: self_group_idx[object_index].max(0) as usize,
            });
        }

        Ok(Some(Self {
            groups,
            objects,
            drawable_count,
        }))
    }

    pub fn drawable_count(&self) -> usize {
        self.drawable_count
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn render_orders(
        &self,
        drawable_draw_orders: &[i32],
        part_draw_orders: &[i32],
        part_enable: &[bool],
        part_offscreen_indices: &[i32],
        offscreen_count: usize,
    ) -> Option<Vec<i32>> {
        let mut render_orders = Vec::with_capacity(self.drawable_count + offscreen_count);
        self.render_orders_into(
            drawable_draw_orders,
            part_draw_orders,
            part_enable,
            part_offscreen_indices,
            offscreen_count,
            &mut render_orders,
        )?;
        Some(render_orders)
    }

    pub(crate) fn render_orders_into(
        &self,
        drawable_draw_orders: &[i32],
        part_draw_orders: &[i32],
        part_enable: &[bool],
        part_offscreen_indices: &[i32],
        offscreen_count: usize,
        render_orders: &mut Vec<i32>,
    ) -> Option<()> {
        if drawable_draw_orders.len() != self.drawable_count {
            return None;
        }

        render_orders.clear();
        render_orders.resize(self.drawable_count + offscreen_count, 0);
        self.expand_group(
            0,
            0,
            drawable_draw_orders,
            part_draw_orders,
            part_enable,
            part_offscreen_indices,
            render_orders,
        )
    }

    fn ordered_members(
        &self,
        group_index: usize,
        drawable_orders: &[i32],
        part_orders: &[i32],
        part_enable: &[bool],
    ) -> Option<impl Iterator<Item = &GroupObject>> {
        let group = self.groups.get(group_index)?;
        let members = self
            .objects
            .get(group.object_begin..group.object_begin.checked_add(group.object_count)?)?;
        let min = group.base_draw_order;
        let max = group.max_draw_order.max(min);
        // Order values can be sparse. Store one entry per member and use its
        // original offset to preserve the model's ordering for ties.
        let mut ordered = Vec::with_capacity(members.len());
        for (offset, object) in members.iter().enumerate() {
            let order = match object.object_type {
                OBJECT_TYPE_DRAWABLE => *drawable_orders.get(object.object_idx)?,
                OBJECT_TYPE_PART
                    if part_enable.get(object.object_idx).copied().unwrap_or(false) =>
                {
                    *part_orders.get(object.object_idx)?
                }
                _ => min,
            };
            ordered.push((order.clamp(min, max), offset));
        }
        ordered.sort_unstable();
        Some(ordered.into_iter().map(move |(_, offset)| &members[offset]))
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_group(
        &self,
        group_index: usize,
        start_rank: i32,
        drawable_draw_orders: &[i32],
        part_draw_orders: &[i32],
        part_enable: &[bool],
        part_offscreen_indices: &[i32],
        render_orders: &mut [i32],
    ) -> Option<()> {
        let mut rank = start_rank;
        for object in self.ordered_members(
            group_index,
            drawable_draw_orders,
            part_draw_orders,
            part_enable,
        )? {
            if object.object_type == OBJECT_TYPE_PART {
                if let Some(offscreen) = part_offscreen_indices
                    .get(object.object_idx)
                    .copied()
                    .filter(|&value| value >= 0)
                {
                    let slot = self.drawable_count.checked_add(offscreen as usize)?;
                    *render_orders.get_mut(slot)? = rank;
                    rank += 1;
                }
                let child = object.self_group_idx;
                if child < self.groups.len() && child != group_index {
                    self.expand_group(
                        child,
                        rank,
                        drawable_draw_orders,
                        part_draw_orders,
                        part_enable,
                        part_offscreen_indices,
                        render_orders,
                    )?;
                    rank += i32::try_from(self.groups.get(child)?.subtree_drawable_count).ok()?;
                }
            } else {
                *render_orders.get_mut(object.object_idx)? = rank;
                rank += 1;
            }
        }

        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_orders_keep_member_order_when_clamped_values_tie() {
        let groups = Moc3DrawOrderGroups {
            groups: vec![Group {
                object_begin: 0,
                object_count: 4,
                subtree_drawable_count: 4,
                base_draw_order: -1_000_000,
                max_draw_order: 1_000_000,
            }],
            objects: [3, 2, 1, 0]
                .map(|object_idx| GroupObject {
                    object_type: OBJECT_TYPE_DRAWABLE,
                    object_idx,
                    self_group_idx: 0,
                })
                .to_vec(),
            drawable_count: 4,
        };
        let orders = [i32::MIN, 9, -1_000_000, 9];
        assert_eq!(
            groups.render_orders(&orders, &[], &[], &[], 0).unwrap(),
            [1, 3, 0, 2]
        );
        assert_eq!(
            groups
                .render_nodes(&orders, &[], &[], &[], &mut [])
                .unwrap(),
            [2, 0, 3, 1].map(crate::moc3::Moc3RenderNode::Drawable)
        );
    }

    #[test]
    fn empty_group_with_inverted_sentinel_draw_order_range_does_not_panic() {
        let groups = Moc3DrawOrderGroups {
            groups: vec![
                Group {
                    object_begin: 0,
                    object_count: 1,
                    subtree_drawable_count: 0,
                    base_draw_order: 0,
                    max_draw_order: 0,
                },
                Group {
                    object_begin: 1,
                    object_count: 0,
                    subtree_drawable_count: 0,
                    base_draw_order: i32::MAX,
                    max_draw_order: -i32::MAX,
                },
            ],
            objects: vec![GroupObject {
                object_type: OBJECT_TYPE_PART,
                object_idx: 0,
                self_group_idx: 1,
            }],
            drawable_count: 0,
        };

        assert_eq!(
            groups.render_orders(&[], &[0], &[false], &[-1], 0),
            Some(Vec::new())
        );
    }
}
