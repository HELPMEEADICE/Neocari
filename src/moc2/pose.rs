//! Cubism 2 pose groups and part opacity fading.

use super::{json::Moc2Pose, runtime::Moc2Runtime};

#[derive(Debug, Clone)]
struct PoseMember {
    part: Option<usize>,
    parameter: Option<usize>,
    links: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PoseGroup {
    members: Vec<PoseMember>,
}

const CLEAR_TIME_SECONDS: f32 = 0.5;
const PHI: f32 = 0.5;
const MAX_BACK_OPACITY: f32 = 0.15;

/// Runtime state for `pose.json` part fading.
#[derive(Debug, Clone, Default)]
pub struct Moc2PoseRuntime {
    groups: Vec<PoseGroup>,
}

impl Moc2PoseRuntime {
    pub(crate) fn new(pose: &Moc2Pose, runtime: &mut Moc2Runtime) -> Self {
        let groups = pose
            .groups()
            .iter()
            .map(|group| PoseGroup {
                members: group
                    .iter()
                    .map(|part| PoseMember {
                        part: runtime.part_index(part.id()),
                        parameter: Some(
                            runtime.ensure_parameter_index(&format!("VISIBLE:{}", part.id())),
                        ),
                        links: part
                            .links()
                            .iter()
                            .filter_map(|link| runtime.part_index(link))
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        let pose_runtime = Self { groups };
        pose_runtime.initialize(runtime);
        pose_runtime
    }

    /// Returns the number of pose groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Applies the initial `VISIBLE:` state of every pose part.
    pub fn initialize(&self, runtime: &mut Moc2Runtime) {
        for group in &self.groups {
            for member in &group.members {
                let (Some(part), Some(parameter)) = (member.part, member.parameter) else {
                    continue;
                };
                let visible = runtime.parameter_value_at(parameter).unwrap_or(0.0) != 0.0;
                runtime.set_part_opacity_by_index(part, if visible { 1.0 } else { 0.0 });
                runtime.write_parameter(parameter, if visible { 1.0 } else { 0.0 });
            }
        }
    }

    /// Advances part fading by `delta_seconds`.
    pub fn update(&self, runtime: &mut Moc2Runtime, delta_seconds: f32) {
        let delta_seconds = delta_seconds.max(0.0);
        for group in &self.groups {
            self.normalize_group(runtime, group, delta_seconds);
            self.copy_link_opacities(runtime, group);
        }
    }

    fn normalize_group(&self, runtime: &mut Moc2Runtime, group: &PoseGroup, delta_seconds: f32) {
        let mut visible_member = None;
        let mut visible_opacity = 1.0;

        for (index, member) in group.members.iter().enumerate() {
            let (Some(part), Some(parameter)) = (member.part, member.parameter) else {
                continue;
            };
            if runtime.parameter_value_at(parameter).unwrap_or(0.0) == 0.0 {
                continue;
            }
            if visible_member.is_some() {
                break;
            }
            visible_member = Some(index);
            visible_opacity =
                (runtime.part_opacity_at(part) + delta_seconds / CLEAR_TIME_SECONDS).min(1.0);
        }

        let visible_member = visible_member.unwrap_or(0);
        if visible_member >= group.members.len() {
            return;
        }
        if let Some(part) = group.members[visible_member].part {
            runtime.set_part_opacity_by_index(part, visible_opacity);
        }

        for (index, member) in group.members.iter().enumerate() {
            let Some(part) = member.part else {
                continue;
            };
            if index == visible_member {
                continue;
            }
            let mut target_opacity = if visible_opacity < PHI {
                visible_opacity * (PHI - 1.0) / PHI + 1.0
            } else {
                (1.0 - visible_opacity) * PHI / (1.0 - PHI)
            };
            let back_opacity = (1.0 - target_opacity) * (1.0 - visible_opacity);
            if back_opacity > MAX_BACK_OPACITY {
                target_opacity = 1.0 - MAX_BACK_OPACITY / (1.0 - visible_opacity);
            }
            let opacity = runtime.part_opacity_at(part).min(target_opacity);
            runtime.set_part_opacity_by_index(part, opacity);
        }
    }

    fn copy_link_opacities(&self, runtime: &mut Moc2Runtime, group: &PoseGroup) {
        for member in &group.members {
            let Some(part) = member.part else {
                continue;
            };
            let opacity = runtime.part_opacity_at(part);
            for link in &member.links {
                runtime.set_part_opacity_by_index(*link, opacity);
            }
        }
    }
}
