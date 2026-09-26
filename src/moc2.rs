//! Cubism 2 (.moc) parser and default-pose mesh generation.
use std::{collections::HashMap, sync::Arc};

use crate::moc3::{Moc3DrawableMesh, Moc3DrawableVertex};

const MAX_OBJECTS: usize = 2_000_000;
const MAX_ARRAY_ITEMS: usize = 16_000_000;
const FORMAT_VERSION: u8 = 11;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Moc2Error {
    #[error("invalid moc2 signature")]
    InvalidMagic,
    #[error("unsupported moc2 format version {0}")]
    UnsupportedVersion(u8),
    #[error("unexpected end of moc2 data at byte {0}")]
    UnexpectedEof(usize),
    #[error("invalid moc2 integer encoding at byte {0}")]
    InvalidInteger(usize),
    #[error("invalid moc2 object reference {0}")]
    InvalidReference(i32),
    #[error("unsupported moc2 object type {0}")]
    UnsupportedObjectType(u32),
    #[error("invalid moc2 model data: {0}")]
    InvalidData(&'static str),
    #[error("unsupported moc2 texture option {0:#x}")]
    UnsupportedTextureOption(i32),
}

type Ref = Arc<Object>;

#[derive(Debug)]
#[allow(dead_code)]
enum Object {
    Text(String),
    Array(Vec<Ref>),
    Integers(Vec<i32>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Avatar {
        drawables: Ref,
        deformers: Ref,
    },
    Model {
        parameters: Option<Ref>,
        parts: Ref,
        width: i32,
        height: i32,
    },
    Part {
        visible: bool,
        deformers: Ref,
        drawables: Ref,
    },
    Mesh {
        id: Option<String>,
        target: Option<String>,
        pivots: Ref,
        draw_orders: Ref,
        opacities: Ref,
        texture: i32,
        point_count: usize,
        polygon_count: usize,
        indices: Ref,
        positions: Ref,
        uvs: Ref,
        flags: i32,
        clips: Option<String>,
    },
    Warp {
        id: Option<String>,
        target: Option<String>,
        columns: usize,
        rows: usize,
        pivots: Ref,
        positions: Ref,
        opacities: Option<Ref>,
    },
    Rotation {
        id: Option<String>,
        target: Option<String>,
        pivots: Ref,
        affines: Ref,
        opacities: Option<Ref>,
    },
    PivotManager(Ref),
    ParameterPivots {
        id: Option<String>,
        values: Ref,
    },
    Affine {
        x: f32,
        y: f32,
        sx: f32,
        sy: f32,
        angle: f32,
        rx: bool,
        ry: bool,
    },
    Parameter {
        id: Option<String>,
        default: f32,
    },
    ParameterSet(Ref),
}

#[derive(Debug, Clone)]
/// A parsed Cubism 2 model with drawables evaluated at its declared default parameters.
///
/// Its drawables use Neocari's shared MOC3 renderer-facing mesh representation and can
/// be passed directly to WgpuMeshBuffers::from_drawables.
pub struct Moc2Model {
    version: u8,
    canvas_width: u32,
    canvas_height: u32,
    drawables: Vec<Moc3DrawableMesh>,
}

impl Moc2Model {
    /// Parse a Cubism 2 .moc file and generate meshes in the default pose.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Moc2Error> {
        let mut reader = Reader::new(bytes);
        if reader.read_byte()? != b'm' || reader.read_byte()? != b'o' || reader.read_byte()? != b'c'
        {
            return Err(Moc2Error::InvalidMagic);
        }
        let version = reader.read_byte()?;
        if version == 0 || version > FORMAT_VERSION {
            return Err(Moc2Error::UnsupportedVersion(version));
        }
        reader.version = version;
        let root = reader
            .read_object()?
            .ok_or(Moc2Error::InvalidData("missing model root"))?;
        if version >= 8 && (reader.read_u16()? != 0x8888 || reader.read_u16()? != 0x8888) {
            return Err(Moc2Error::InvalidData("invalid end marker"));
        }
        let Object::Model {
            parameters,
            parts,
            width,
            height,
        } = root.as_ref()
        else {
            return Err(Moc2Error::InvalidData("root object is not a model"));
        };
        let canvas_width = u32::try_from(*width)
            .ok()
            .filter(|v| *v > 0)
            .ok_or(Moc2Error::InvalidData("invalid canvas width"))?;
        let canvas_height = u32::try_from(*height)
            .ok()
            .filter(|v| *v > 0)
            .ok_or(Moc2Error::InvalidData("invalid canvas height"))?;
        let defaults = read_parameter_defaults(parameters.as_ref())?;
        let mut meshes = Vec::<(Ref, bool)>::new();
        let mut deformer_objects = Vec::new();
        for part in object_array(parts)? {
            let Object::Part {
                visible,
                deformers,
                drawables,
            } = part.as_ref()
            else {
                return Err(Moc2Error::InvalidData("parts list contains a non-part"));
            };
            deformer_objects.extend(object_array(deformers)?.iter().cloned());
            meshes.extend(
                object_array(drawables)?
                    .iter()
                    .cloned()
                    .map(|mesh| (mesh, *visible)),
            );
        }
        let deformers = index_deformers(&deformer_objects)?;
        let mesh_indices = index_meshes(&meshes)?;
        let mut drawables = Vec::with_capacity(meshes.len());
        let mut transforms = HashMap::new();
        for id in deformers.keys() {
            evaluate_deformer_transform(id, &deformers, &defaults, &mut transforms, 0)?;
        }
        for (mesh, part_visible) in &meshes {
            let Object::Mesh {
                id: _,
                target,
                pivots,
                draw_orders,
                opacities,
                texture,
                point_count,
                polygon_count,
                indices,
                positions,
                uvs,
                flags,
                clips,
            } = mesh.as_ref()
            else {
                return Err(Moc2Error::InvalidData("drawable list contains a non-mesh"));
            };
            if flags & 1 != 0 {
                return Err(Moc2Error::UnsupportedTextureOption(*flags));
            }
            let positions = interpolate_vector(positions, pivots, &defaults)?;
            let draw_order = interpolate_scalar_int(draw_orders, pivots, &defaults)?;
            let opacity = interpolate_scalar_float(opacities, pivots, &defaults)?;
            let uv_values = float_array(uvs)?;
            let raw_indices = integer_array(indices)?;
            if *point_count == 0
                || *polygon_count == 0
                || positions.len() != point_count.saturating_mul(2)
                || uv_values.len() != point_count.saturating_mul(2)
                || raw_indices.len() != polygon_count.saturating_mul(3)
            {
                return Err(Moc2Error::InvalidData(
                    "mesh array lengths do not match counts",
                ));
            }
            let mut vertices = Vec::with_capacity(*point_count);
            for index in 0..*point_count {
                let position = [positions[index * 2], positions[index * 2 + 1]];
                let position = match target.as_deref() {
                    Some(id) if id != "DST_BASE" => transforms
                        .get(id)
                        .ok_or(Moc2Error::InvalidData("missing deformer transform"))?
                        .transform_point(position)?,
                    _ => position,
                };
                // Keep source V because wgpu texture coordinates use a top-left origin.
                vertices.push(Moc3DrawableVertex::new(
                    position,
                    [uv_values[index * 2], uv_values[index * 2 + 1]],
                ));
            }
            let indices = raw_indices
                .iter()
                .map(|index| {
                    u16::try_from(*index)
                        .ok()
                        .filter(|i| usize::from(*i) < *point_count)
                        .ok_or(Moc2Error::InvalidData("mesh index is out of bounds"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let clips = clips
                .as_deref()
                .map(|ids| {
                    ids.split(',')
                        .filter(|id| !id.is_empty())
                        .map(|id| {
                            mesh_indices
                                .get(id)
                                .copied()
                                .map(|i| i as i32)
                                .ok_or(Moc2Error::InvalidData("clip id does not name a mesh"))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            let deformer_opacity = match target.as_deref() {
                Some(id) if id != "DST_BASE" => transforms
                    .get(id)
                    .ok_or(Moc2Error::InvalidData("missing deformer transform"))?
                    .total_opacity(),
                _ => 1.0,
            };
            // V2 texture numbers use the same zero-based numbering as the manifest.
            let texture = if *texture < 0 { 0 } else { *texture };
            let blend_flags = match (flags & 30) >> 1 {
                1 => 4, // Screen composition is represented by its dedicated WGPU blend state.
                2 => 2,
                _ => 0,
            };
            drawables.push(Moc3DrawableMesh::from_parts(
                texture,
                blend_flags,
                if *part_visible {
                    opacity * deformer_opacity
                } else {
                    0.0
                },
                draw_order as f32,
                vertices,
                indices,
                clips,
            ));
        }
        let mut sorted = (0..drawables.len()).collect::<Vec<_>>();
        sorted.sort_by(|a, b| {
            drawables[*a]
                .draw_order()
                .total_cmp(&drawables[*b].draw_order())
                .then_with(|| a.cmp(b))
        });
        for (rank, index) in sorted.into_iter().enumerate() {
            drawables[index].set_render_order(rank as i32);
        }
        Ok(Self {
            version,
            canvas_width,
            canvas_height,
            drawables,
        })
    }

    /// Serialization version stored in the .moc header.
    pub fn format_version(&self) -> u8 {
        self.version
    }
    /// Canvas width in pixels.
    pub fn canvas_width(&self) -> u32 {
        self.canvas_width
    }
    /// Canvas height in pixels.
    pub fn canvas_height(&self) -> u32 {
        self.canvas_height
    }
    /// Drawables in model order, with default parameters and deformer transforms applied.
    pub fn drawables(&self) -> &[Moc3DrawableMesh] {
        &self.drawables
    }
}

fn object_array(value: &Ref) -> Result<&[Ref], Moc2Error> {
    match value.as_ref() {
        Object::Array(values) => Ok(values),
        _ => Err(Moc2Error::InvalidData("expected an object array")),
    }
}
fn integer_array(value: &Ref) -> Result<&[i32], Moc2Error> {
    match value.as_ref() {
        Object::Integers(values) => Ok(values),
        _ => Err(Moc2Error::InvalidData("expected an integer array")),
    }
}
fn float_array(value: &Ref) -> Result<&[f32], Moc2Error> {
    match value.as_ref() {
        Object::Floats(values) => Ok(values),
        _ => Err(Moc2Error::InvalidData("expected a float array")),
    }
}
fn read_parameter_defaults(parameters: Option<&Ref>) -> Result<Vec<(String, f32)>, Moc2Error> {
    let Some(parameters) = parameters else {
        return Ok(Vec::new());
    };
    let Object::ParameterSet(list) = parameters.as_ref() else {
        return Err(Moc2Error::InvalidData(
            "parameter definition set has wrong type",
        ));
    };
    object_array(list)?
        .iter()
        .map(|item| match item.as_ref() {
            Object::Parameter {
                id: Some(id),
                default,
            } => Ok((id.clone(), *default)),
            _ => Err(Moc2Error::InvalidData("invalid parameter definition")),
        })
        .collect()
}

#[derive(Debug)]
struct Dimension {
    id: Option<String>,
    values: Vec<f32>,
}
fn dimensions(pivots: &Ref) -> Result<Vec<Dimension>, Moc2Error> {
    let Object::PivotManager(list) = pivots.as_ref() else {
        return Err(Moc2Error::InvalidData("expected a pivot manager"));
    };
    object_array(list)?
        .iter()
        .map(|item| match item.as_ref() {
            Object::ParameterPivots { id, values } => Ok(Dimension {
                id: id.clone(),
                values: float_array(values)?.to_vec(),
            }),
            _ => Err(Moc2Error::InvalidData("invalid pivot table entry")),
        })
        .collect()
}
fn corner_weights(
    pivots: &Ref,
    defaults: &[(String, f32)],
) -> Result<Vec<(usize, f32)>, Moc2Error> {
    let dims = dimensions(pivots)?
        .into_iter()
        .filter(|d| d.values.len() > 1)
        .collect::<Vec<_>>();
    if dims.len() > 8 {
        return Err(Moc2Error::InvalidData(
            "pivot manager has too many dimensions",
        ));
    }
    let mut lower = Vec::with_capacity(dims.len());
    let mut fraction = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    let mut strides = Vec::with_capacity(dims.len());
    for dim in &dims {
        if dim.values.is_empty() {
            return Err(Moc2Error::InvalidData("empty parameter pivot list"));
        }
        let value = dim
            .id
            .as_deref()
            .and_then(|id| defaults.iter().find(|(key, _)| key == id))
            .map(|(_, value)| *value)
            .unwrap_or(dim.values[0]);
        let mut lo = 0usize;
        let mut t = 0.0f32;
        if value >= dim.values[dim.values.len() - 1] {
            lo = dim.values.len() - 2;
            t = 1.0;
        } else if value > dim.values[0] {
            for i in 0..dim.values.len() - 1 {
                if value <= dim.values[i + 1] {
                    lo = i;
                    let span = dim.values[i + 1] - dim.values[i];
                    t = if span > 0.0 {
                        (value - dim.values[i]) / span
                    } else {
                        0.0
                    };
                    break;
                }
            }
        }
        lower.push(lo);
        fraction.push(t.clamp(0.0, 1.0));
        strides.push(stride);
        stride = stride
            .checked_mul(dim.values.len())
            .ok_or(Moc2Error::InvalidData("pivot table too large"))?;
    }
    let mut corners = Vec::with_capacity(1usize << dims.len());
    for bits in 0..(1usize << dims.len()) {
        let mut index = 0usize;
        let mut weight = 1.0;
        for d in 0..dims.len() {
            let high = bits & (1 << d) != 0;
            index += (lower[d] + usize::from(high)) * strides[d];
            weight *= if high { fraction[d] } else { 1.0 - fraction[d] };
        }
        corners.push((index, weight));
    }
    if corners.is_empty() {
        corners.push((0, 1.0));
    }
    Ok(corners)
}
fn pivot_count(pivots: &Ref) -> Result<usize, Moc2Error> {
    dimensions(pivots)?.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(dim.values.len())
            .ok_or(Moc2Error::InvalidData("pivot table too large"))
    })
}
fn interpolate_vector(
    values: &Ref,
    pivots: &Ref,
    defaults: &[(String, f32)],
) -> Result<Vec<f32>, Moc2Error> {
    let count = pivot_count(pivots)?;
    if count == 0 {
        return Err(Moc2Error::InvalidData("pivot vector length is invalid"));
    }
    let corners = corner_weights(pivots, defaults)?;
    match values.as_ref() {
        Object::Floats(values) => {
            if values.len() % count != 0 {
                return Err(Moc2Error::InvalidData("pivot vector length is invalid"));
            }
            let stride = values.len() / count;
            let mut output = vec![0.0; stride];
            for (pivot, weight) in corners {
                let start = pivot
                    .checked_mul(stride)
                    .ok_or(Moc2Error::InvalidData("pivot vector too large"))?;
                let slice = values
                    .get(start..start + stride)
                    .ok_or(Moc2Error::InvalidData("pivot index out of bounds"))?;
                for (dst, src) in output.iter_mut().zip(slice) {
                    *dst += src * weight;
                }
            }
            Ok(output)
        }
        Object::Array(keyforms) => {
            if keyforms.len() != count {
                return Err(Moc2Error::InvalidData("pivot keyform count is invalid"));
            }
            let first = keyforms
                .first()
                .ok_or(Moc2Error::InvalidData("missing pivot keyforms"))?;
            let stride = float_array(first)?.len();
            let mut output = vec![0.0; stride];
            for (pivot, weight) in corners {
                let values = keyforms
                    .get(pivot)
                    .ok_or(Moc2Error::InvalidData("pivot index out of bounds"))?;
                let values = float_array(values)?;
                if values.len() != stride {
                    return Err(Moc2Error::InvalidData("pivot keyform lengths differ"));
                }
                for (dst, src) in output.iter_mut().zip(values) {
                    *dst += src * weight;
                }
            }
            Ok(output)
        }
        _ => Err(Moc2Error::InvalidData("expected pivot vector data")),
    }
}
fn interpolate_scalar_float(
    values: &Ref,
    pivots: &Ref,
    defaults: &[(String, f32)],
) -> Result<f32, Moc2Error> {
    let values = float_array(values)?;
    corner_weights(pivots, defaults)?
        .iter()
        .try_fold(0.0, |sum, (index, weight)| {
            values
                .get(*index)
                .map(|value| sum + value * weight)
                .ok_or(Moc2Error::InvalidData("pivot index exceeds float array"))
        })
}
fn interpolate_scalar_int(
    values: &Ref,
    pivots: &Ref,
    defaults: &[(String, f32)],
) -> Result<i32, Moc2Error> {
    let values = integer_array(values)?;
    let mut result = 0.0f32;
    for (index, weight) in corner_weights(pivots, defaults)? {
        let value = values
            .get(index)
            .ok_or(Moc2Error::InvalidData("pivot index exceeds integer array"))?;
        result += *value as f32 * weight;
    }
    Ok(result.round() as i32)
}
fn index_meshes(meshes: &[(Ref, bool)]) -> Result<HashMap<String, usize>, Moc2Error> {
    let mut result = HashMap::new();
    for (index, (mesh, _)) in meshes.iter().enumerate() {
        let Object::Mesh { id: Some(id), .. } = mesh.as_ref() else {
            return Err(Moc2Error::InvalidData("mesh has no id"));
        };
        result.insert(id.clone(), index);
    }
    Ok(result)
}
fn index_deformers(deformers: &[Ref]) -> Result<HashMap<String, Ref>, Moc2Error> {
    let mut result = HashMap::new();
    for item in deformers {
        let id = match item.as_ref() {
            Object::Warp { id, .. } | Object::Rotation { id, .. } => id,
            _ => return Err(Moc2Error::InvalidData("unknown deformer type")),
        };
        if let Some(id) = id {
            result.insert(id.clone(), Arc::clone(item));
        }
    }
    Ok(result)
}
fn affine_values(
    affines: &Ref,
    pivots: &Ref,
    defaults: &[(String, f32)],
) -> Result<([f32; 5], [bool; 2]), Moc2Error> {
    let entries = object_array(affines)?;
    let mut values = [0.0f32; 5];
    let mut reflect = [false; 2];
    let mut strongest = -1.0f32;
    for (index, weight) in corner_weights(pivots, defaults)? {
        let Object::Affine {
            x,
            y,
            sx,
            sy,
            angle,
            rx,
            ry,
        } = entries
            .get(index)
            .ok_or(Moc2Error::InvalidData("pivot index exceeds affine list"))?
            .as_ref()
        else {
            return Err(Moc2Error::InvalidData("invalid affine entry"));
        };
        values[0] += x * weight;
        values[1] += y * weight;
        values[2] += sx * weight;
        values[3] += sy * weight;
        values[4] += angle * weight;
        if weight > strongest {
            strongest = weight;
            reflect = [*rx, *ry];
        }
    }
    Ok((values, reflect))
}
#[derive(Debug, Clone)]
enum DeformerTransform {
    Warp {
        rows: usize,
        columns: usize,
        grid: Vec<f32>,
        total_scale: f32,
        total_opacity: f32,
    },
    Rotation {
        origin: [f32; 2],
        angle_degrees: f32,
        total_scale: f32,
        reflect: [bool; 2],
        total_opacity: f32,
    },
}
impl DeformerTransform {
    fn transform_point(&self, point: [f32; 2]) -> Result<[f32; 2], Moc2Error> {
        let output = match self {
            Self::Warp {
                rows,
                columns,
                grid,
                ..
            } => map_warp_grid(point, grid, *rows, *columns)?,
            Self::Rotation {
                origin,
                angle_degrees,
                total_scale,
                reflect,
                ..
            } => {
                let radians = angle_degrees.to_radians();
                let (sin, cos) = radians.sin_cos();
                let sx = total_scale * if reflect[0] { -1.0 } else { 1.0 };
                let sy = total_scale * if reflect[1] { -1.0 } else { 1.0 };
                [
                    cos * sx * point[0] - sin * sy * point[1] + origin[0],
                    sin * sx * point[0] + cos * sy * point[1] + origin[1],
                ]
            }
        };
        if output.iter().all(|value| value.is_finite()) {
            Ok(output)
        } else {
            Err(Moc2Error::InvalidData(
                "deformer generated a non-finite position",
            ))
        }
    }
    fn total_scale(&self) -> f32 {
        match self {
            Self::Warp { total_scale, .. } | Self::Rotation { total_scale, .. } => *total_scale,
        }
    }
    fn total_opacity(&self) -> f32 {
        match self {
            Self::Warp { total_opacity, .. } | Self::Rotation { total_opacity, .. } => {
                *total_opacity
            }
        }
    }
    fn is_rotation(&self) -> bool {
        matches!(self, Self::Rotation { .. })
    }
}

fn evaluate_deformer_transform(
    id: &str,
    deformers: &HashMap<String, Ref>,
    defaults: &[(String, f32)],
    transforms: &mut HashMap<String, DeformerTransform>,
    depth: usize,
) -> Result<(), Moc2Error> {
    if id == "DST_BASE" || transforms.contains_key(id) {
        return Ok(());
    }
    if depth > 32 {
        return Err(Moc2Error::InvalidData(
            "deformer chain is cyclic or too deep",
        ));
    }
    let object = deformers.get(id).ok_or(Moc2Error::InvalidData(
        "drawable references a missing deformer",
    ))?;
    let parent_id = match object.as_ref() {
        Object::Warp { target, .. } | Object::Rotation { target, .. } => target
            .as_deref()
            .filter(|target| *target != "DST_BASE")
            .map(str::to_owned),
        _ => return Err(Moc2Error::InvalidData("unknown deformer type")),
    };
    if let Some(parent_id) = parent_id.as_deref() {
        evaluate_deformer_transform(parent_id, deformers, defaults, transforms, depth + 1)?;
    }
    let parent = parent_id
        .as_deref()
        .map(|parent_id| {
            transforms
                .get(parent_id)
                .cloned()
                .ok_or(Moc2Error::InvalidData("missing parent deformer transform"))
        })
        .transpose()?;
    let opacity = match object.as_ref() {
        Object::Warp {
            pivots, opacities, ..
        }
        | Object::Rotation {
            pivots, opacities, ..
        } => opacities
            .as_ref()
            .map(|values| interpolate_scalar_float(values, pivots, defaults))
            .transpose()?
            .unwrap_or(1.0),
        _ => unreachable!(),
    };
    let total_opacity = parent
        .as_ref()
        .map_or(1.0, DeformerTransform::total_opacity)
        * opacity;
    let transform = match object.as_ref() {
        Object::Warp {
            columns,
            rows,
            pivots,
            positions,
            ..
        } => {
            if *rows == 0 || *columns == 0 {
                return Err(Moc2Error::InvalidData("empty warp grid"));
            }
            let mut grid = interpolate_vector(positions, pivots, defaults)?;
            if grid.len() != (rows + 1) * (columns + 1) * 2 {
                return Err(Moc2Error::InvalidData("invalid warp grid length"));
            }
            if let Some(parent) = parent.as_ref() {
                for coordinate in grid.as_chunks_mut::<2>().0 {
                    let transformed = parent.transform_point([coordinate[0], coordinate[1]])?;
                    coordinate.copy_from_slice(&transformed);
                }
            }
            DeformerTransform::Warp {
                rows: *rows,
                columns: *columns,
                grid,
                total_scale: parent.as_ref().map_or(1.0, DeformerTransform::total_scale),
                total_opacity,
            }
        }
        Object::Rotation {
            pivots, affines, ..
        } => {
            let (a, reflect) = affine_values(affines, pivots, defaults)?;
            let (origin, angle_degrees, total_scale) = if let Some(parent) = parent.as_ref() {
                let origin = parent.transform_point([a[0], a[1]])?;
                let direction = [0.0, if parent.is_rotation() { -10.0 } else { -0.1 }];
                let transformed_direction_point =
                    parent.transform_point([a[0] + direction[0], a[1] + direction[1]])?;
                let transformed_direction = [
                    transformed_direction_point[0] - origin[0],
                    transformed_direction_point[1] - origin[1],
                ];
                let source_angle = direction[1].atan2(direction[0]);
                let destination_angle = transformed_direction[1].atan2(transformed_direction[0]);
                let mut parent_angle = source_angle - destination_angle;
                while parent_angle < -std::f32::consts::PI {
                    parent_angle += 2.0 * std::f32::consts::PI;
                }
                while parent_angle > std::f32::consts::PI {
                    parent_angle -= 2.0 * std::f32::consts::PI;
                }
                (
                    origin,
                    a[4] - parent_angle.to_degrees(),
                    parent.total_scale() * a[2],
                )
            } else {
                ([a[0], a[1]], a[4], a[2])
            };
            DeformerTransform::Rotation {
                origin,
                angle_degrees,
                total_scale,
                reflect,
                total_opacity,
            }
        }
        _ => return Err(Moc2Error::InvalidData("unknown deformer type")),
    };
    transforms.insert(id.to_owned(), transform);
    Ok(())
}
fn map_warp_grid(
    point: [f32; 2],
    grid: &[f32],
    rows: usize,
    columns: usize,
) -> Result<[f32; 2], Moc2Error> {
    let row_coord = point[0] * rows as f32;
    let col_coord = point[1] * columns as f32;
    let row = (row_coord.floor() as isize).clamp(0, rows as isize - 1) as usize;
    let col = (col_coord.floor() as isize).clamp(0, columns as isize - 1) as usize;
    let row_fraction = row_coord - row as f32;
    let col_fraction = col_coord - col as f32;
    let row_stride = rows + 1;
    let at = |r: usize, c: usize| {
        let index = (r + c * row_stride) * 2;
        [grid[index], grid[index + 1]]
    };
    let p00 = at(row, col);
    let p10 = at(row + 1, col);
    let p01 = at(row, col + 1);
    let p11 = at(row + 1, col + 1);
    let inside_grid = row_coord >= 0.0
        && row_coord < rows as f32
        && col_coord >= 0.0
        && col_coord < columns as f32;
    let output = if inside_grid {
        if row_fraction + col_fraction < 1.0 {
            std::array::from_fn(|axis| {
                p00[axis] * (1.0 - row_fraction - col_fraction)
                    + p10[axis] * row_fraction
                    + p01[axis] * col_fraction
            })
        } else {
            std::array::from_fn(|axis| {
                p11[axis] * (row_fraction + col_fraction - 1.0)
                    + p01[axis] * (1.0 - row_fraction)
                    + p10[axis] * (1.0 - col_fraction)
            })
        }
    } else {
        std::array::from_fn(|axis| {
            let low = p00[axis] + (p10[axis] - p00[axis]) * row_fraction;
            let high = p01[axis] + (p11[axis] - p01[axis]) * row_fraction;
            low + (high - low) * col_fraction
        })
    };
    if output.iter().all(|v| v.is_finite()) {
        Ok(output)
    } else {
        Err(Moc2Error::InvalidData(
            "warp generated a non-finite position",
        ))
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
    bit_offset: u8,
    current_byte: u8,
    version: u8,
    objects: Vec<Ref>,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            offset: 0,
            bit_offset: 0,
            current_byte: 0,
            version: 0,
            objects: Vec::new(),
        }
    }
    fn align(&mut self) {
        self.bit_offset = 0;
    }
    fn read_byte(&mut self) -> Result<u8, Moc2Error> {
        self.align();
        let value = *self
            .bytes
            .get(self.offset)
            .ok_or(Moc2Error::UnexpectedEof(self.offset))?;
        self.offset += 1;
        Ok(value)
    }
    fn read_bit(&mut self) -> Result<bool, Moc2Error> {
        if self.bit_offset == 0 || self.bit_offset == 8 {
            self.current_byte = self.read_byte()?;
            self.bit_offset = 0;
        }
        let value = self.current_byte >> (7 - self.bit_offset) & 1 != 0;
        self.bit_offset += 1;
        Ok(value)
    }
    fn read_number(&mut self) -> Result<u32, Moc2Error> {
        self.align();
        let start = self.offset;
        let mut value = 0;
        for i in 0..4 {
            let byte = self.read_byte()?;
            value = (value << 7) | u32::from(byte & 0x7f);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            if i == 3 {
                return Err(Moc2Error::InvalidInteger(start));
            }
        }
        Err(Moc2Error::InvalidInteger(start))
    }
    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], Moc2Error> {
        self.align();
        let start = self.offset;
        let end = start
            .checked_add(N)
            .ok_or(Moc2Error::UnexpectedEof(start))?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(Moc2Error::UnexpectedEof(start))?;
        self.offset = end;
        Ok(bytes.try_into().expect("fixed-size slice"))
    }
    fn read_u16(&mut self) -> Result<u16, Moc2Error> {
        Ok(u16::from_be_bytes(self.read_exact()?))
    }
    fn read_i32(&mut self) -> Result<i32, Moc2Error> {
        Ok(i32::from_be_bytes(self.read_exact()?))
    }
    fn read_f32(&mut self) -> Result<f32, Moc2Error> {
        Ok(f32::from_bits(self.read_i32()? as u32))
    }
    fn read_bool(&mut self) -> Result<bool, Moc2Error> {
        Ok(self.read_byte()? != 0)
    }
    fn read_f64(&mut self) -> Result<f64, Moc2Error> {
        Ok(f64::from_bits(u64::from_be_bytes(self.read_exact()?)))
    }
    fn read_string(&mut self) -> Result<String, Moc2Error> {
        self.align();
        let len = self.read_number()? as usize;
        let start = self.offset;
        let end = start
            .checked_add(len)
            .ok_or(Moc2Error::UnexpectedEof(start))?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(Moc2Error::UnexpectedEof(start))?;
        self.offset = end;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Moc2Error::InvalidData("invalid UTF-8 string"))
    }
    fn array_length(&mut self) -> Result<usize, Moc2Error> {
        let len = self.read_number()? as usize;
        if len > MAX_ARRAY_ITEMS {
            Err(Moc2Error::InvalidData("array exceeds parser limits"))
        } else {
            Ok(len)
        }
    }
    fn read_ints(&mut self) -> Result<Object, Moc2Error> {
        let len = self.array_length()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| Moc2Error::InvalidData("integer array allocation failed"))?;
        for _ in 0..len {
            values.push(self.read_i32()?);
        }
        Ok(Object::Integers(values))
    }
    fn read_doubles(&mut self) -> Result<Object, Moc2Error> {
        let len = self.array_length()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| Moc2Error::InvalidData("double array allocation failed"))?;
        for _ in 0..len {
            values.push(self.read_f64()?);
        }
        Ok(Object::Doubles(values))
    }
    fn read_floats(&mut self) -> Result<Object, Moc2Error> {
        let len = self.array_length()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| Moc2Error::InvalidData("float array allocation failed"))?;
        for _ in 0..len {
            values.push(self.read_f32()?);
        }
        Ok(Object::Floats(values))
    }
    fn read_optional_text(&mut self) -> Result<Option<String>, Moc2Error> {
        self.read_object()?
            .map(|item| match item.as_ref() {
                Object::Text(text) => Ok(text.clone()),
                _ => Err(Moc2Error::InvalidData("expected an id string")),
            })
            .transpose()
    }
    fn read_optional_clip(&mut self) -> Result<Option<String>, Moc2Error> {
        Ok(match self.read_object()? {
            Some(item) => match item.as_ref() {
                Object::Text(text) => Some(text.clone()),
                // Some older v2 exporters serialize opaque clipping metadata here.
                // It is not an ID list the renderer can resolve, so leave it unmasked.
                _ => None,
            },
            None => None,
        })
    }
    fn read_object(&mut self) -> Result<Option<Ref>, Moc2Error> {
        self.align();
        let kind = self.read_number()?;
        if kind == 0 {
            return Ok(None);
        }
        if kind == 33 {
            let index = self.read_i32()?;
            return usize::try_from(index)
                .ok()
                .and_then(|i| self.objects.get(i))
                .cloned()
                .map(Some)
                .ok_or(Moc2Error::InvalidReference(index));
        }
        if self.objects.len() >= MAX_OBJECTS {
            return Err(Moc2Error::InvalidData("object count exceeds parser limits"));
        }
        let object = match kind {
            1 | 50 | 51 | 60 | 134 => Object::Text(self.read_string()?),
            15 => {
                let len = self.array_length()?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(len)
                    .map_err(|_| Moc2Error::InvalidData("object array allocation failed"))?;
                for _ in 0..len {
                    values.push(
                        self.read_object()?
                            .ok_or(Moc2Error::InvalidData("null array entry"))?,
                    );
                }
                Object::Array(values)
            }
            16 | 25 => self.read_ints()?,
            26 => self.read_doubles()?,
            27 => self.read_floats()?,
            65 => self.read_warp()?,
            66 => Object::PivotManager(
                self.read_object()?
                    .ok_or(Moc2Error::InvalidData("null pivot table"))?,
            ),
            67 => self.read_param_pivots()?,
            68 => self.read_rotation()?,
            69 => self.read_affine()?,
            70 => self.read_mesh()?,
            131 => self.read_parameter()?,
            133 => self.read_part()?,
            142 => self.read_avatar()?,
            136 => self.read_model()?,
            137 => Object::ParameterSet(
                self.read_object()?
                    .ok_or(Moc2Error::InvalidData("null parameter list"))?,
            ),
            _ => return Err(Moc2Error::UnsupportedObjectType(kind)),
        };
        let object = Arc::new(object);
        self.objects.push(Arc::clone(&object));
        Ok(Some(object))
    }
    fn read_model(&mut self) -> Result<Object, Moc2Error> {
        let parameters = self.read_object()?;
        let parts = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null model parts"))?;
        Ok(Object::Model {
            parameters,
            parts,
            width: self.read_i32()?,
            height: self.read_i32()?,
        })
    }
    fn read_part(&mut self) -> Result<Object, Moc2Error> {
        let _locked = self.read_bit()?;
        let visible = self.read_bit(); // keep the packed flag byte
        let visible = visible?;
        let _id = self.read_optional_text()?;
        let deformers = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null part deformer list"))?;
        let drawables = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null part drawable list"))?;
        Ok(Object::Part {
            visible,
            deformers,
            drawables,
        })
    }
    fn read_avatar(&mut self) -> Result<Object, Moc2Error> {
        let _id = self.read_optional_text()?;
        let drawables = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null avatar drawables"))?;
        let deformers = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null avatar deformers"))?;
        Ok(Object::Avatar {
            drawables,
            deformers,
        })
    }
    fn read_mesh(&mut self) -> Result<Object, Moc2Error> {
        let id = self.read_optional_text()?;
        let target = self.read_optional_text()?;
        let pivots = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null mesh pivot manager"))?;
        let _average_order = self.read_i32()?;
        let draw_orders = Arc::new(self.read_ints()?);
        let opacities = Arc::new(self.read_floats()?);
        let clips = if self.version >= 11 {
            self.read_optional_clip()?
        } else {
            None
        };
        let texture = self.read_i32()?;
        let point_count = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|n| *n <= MAX_ARRAY_ITEMS)
            .ok_or(Moc2Error::InvalidData("invalid point count"))?;
        let polygon_count = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|n| *n <= MAX_ARRAY_ITEMS / 3)
            .ok_or(Moc2Error::InvalidData("invalid polygon count"))?;
        let indices = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null mesh indices"))?;
        let positions = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null mesh positions"))?;
        let uvs = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null mesh UVs"))?;
        let flags = if self.version >= 8 {
            self.read_i32()?
        } else {
            0
        };
        Ok(Object::Mesh {
            id,
            target,
            pivots,
            draw_orders,
            opacities,
            texture,
            point_count,
            polygon_count,
            indices,
            positions,
            uvs,
            flags,
            clips,
        })
    }
    fn read_warp(&mut self) -> Result<Object, Moc2Error> {
        let id = self.read_optional_text()?;
        let target = self.read_optional_text()?;
        let columns = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|n| *n <= 4096)
            .ok_or(Moc2Error::InvalidData("invalid warp columns"))?;
        let rows = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|n| *n <= 4096)
            .ok_or(Moc2Error::InvalidData("invalid warp rows"))?;
        let pivots = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null warp pivots"))?;
        let positions = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null warp positions"))?;
        let opacities = if self.version >= 10 {
            Some(Arc::new(self.read_floats()?))
        } else {
            None
        };
        Ok(Object::Warp {
            id,
            target,
            columns,
            rows,
            pivots,
            positions,
            opacities,
        })
    }
    fn read_rotation(&mut self) -> Result<Object, Moc2Error> {
        let id = self.read_optional_text()?;
        let target = self.read_optional_text()?;
        let pivots = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null rotation pivots"))?;
        let affines = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null affine array"))?;
        let opacities = if self.version >= 10 {
            Some(Arc::new(self.read_floats()?))
        } else {
            None
        };
        Ok(Object::Rotation {
            id,
            target,
            pivots,
            affines,
            opacities,
        })
    }
    fn read_param_pivots(&mut self) -> Result<Object, Moc2Error> {
        let id = self.read_optional_text()?;
        let count = self.read_i32()?;
        if count < 0 || count as usize > MAX_ARRAY_ITEMS {
            return Err(Moc2Error::InvalidData("invalid pivot count"));
        }
        let values = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null pivot values"))?;
        if float_array(&values)?.len() != count as usize {
            return Err(Moc2Error::InvalidData("pivot count mismatch"));
        }
        Ok(Object::ParameterPivots { id, values })
    }
    fn read_affine(&mut self) -> Result<Object, Moc2Error> {
        let x = self.read_f32()?;
        let y = self.read_f32()?;
        let sx = self.read_f32()?;
        let sy = self.read_f32()?;
        let angle = self.read_f32()?;
        let (rx, ry) = if self.version >= 10 {
            (self.read_bool()?, self.read_bool()?)
        } else {
            (false, false)
        };
        Ok(Object::Affine {
            x,
            y,
            sx,
            sy,
            angle,
            rx,
            ry,
        })
    }
    fn read_parameter(&mut self) -> Result<Object, Moc2Error> {
        let _min = self.read_f32()?;
        let _max = self.read_f32()?;
        let default = self.read_f32()?;
        let id = self.read_optional_text()?;
        Ok(Object::Parameter { id, default })
    }
}
