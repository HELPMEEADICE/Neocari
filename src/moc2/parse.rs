//! Binary `.moc` reader and scene conversion.

use std::{collections::HashMap, sync::Arc};

use super::{
    Moc2Error,
    scene::{
        Moc2Affine, Moc2Canvas, Moc2Deformer, Moc2DeformerKind, Moc2Drawable, Moc2Parameter,
        Moc2Part, Moc2PivotDimension, Moc2PivotTable, Moc2Scene,
    },
};

const MAX_OBJECTS: usize = 2_000_000;
const MAX_ARRAY_ITEMS: usize = 16_000_000;
const MAX_GRID_CELLS: usize = 4096;
const FORMAT_VERSION: u8 = 11;
const CLIP_ID_VERSION: u8 = 11;
const OPACITY_VERSION: u8 = 10;
const TEXTURE_OPTION_VERSION: u8 = 8;
const REFLECT_VERSION: u8 = 10;
const BASE_TARGET: &str = "DST_BASE";

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
        id: Option<String>,
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
        minimum: f32,
        maximum: f32,
        default: f32,
    },
    ParameterSet(Ref),
}

/// Parses a `.moc` file into a typed, animatable scene.
pub(crate) fn parse_scene(bytes: &[u8]) -> Result<Moc2Scene, Moc2Error> {
    let mut reader = Reader::new(bytes);
    if reader.read_byte()? != b'm' || reader.read_byte()? != b'o' || reader.read_byte()? != b'c' {
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
    if version >= TEXTURE_OPTION_VERSION
        && (reader.read_u16()? != 0x8888 || reader.read_u16()? != 0x8888)
    {
        return Err(Moc2Error::InvalidData("invalid end marker"));
    }
    // Object references are fully resolved once the root is parsed. Release the
    // parser index while keeping the reachable graph alive through `root`.
    reader.objects.clear();

    let Object::Model {
        parameters,
        parts,
        width,
        height,
    } = root.as_ref()
    else {
        return Err(Moc2Error::InvalidData("root object is not a model"));
    };
    let canvas = Moc2Canvas::new(
        u32::try_from(*width)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(Moc2Error::InvalidData("invalid canvas width"))?,
        u32::try_from(*height)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(Moc2Error::InvalidData("invalid canvas height"))?,
    );

    let mut registry = ParameterRegistry::new(read_parameter_definitions(parameters.as_ref())?);

    let mut part_deformers = Vec::new();
    let mut part_drawables = Vec::new();
    let mut parts_data = Vec::new();
    for part in object_array(parts)? {
        let Object::Part {
            id,
            visible,
            deformers,
            drawables,
        } = part.as_ref()
        else {
            return Err(Moc2Error::InvalidData("parts list contains a non-part"));
        };
        let part_index = parts_data.len();
        let deformers = object_array(deformers)?;
        let drawables = object_array(drawables)?;
        part_deformers.extend(deformers.iter().cloned());
        part_drawables.extend(
            drawables
                .iter()
                .cloned()
                .map(|drawable| (drawable, part_index)),
        );
        parts_data.push(Moc2Part {
            id: id.clone().unwrap_or_default(),
            visible: *visible,
            drawables: Vec::new(),
        });
    }

    let deformers = convert_deformers(&part_deformers, &mut registry)?;
    let deformer_index: HashMap<String, usize> = deformers
        .iter()
        .enumerate()
        .filter(|(_, deformer)| !deformer.id.is_empty())
        .map(|(index, deformer)| (deformer.id.clone(), index))
        .collect();

    let mut drawables = Vec::with_capacity(part_drawables.len());
    let mut clip_ids = Vec::with_capacity(part_drawables.len());
    for (index, (object, part_index)) in part_drawables.iter().enumerate() {
        let (drawable, clips) =
            convert_drawable(object, *part_index, &deformer_index, &mut registry)?;
        parts_data[*part_index].drawables.push(index);
        drawables.push(drawable);
        clip_ids.push(clips);
    }

    let drawable_index: HashMap<String, usize> = drawables
        .iter()
        .enumerate()
        .filter(|(_, drawable)| !drawable.id.is_empty())
        .map(|(index, drawable)| (drawable.id.clone(), index))
        .collect();
    for (drawable, clips) in drawables.iter_mut().zip(&clip_ids) {
        drawable.clips = resolve_clips(clips, &drawable_index);
    }

    Ok(Moc2Scene::from_parts(
        version,
        canvas,
        registry.into_parameters(),
        parts_data,
        drawables,
        deformers,
    ))
}

fn read_parameter_definitions(parameters: Option<&Ref>) -> Result<Vec<Moc2Parameter>, Moc2Error> {
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
                minimum,
                maximum,
                default,
            } => Ok(Moc2Parameter::new(id.clone(), *minimum, *maximum, *default)),
            Object::Parameter { id: None, .. } => {
                Err(Moc2Error::InvalidData("parameter definition has no id"))
            }
            _ => Err(Moc2Error::InvalidData("invalid parameter definition")),
        })
        .collect()
}

struct ParameterRegistry {
    parameters: Vec<Moc2Parameter>,
    index: HashMap<String, usize>,
}

impl ParameterRegistry {
    fn new(parameters: Vec<Moc2Parameter>) -> Self {
        let mut registry = Self {
            parameters: Vec::with_capacity(parameters.len() + 8),
            index: HashMap::new(),
        };
        for parameter in parameters {
            registry
                .index
                .entry(parameter.id.clone())
                .or_insert_with(|| {
                    let index = registry.parameters.len();
                    registry.parameters.push(parameter);
                    index
                });
        }
        registry
    }

    /// Resolves a parameter id, appending an unconstrained parameter when the
    /// model does not declare it. Cubism 2 does the same for ids that only
    /// appear inside motion or physics files.
    fn resolve(&mut self, id: &str) -> usize {
        if let Some(index) = self.index.get(id) {
            return *index;
        }
        let index = self.parameters.len();
        self.parameters
            .push(Moc2Parameter::new(id, -1_000_000.0, 1_000_000.0, 0.0));
        self.index.insert(id.to_owned(), index);
        index
    }

    fn into_parameters(self) -> Vec<Moc2Parameter> {
        self.parameters
    }
}

fn pivot_table(
    pivots: &Ref,
    registry: &mut ParameterRegistry,
) -> Result<Moc2PivotTable, Moc2Error> {
    let Object::PivotManager(list) = pivots.as_ref() else {
        return Err(Moc2Error::InvalidData("expected a pivot manager"));
    };
    let mut dimensions = Vec::new();
    for item in object_array(list)? {
        let Object::ParameterPivots { id, values } = item.as_ref() else {
            return Err(Moc2Error::InvalidData("invalid pivot table entry"));
        };
        let id = id.as_deref().ok_or(Moc2Error::InvalidData(
            "pivot dimension has no parameter id",
        ))?;
        let values = float_array(values)?.to_vec();
        if values.is_empty() {
            return Err(Moc2Error::InvalidData("empty parameter pivot list"));
        }
        if values.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(Moc2Error::InvalidData(
                "parameter pivot values are not sorted",
            ));
        }
        dimensions.push(Moc2PivotDimension {
            parameter_index: registry.resolve(id),
            values,
        });
    }
    Moc2PivotTable::new(dimensions).ok_or(Moc2Error::InvalidData(
        "pivot table has too many dimensions",
    ))
}

/// Reads keyform-major data that is stored either as one flat array or as one
/// array per keyform.
fn pivot_values(values: &Ref, pivot_count: usize, stride: usize) -> Result<Vec<f32>, Moc2Error> {
    let expected = pivot_count
        .checked_mul(stride)
        .ok_or(Moc2Error::InvalidData("pivot data is too large"))?;
    match values.as_ref() {
        Object::Floats(values) => {
            if values.len() != expected {
                return Err(Moc2Error::InvalidData("pivot vector length is invalid"));
            }
            Ok(values.to_vec())
        }
        Object::Array(keyforms) => {
            if keyforms.len() != pivot_count {
                return Err(Moc2Error::InvalidData("pivot keyform count is invalid"));
            }
            let mut output = Vec::with_capacity(expected);
            for keyform in keyforms {
                let values = float_array(keyform)?;
                if values.len() != stride {
                    return Err(Moc2Error::InvalidData("pivot keyform lengths differ"));
                }
                output.extend_from_slice(values);
            }
            Ok(output)
        }
        _ => Err(Moc2Error::InvalidData("expected pivot vector data")),
    }
}

fn convert_deformers(
    objects: &[Ref],
    registry: &mut ParameterRegistry,
) -> Result<Vec<Moc2Deformer>, Moc2Error> {
    let mut parsed = Vec::with_capacity(objects.len());
    for object in objects {
        parsed.push(match object.as_ref() {
            Object::Warp {
                id,
                target,
                columns,
                rows,
                pivots,
                positions,
                opacities,
            } => {
                let pivots = pivot_table(pivots, registry)?;
                let x_cells = *rows;
                let y_cells = *columns;
                let grid_points = (x_cells + 1)
                    .checked_mul(y_cells + 1)
                    .ok_or(Moc2Error::InvalidData("warp grid is too large"))?;
                let points = pivot_values(positions, pivots.pivot_count(), grid_points * 2)?;
                let opacities = match opacities {
                    Some(values) => {
                        let values = float_array(values)?;
                        if values.len() != pivots.pivot_count() {
                            return Err(Moc2Error::InvalidData("warp opacity count mismatch"));
                        }
                        values.to_vec()
                    }
                    None => Vec::new(),
                };
                ParsedDeformer {
                    id: id.clone().unwrap_or_default(),
                    target: target.clone(),
                    pivots,
                    kind: Moc2DeformerKind::Warp {
                        x_cells,
                        y_cells,
                        points,
                    },
                    opacities,
                }
            }
            Object::Rotation {
                id,
                target,
                pivots,
                affines,
                opacities,
            } => {
                let pivots = pivot_table(pivots, registry)?;
                let affines = object_array(affines)?
                    .iter()
                    .map(|item| match item.as_ref() {
                        Object::Affine {
                            x,
                            y,
                            sx,
                            sy,
                            angle,
                            rx,
                            ry,
                        } => Ok(Moc2Affine {
                            origin_x: *x,
                            origin_y: *y,
                            scale_x: *sx,
                            scale_y: *sy,
                            rotation_degrees: *angle,
                            reflect_x: *rx,
                            reflect_y: *ry,
                        }),
                        _ => Err(Moc2Error::InvalidData("invalid affine entry")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if affines.len() != pivots.pivot_count() {
                    return Err(Moc2Error::InvalidData("affine keyform count mismatch"));
                }
                let opacities = match opacities {
                    Some(values) => {
                        let values = float_array(values)?;
                        if values.len() != pivots.pivot_count() {
                            return Err(Moc2Error::InvalidData("rotation opacity count mismatch"));
                        }
                        values.to_vec()
                    }
                    None => Vec::new(),
                };
                ParsedDeformer {
                    id: id.clone().unwrap_or_default(),
                    target: target.clone(),
                    pivots,
                    kind: Moc2DeformerKind::Rotation { affines },
                    opacities,
                }
            }
            _ => return Err(Moc2Error::InvalidData("unknown deformer type")),
        });
    }

    // Cubism 2 evaluates deformers parent-first. Order the parsed deformers so a
    // single forward pass can resolve every parent.
    let mut ordered = Vec::with_capacity(parsed.len());
    let mut placed: HashMap<String, usize> = HashMap::new();
    let mut remaining: Vec<Option<ParsedDeformer>> = parsed.into_iter().map(Some).collect();
    loop {
        let mut progress = false;
        for slot in &mut remaining {
            let Some(deformer) = slot.as_ref() else {
                continue;
            };
            let ready = match deformer.target.as_deref() {
                None | Some(BASE_TARGET) => true,
                Some(target) => placed.contains_key(target),
            };
            if !ready {
                continue;
            }
            let target = match deformer.target.as_deref() {
                None | Some(BASE_TARGET) => None,
                Some(target) => Some(
                    *placed
                        .get(target)
                        .ok_or(Moc2Error::InvalidData("missing parent deformer"))?,
                ),
            };
            let deformer = slot.take().expect("checked above");
            let index = ordered.len();
            if !deformer.id.is_empty() {
                placed.insert(deformer.id.clone(), index);
            }
            ordered.push(Moc2Deformer {
                id: deformer.id,
                target,
                pivots: deformer.pivots,
                opacities: deformer.opacities,
                kind: deformer.kind,
            });
            progress = true;
        }
        if !progress {
            break;
        }
    }
    if remaining.iter().any(Option::is_some) {
        return Err(Moc2Error::InvalidData(
            "deformer hierarchy is cyclic or references a missing parent",
        ));
    }

    Ok(ordered)
}

struct ParsedDeformer {
    id: String,
    target: Option<String>,
    pivots: Moc2PivotTable,
    kind: Moc2DeformerKind,
    opacities: Vec<f32>,
}

fn convert_drawable(
    object: &Ref,
    part_index: usize,
    deformer_index: &HashMap<String, usize>,
    registry: &mut ParameterRegistry,
) -> Result<(Moc2Drawable, Vec<String>), Moc2Error> {
    let Object::Mesh {
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
    } = object.as_ref()
    else {
        return Err(Moc2Error::InvalidData("drawable list contains a non-mesh"));
    };
    if flags & 1 != 0 {
        return Err(Moc2Error::UnsupportedTextureOption(*flags));
    }
    if *point_count == 0 || *polygon_count == 0 {
        return Err(Moc2Error::InvalidData("empty drawable mesh"));
    }

    let pivots = pivot_table(pivots, registry)?;
    let pivot_count = pivots.pivot_count();
    let positions = pivot_values(positions, pivot_count, point_count * 2)?;
    let uvs = float_array(uvs)?.to_vec();
    let raw_indices = integer_array(indices)?;
    if uvs.len() != point_count * 2 || raw_indices.len() != polygon_count * 3 {
        return Err(Moc2Error::InvalidData(
            "mesh array lengths do not match counts",
        ));
    }
    let indices = raw_indices
        .iter()
        .map(|index| {
            u16::try_from(*index)
                .ok()
                .filter(|index| usize::from(*index) < *point_count)
                .ok_or(Moc2Error::InvalidData("mesh index is out of bounds"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let draw_orders = integer_array(draw_orders)?.to_vec();
    let opacities = float_array(opacities)?.to_vec();
    if draw_orders.len() != pivot_count || opacities.len() != pivot_count {
        return Err(Moc2Error::InvalidData(
            "drawable keyform counts do not match the pivot table",
        ));
    }

    let target = match target.as_deref() {
        None | Some(BASE_TARGET) => None,
        Some(target) => Some(
            *deformer_index
                .get(target)
                .ok_or(Moc2Error::InvalidData("missing drawable deformer"))?,
        ),
    };

    let mut blend_flags = 0u8;
    match (flags & 30) >> 1 {
        1 => blend_flags = 4, // Screen composition uses a dedicated blend state.
        2 => blend_flags = 2, // Multiplicative composition.
        _ => {}
    }
    let culling = flags & 32 == 0;

    let clips = clips
        .as_deref()
        .map(|ids| {
            ids.split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    Ok((
        Moc2Drawable {
            id: id.clone().unwrap_or_default(),
            target,
            texture: (*texture).max(0),
            blend_flags,
            culling,
            point_count: *point_count,
            indices,
            uvs,
            pivots,
            positions,
            draw_orders,
            opacities,
            clips: Vec::new(),
            part_index,
        },
        clips,
    ))
}

/// Maps clip ids to drawable indices.
///
/// Cubism 2 drops the whole clip context when any referenced drawable is
/// missing, so the drawable is rendered without a mask.
fn resolve_clips(clips: &[String], drawable_index: &HashMap<String, usize>) -> Vec<usize> {
    if clips.is_empty() {
        return Vec::new();
    }
    let mut resolved = Vec::with_capacity(clips.len());
    for id in clips {
        match drawable_index.get(id) {
            Some(index) => resolved.push(*index),
            None => return Vec::new(),
        }
    }
    resolved
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

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
    bit_offset: u8,
    current_byte: u8,
    version: u8,
    objects: Vec<Option<Ref>>,
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
        if kind == 33 {
            let index = self.read_i32()?;
            return usize::try_from(index)
                .ok()
                .and_then(|index| self.objects.get(index))
                .cloned()
                .ok_or(Moc2Error::InvalidReference(index));
        }
        if self.objects.len() >= MAX_OBJECTS {
            return Err(Moc2Error::InvalidData("object count exceeds parser limits"));
        }
        if kind == 0 {
            // Cubism 2 numbers every non-reference object, including null.
            // Dropping this slot shifts later references onto unrelated IDs.
            self.objects.push(None);
            return Ok(None);
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
        self.objects.push(Some(Arc::clone(&object)));
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
        let visible = self.read_bit()?;
        let id = self.read_optional_text()?;
        let deformers = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null part deformer list"))?;
        let drawables = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null part drawable list"))?;
        Ok(Object::Part {
            id,
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
        let clips = if self.version >= CLIP_ID_VERSION {
            self.read_optional_clip()?
        } else {
            None
        };
        let texture = self.read_i32()?;
        let point_count = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|count| *count <= MAX_ARRAY_ITEMS)
            .ok_or(Moc2Error::InvalidData("invalid point count"))?;
        let polygon_count = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|count| *count <= MAX_ARRAY_ITEMS / 3)
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
        let flags = if self.version >= TEXTURE_OPTION_VERSION {
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
            .filter(|count| *count <= MAX_GRID_CELLS)
            .ok_or(Moc2Error::InvalidData("invalid warp columns"))?;
        let rows = usize::try_from(self.read_i32()?)
            .ok()
            .filter(|count| *count <= MAX_GRID_CELLS)
            .ok_or(Moc2Error::InvalidData("invalid warp rows"))?;
        let pivots = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null warp pivots"))?;
        let positions = self
            .read_object()?
            .ok_or(Moc2Error::InvalidData("null warp positions"))?;
        let opacities = if self.version >= OPACITY_VERSION {
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
        let opacities = if self.version >= OPACITY_VERSION {
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
        let (rx, ry) = if self.version >= REFLECT_VERSION {
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
        let minimum = self.read_f32()?;
        let maximum = self.read_f32()?;
        let default = self.read_f32()?;
        let id = self.read_optional_text()?;
        Ok(Object::Parameter {
            id,
            minimum,
            maximum,
            default,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_objects_preserve_deformer_reference_indices() {
        // A null still occupies slot 1. The parent ID is slot 2, and the
        // following ID must not be mistaken for it (Mashiro school_summer-2023).
        let mut bytes = Vec::new();
        for id in [
            Some("B_ARM_R_c.01"),
            None,
            Some("B_ARM_R_c.02"),
            Some("B_ARM_R.18"),
        ] {
            match id {
                Some(id) => {
                    bytes.extend_from_slice(&[50, id.len() as u8]);
                    bytes.extend_from_slice(id.as_bytes());
                }
                None => bytes.push(0),
            }
        }
        bytes.push(33);
        bytes.extend_from_slice(&2i32.to_be_bytes());

        let mut reader = Reader::new(&bytes);
        reader.read_object().unwrap();
        assert!(reader.read_object().unwrap().is_none());
        let parent = reader.read_object().unwrap().unwrap();
        reader.read_object().unwrap();
        let reference = reader.read_object().unwrap().unwrap();
        assert!(Arc::ptr_eq(&parent, &reference));
        assert!(matches!(reference.as_ref(), Object::Text(id) if id == "B_ARM_R_c.02"));
    }

    #[test]
    fn references_to_null_do_not_allocate_another_slot() {
        let mut bytes = vec![0, 33];
        bytes.extend_from_slice(&0i32.to_be_bytes());
        bytes.extend_from_slice(&[50, 1, b'x', 33]);
        bytes.extend_from_slice(&1i32.to_be_bytes());

        let mut reader = Reader::new(&bytes);
        assert!(reader.read_object().unwrap().is_none());
        assert!(reader.read_object().unwrap().is_none());
        let id = reader.read_object().unwrap().unwrap();
        let reference = reader.read_object().unwrap().unwrap();
        assert!(Arc::ptr_eq(&id, &reference));
        assert_eq!(reader.objects.len(), 2);
    }
}
