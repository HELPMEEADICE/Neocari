//! Parsers for Cubism 2 sidecar JSON files.
//!
//! Cubism 2 models reference a settings file, expressions, pose data, and
//! physics data. These documents use a different schema than the Cubism 3
//! `*.json` sidecars, so they are modelled separately from [`crate::json`].

use std::collections::BTreeMap;

use serde::Deserialize;

use super::Moc2Error;

const SETTINGS_FORMAT: &str = "model.json";
const EXPRESSION_FORMAT: &str = "exp.json";
const POSE_FORMAT: &str = "pose.json";
const PHYSICS_FORMAT: &str = "physics.json";
const DEFAULT_FADE_MS: f32 = 1000.0;

fn invalid_json(format: &'static str, error: serde_json::Error) -> Moc2Error {
    Moc2Error::InvalidJson {
        format,
        message: error.to_string(),
    }
}

/// Parsed Cubism 2 model settings (`*.model.json`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Moc2Settings {
    name: Option<String>,
    #[serde(alias = "Model")]
    model: Option<String>,
    #[serde(alias = "Textures")]
    textures: Vec<String>,
    physics: Option<String>,
    pose: Option<String>,
    layout: BTreeMap<String, f32>,
    init_params: Vec<Moc2InitialParameter>,
    init_parts_visible: Vec<Moc2InitialPartVisible>,
    hit_areas: Vec<Moc2HitArea>,
    motions: BTreeMap<String, Vec<Moc2MotionReference>>,
    expressions: Vec<Moc2ExpressionReference>,
}

impl Moc2Settings {
    /// Parses a Cubism 2 model settings document.
    pub fn from_json_str(source: &str) -> Result<Self, Moc2Error> {
        serde_json::from_str(source).map_err(|error| invalid_json(SETTINGS_FORMAT, error))
    }

    /// Returns the optional model display name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the referenced `.moc` file, relative to the settings file.
    pub fn model_file(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Returns texture files, relative to the settings file.
    pub fn texture_files(&self) -> &[String] {
        &self.textures
    }

    /// Returns the optional physics file.
    pub fn physics_file(&self) -> Option<&str> {
        self.physics.as_deref()
    }

    /// Returns the optional pose file.
    pub fn pose_file(&self) -> Option<&str> {
        self.pose.as_deref()
    }

    /// Returns the optional layout overrides.
    pub fn layout(&self) -> &BTreeMap<String, f32> {
        &self.layout
    }

    /// Returns parameters applied when the model loads.
    pub fn init_params(&self) -> &[Moc2InitialParameter] {
        &self.init_params
    }

    /// Returns part opacities applied when the model loads.
    pub fn init_parts_visible(&self) -> &[Moc2InitialPartVisible] {
        &self.init_parts_visible
    }

    /// Returns named hit areas.
    pub fn hit_areas(&self) -> &[Moc2HitArea] {
        &self.hit_areas
    }

    /// Returns motion groups keyed by their Cubism 2 group name.
    pub fn motion_groups(&self) -> &BTreeMap<String, Vec<Moc2MotionReference>> {
        &self.motions
    }

    /// Returns the motions of one group.
    pub fn motions(&self, group: &str) -> Option<&[Moc2MotionReference]> {
        self.motions.get(group).map(Vec::as_slice)
    }

    /// Returns expression references declared by the model.
    pub fn expressions(&self) -> &[Moc2ExpressionReference] {
        &self.expressions
    }

    /// Looks up an expression file by expression name.
    pub fn expression_file(&self, name: &str) -> Option<&str> {
        self.expressions
            .iter()
            .find(|expression| expression.name == name)
            .map(|expression| expression.file.as_str())
    }
}

/// One parameter initialised when a Cubism 2 model loads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Moc2InitialParameter {
    id: String,
    value: f32,
}

impl Moc2InitialParameter {
    /// Returns the parameter id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the initial value.
    pub fn value(&self) -> f32 {
        self.value
    }
}

/// One part opacity initialised when a Cubism 2 model loads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Moc2InitialPartVisible {
    id: String,
    value: f32,
}

impl Moc2InitialPartVisible {
    /// Returns the part id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the initial opacity.
    pub fn value(&self) -> f32 {
        self.value
    }
}

/// A named hit area declared by a Cubism 2 model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Moc2HitArea {
    name: String,
    id: String,
}

impl Moc2HitArea {
    /// Returns the user-facing hit area name, such as `Head`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the drawable id used for hit testing.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// One motion file reference inside the settings document.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct Moc2MotionReference {
    file: Option<String>,
    fade_in: f32,
    fade_out: f32,
    sound: Option<String>,
}

impl Default for Moc2MotionReference {
    fn default() -> Self {
        Self {
            file: None,
            fade_in: DEFAULT_FADE_MS,
            fade_out: DEFAULT_FADE_MS,
            sound: None,
        }
    }
}

impl Moc2MotionReference {
    /// Returns the motion file, relative to the settings file.
    pub fn file(&self) -> Option<&str> {
        self.file.as_deref()
    }

    /// Returns the fade-in duration in milliseconds.
    pub fn fade_in_ms(&self) -> f32 {
        self.fade_in
    }

    /// Returns the fade-out duration in milliseconds.
    pub fn fade_out_ms(&self) -> f32 {
        self.fade_out
    }

    /// Returns the optional sound file associated with the motion.
    pub fn sound(&self) -> Option<&str> {
        self.sound.as_deref()
    }
}

/// One expression file reference inside the settings document.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Moc2ExpressionReference {
    name: String,
    file: String,
}

impl Moc2ExpressionReference {
    /// Returns the expression name used by the application.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the expression file, relative to the settings file.
    pub fn file(&self) -> &str {
        &self.file
    }
}

/// How an expression parameter is combined with the current value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Moc2ExpressionBlend {
    /// Add the expression value, scaled by the expression weight.
    Add,
    /// Multiply the current value, scaled by the expression weight.
    Multiply,
    /// Overwrite the current value, scaled by the expression weight.
    Overwrite,
}

/// One parameter entry of a Cubism 2 expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2ExpressionParameter {
    id: String,
    value: f32,
    blend: Moc2ExpressionBlend,
}

impl Moc2ExpressionParameter {
    /// Returns the target parameter id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the value after Cubism 2's default-value normalisation.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Returns how the value is combined with the model parameter.
    pub fn blend(&self) -> Moc2ExpressionBlend {
        self.blend
    }
}

/// Parsed Cubism 2 expression (`*.exp.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2Expression {
    fade_in_ms: f32,
    fade_out_ms: f32,
    parameters: Vec<Moc2ExpressionParameter>,
}

impl Moc2Expression {
    /// Parses an expression document.
    pub fn from_json_str(source: &str) -> Result<Self, Moc2Error> {
        let raw: RawExpression =
            serde_json::from_str(source).map_err(|error| invalid_json(EXPRESSION_FORMAT, error))?;

        let fade_in_ms = resolved_fade(raw.fade_in);
        let fade_out_ms = resolved_fade(raw.fade_out);
        let parameters = raw
            .params
            .iter()
            .map(|parameter| {
                let blend = match parameter.calc.as_deref() {
                    Some("mult") => Moc2ExpressionBlend::Multiply,
                    Some("set") => Moc2ExpressionBlend::Overwrite,
                    _ => Moc2ExpressionBlend::Add,
                };
                let value = match blend {
                    Moc2ExpressionBlend::Add => parameter.val - parameter.def.unwrap_or(0.0),
                    Moc2ExpressionBlend::Multiply => {
                        let default = parameter.def.unwrap_or(1.0);
                        parameter.val / if default == 0.0 { 1.0 } else { default }
                    }
                    Moc2ExpressionBlend::Overwrite => parameter.val,
                };
                Moc2ExpressionParameter {
                    id: parameter.id.clone(),
                    value,
                    blend,
                }
            })
            .collect();

        Ok(Self {
            fade_in_ms,
            fade_out_ms,
            parameters,
        })
    }

    /// Returns the fade-in duration in milliseconds.
    pub fn fade_in_ms(&self) -> f32 {
        self.fade_in_ms
    }

    /// Returns the fade-out duration in milliseconds.
    pub fn fade_out_ms(&self) -> f32 {
        self.fade_out_ms
    }

    /// Returns the expression parameters.
    pub fn parameters(&self) -> &[Moc2ExpressionParameter] {
        &self.parameters
    }
}

fn resolved_fade(value: Option<f32>) -> f32 {
    match value {
        Some(value) if value > 0.0 => value,
        _ => DEFAULT_FADE_MS,
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawExpression {
    fade_in: Option<f32>,
    fade_out: Option<f32>,
    params: Vec<RawExpressionParameter>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawExpressionParameter {
    id: String,
    val: f32,
    calc: Option<String>,
    def: Option<f32>,
}

/// One part entry of a Cubism 2 pose group.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2PosePart {
    id: String,
    links: Vec<String>,
}

impl Moc2PosePart {
    /// Returns the part id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns linked part ids that follow this part's opacity.
    pub fn links(&self) -> &[String] {
        &self.links
    }
}

/// Parsed Cubism 2 pose document (`*.pose.json`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Moc2Pose {
    groups: Vec<Vec<Moc2PosePart>>,
}

impl Moc2Pose {
    /// Parses a pose document.
    pub fn from_json_str(source: &str) -> Result<Self, Moc2Error> {
        let raw: RawPose =
            serde_json::from_str(source).map_err(|error| invalid_json(POSE_FORMAT, error))?;
        Ok(Self {
            groups: raw
                .parts_visible
                .into_iter()
                .map(|entry| {
                    entry
                        .group
                        .into_iter()
                        .map(|part| Moc2PosePart {
                            id: part.id,
                            links: part.link,
                        })
                        .collect()
                })
                .collect(),
        })
    }

    /// Returns pose groups. Parts in one group are mutually exclusive.
    pub fn groups(&self) -> &[Vec<Moc2PosePart>] {
        &self.groups
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPose {
    parts_visible: Vec<RawPoseEntry>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPoseEntry {
    group: Vec<RawPosePartEntry>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPosePartEntry {
    id: String,
    link: Vec<String>,
}

/// Parameter driven by a Cubism 2 physics hair chain.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Moc2PhysicsSourceKind {
    /// Translate the physics pivot horizontally.
    X,
    /// Translate the physics pivot vertically.
    Y,
    /// Rotate the gravity direction.
    Angle,
}

/// Parameter written by a Cubism 2 physics hair chain.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Moc2PhysicsTargetKind {
    /// Write the current chain angle.
    Angle,
    /// Write the chain angular velocity.
    AngleVelocity,
}

/// One input parameter of a physics hair chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2PhysicsSource {
    id: String,
    kind: Moc2PhysicsSourceKind,
    scale: f32,
    weight: f32,
}

impl Moc2PhysicsSource {
    /// Returns the driving parameter id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns how the parameter drives the chain.
    pub fn kind(&self) -> Moc2PhysicsSourceKind {
        self.kind
    }

    /// Returns the input scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the input blend weight.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

/// One output parameter of a physics hair chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2PhysicsTarget {
    id: String,
    kind: Moc2PhysicsTargetKind,
    scale: f32,
    weight: f32,
}

impl Moc2PhysicsTarget {
    /// Returns the driven parameter id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns which chain value is written.
    pub fn kind(&self) -> Moc2PhysicsTargetKind {
        self.kind
    }

    /// Returns the output scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the output blend weight.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

/// One Cubism 2 physics hair chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Moc2PhysicsHair {
    label: Option<String>,
    length: f32,
    resistance: f32,
    mass: f32,
    sources: Vec<Moc2PhysicsSource>,
    targets: Vec<Moc2PhysicsTarget>,
}

impl Moc2PhysicsHair {
    /// Returns the optional chain label from the model file.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns the chain length.
    pub fn length(&self) -> f32 {
        self.length
    }

    /// Returns the chain resistance.
    pub fn resistance(&self) -> f32 {
        self.resistance
    }

    /// Returns the chain mass.
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// Returns the input parameters.
    pub fn sources(&self) -> &[Moc2PhysicsSource] {
        &self.sources
    }

    /// Returns the output parameters.
    pub fn targets(&self) -> &[Moc2PhysicsTarget] {
        &self.targets
    }
}

/// Parsed Cubism 2 physics document (`*.physics` / `*.physics.json`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Moc2Physics {
    hairs: Vec<Moc2PhysicsHair>,
}

impl Moc2Physics {
    /// Parses a physics document.
    pub fn from_json_str(source: &str) -> Result<Self, Moc2Error> {
        let raw: RawPhysics =
            serde_json::from_str(source).map_err(|error| invalid_json(PHYSICS_FORMAT, error))?;
        let mut hairs = Vec::with_capacity(raw.physics_hair.len());
        for hair in raw.physics_hair {
            let setup = hair.setup.unwrap_or_default();
            let mut sources = Vec::with_capacity(hair.src.len());
            for source in hair.src {
                let kind = match source.ptype.as_str() {
                    "x" => Moc2PhysicsSourceKind::X,
                    "y" => Moc2PhysicsSourceKind::Y,
                    "angle" => Moc2PhysicsSourceKind::Angle,
                    other => {
                        return Err(Moc2Error::InvalidJson {
                            format: PHYSICS_FORMAT,
                            message: format!("unsupported physics source type {other}"),
                        });
                    }
                };
                sources.push(Moc2PhysicsSource {
                    id: source.id,
                    kind,
                    scale: source.scale,
                    weight: source.weight,
                });
            }
            let mut targets = Vec::with_capacity(hair.targets.len());
            for target in hair.targets {
                let kind = match target.ptype.as_str() {
                    "angle" => Moc2PhysicsTargetKind::Angle,
                    "angle_v" => Moc2PhysicsTargetKind::AngleVelocity,
                    other => {
                        return Err(Moc2Error::InvalidJson {
                            format: PHYSICS_FORMAT,
                            message: format!("unsupported physics target type {other}"),
                        });
                    }
                };
                targets.push(Moc2PhysicsTarget {
                    id: target.id,
                    kind,
                    scale: target.scale,
                    weight: target.weight,
                });
            }
            hairs.push(Moc2PhysicsHair {
                label: hair.label,
                length: setup.length,
                resistance: setup.regist,
                mass: setup.mass,
                sources,
                targets,
            });
        }
        Ok(Self { hairs })
    }

    /// Returns all parsed hair chains.
    pub fn hairs(&self) -> &[Moc2PhysicsHair] {
        &self.hairs
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPhysics {
    physics_hair: Vec<RawPhysicsHair>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPhysicsHair {
    label: Option<String>,
    setup: Option<RawPhysicsSetup>,
    src: Vec<RawPhysicsSource>,
    targets: Vec<RawPhysicsTarget>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPhysicsSetup {
    length: f32,
    #[serde(alias = "resist")]
    regist: f32,
    mass: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPhysicsSource {
    id: String,
    ptype: String,
    scale: f32,
    weight: f32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPhysicsTarget {
    id: String,
    ptype: String,
    scale: f32,
    weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_parse_layout_motions_and_expressions() {
        let source = r#"{
            "name": "kasumi",
            "model": "live2d/kasumi.moc",
            "textures": ["live2d/texture_00.png"],
            "physics": "live2d/kasumi.physics",
            "layout": { "width": 2.0, "center_x": 0.1 },
            "init_params": [{ "id": "PARAM_ANGLE_X", "value": 5 }],
            "init_parts_visible": [{ "id": "PARTS_01", "value": 0 }],
            "hit_areas": [{ "name": "Head", "id": "D_HEAD" }],
            "motions": {
                "idle": [{ "file": "live2d/idle01.mtn", "fade_in": 200, "fade_out": 300 }]
            },
            "expressions": [{ "name": "smile", "file": "live2d/smile.exp" }]
        }"#;
        let settings = Moc2Settings::from_json_str(source).unwrap();

        assert_eq!(settings.model_file(), Some("live2d/kasumi.moc"));
        assert_eq!(settings.layout().get("center_x"), Some(&0.1));
        assert_eq!(settings.init_params()[0].value(), 5.0);
        assert_eq!(settings.init_parts_visible()[0].value(), 0.0);
        assert_eq!(settings.hit_areas()[0].name(), "Head");
        let motion = &settings.motions("idle").unwrap()[0];
        assert_eq!(motion.file(), Some("live2d/idle01.mtn"));
        assert_eq!(motion.fade_in_ms(), 200.0);
        assert_eq!(settings.expression_file("smile"), Some("live2d/smile.exp"));
    }

    #[test]
    fn expression_normalises_default_values() {
        let source = r#"{
            "fade_in": 0,
            "fade_out": 250,
            "params": [
                { "id": "PARAM_EYE_L_OPEN", "val": 0, "calc": "mult" },
                { "id": "PARAM_BROW_L_Y", "val": 0.4, "def": 0.2 },
                { "id": "PARAM_MOUTH_FORM", "val": 1, "calc": "set" }
            ]
        }"#;
        let expression = Moc2Expression::from_json_str(source).unwrap();

        assert_eq!(expression.fade_in_ms(), 1000.0);
        assert_eq!(expression.fade_out_ms(), 250.0);
        assert_eq!(expression.parameters()[0].value(), 0.0);
        assert_eq!(
            expression.parameters()[0].blend(),
            Moc2ExpressionBlend::Multiply
        );
        assert!((expression.parameters()[1].value() - 0.2).abs() < 1e-6);
        assert_eq!(
            expression.parameters()[2].blend(),
            Moc2ExpressionBlend::Overwrite
        );
    }

    #[test]
    fn pose_groups_and_links_are_parsed() {
        let source = r#"{
            "type": "Live2D Pose",
            "parts_visible": [
                { "group": [
                    { "id": "PARTS_01", "link": ["PARTS_02"] },
                    { "id": "PARTS_03" }
                ] }
            ]
        }"#;
        let pose = Moc2Pose::from_json_str(source).unwrap();

        assert_eq!(pose.groups().len(), 1);
        assert_eq!(pose.groups()[0][0].links(), ["PARTS_02"]);
        assert_eq!(pose.groups()[0][1].id(), "PARTS_03");
    }

    #[test]
    fn physics_parses_hair_chains() {
        let source = r#"{
            "physics_hair": [{
                "label": "skirt",
                "setup": { "length": 0.4, "regist": 1, "mass": 0.4 },
                "src": [
                    { "id": "PARAM_BODY_ANGLE_X", "ptype": "x", "scale": 0.005, "weight": 1 },
                    { "id": "PARAM_BODY_ANGLE_Z", "ptype": "angle", "scale": 0.8, "weight": 1 }
                ],
                "targets": [
                    { "id": "PARAM_CLOTHES_B", "ptype": "angle", "scale": 0.025, "weight": 1 }
                ]
            }]
        }"#;
        let physics = Moc2Physics::from_json_str(source).unwrap();

        let hair = &physics.hairs()[0];
        assert_eq!(hair.label(), Some("skirt"));
        assert_eq!(hair.length(), 0.4);
        assert_eq!(hair.resistance(), 1.0);
        assert_eq!(hair.sources()[1].kind(), Moc2PhysicsSourceKind::Angle);
        assert_eq!(hair.targets()[0].kind(), Moc2PhysicsTargetKind::Angle);
    }
}
