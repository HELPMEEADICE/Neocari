//! Convenience loaders for complete Cubism model folders.
//!
//! These helpers start from a `.model3.json` path and resolve the files it
//! references relative to that model file. Use [`crate::assets::load_model_runtime`]
//! when the application needs per-frame parameter, motion, or expression updates.
//! Use [`crate::assets::load_model`] for a static default-pose snapshot.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

use crate::{
    core::Matrix44,
    json::{Model3, Physics3, Pose3},
    moc2::{
        Moc2Error, Moc2Expression, Moc2Model, Moc2Motion, Moc2Physics, Moc2Pose, Moc2Runtime,
        Moc2Settings,
    },
    moc3::{
        Moc3ArtMeshKeyforms, Moc3ArtMeshes, Moc3CanvasInfo, Moc3Deformers, Moc3DrawOrderGroups,
        Moc3DrawableMesh, Moc3Glues, Moc3Ids, Moc3KeyformBindings, Moc3OffscreenInfo, Moc3Parts,
        build_moc3_drawable_meshes_for_default_pose_with_offscreen_state,
    },
    runtime::ModelRuntime,
};

#[derive(Debug, Clone)]
/// A model loaded in its default pose.
///
/// This type is useful for import tools, previews, tests, or renderers that only
/// need the initial mesh data. For animation and interactive parameter changes,
/// use [`RuntimeModel`] instead.
pub struct DefaultModel {
    model: Model3,
    canvas: Moc3CanvasInfo,
    meshes: Vec<Moc3DrawableMesh>,
    textures: Vec<DecodedTexture>,
}

impl DefaultModel {
    /// Returns the parsed `.model3.json` data.
    pub fn model(&self) -> &Model3 {
        &self.model
    }

    /// Returns the model canvas information parsed from the `.moc3` file.
    pub fn canvas(&self) -> Moc3CanvasInfo {
        self.canvas
    }

    /// Returns drawable meshes built from the model's default parameter values.
    pub fn meshes(&self) -> &[Moc3DrawableMesh] {
        &self.meshes
    }

    /// Returns decoded RGBA textures in the order referenced by the model file.
    pub fn textures(&self) -> &[DecodedTexture] {
        &self.textures
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A PNG texture decoded into tightly packed RGBA8 pixels.
///
/// The pixel buffer is arranged row-major with four bytes per pixel:
/// red, green, blue, alpha.
pub struct DecodedTexture {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl DecodedTexture {
    /// Creates a decoded texture from raw RGBA8 data.
    ///
    /// The constructor does not validate that `rgba.len() == width * height * 4`;
    /// callers that build textures manually should keep those values consistent.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Self {
            width,
            height,
            rgba,
        }
    }

    /// Returns the texture width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the raw RGBA8 pixel data.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur while loading a complete model from disk.
pub enum AssetLoadError {
    /// A referenced file could not be read.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Path of the file that failed to load.
        path: String,
        /// Original I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A Cubism JSON file was invalid or unsupported.
    #[error("failed to parse model json: {0}")]
    Json(#[source] crate::Error),
    /// The referenced `.moc3` file was invalid or unsupported.
    #[error("failed to parse moc3: {0}")]
    Moc3(#[source] crate::Error),
    /// A Cubism 2 model settings JSON file was malformed.
    #[error("failed to parse model2 json: {message}")]
    Moc2Json { message: String },
    /// A Cubism 2 sidecar file could not be parsed.
    #[error("failed to parse Cubism 2 {format}: {source}")]
    Moc2File {
        /// Human-readable format name, such as `mtn` or `exp`.
        format: &'static str,
        /// Original parse error.
        #[source]
        source: Moc2Error,
    },
    /// The referenced MOC file was invalid or unsupported.
    #[error("failed to parse moc2: {0}")]
    Moc2(#[source] Moc2Error),
    /// A referenced texture could not be decoded.
    #[error("failed to decode {path}: {source}")]
    Image {
        /// Path of the image that failed to decode.
        path: String,
        /// Original image decoding error.
        #[source]
        source: image::ImageError,
    },
    /// The supplied model path has no parent directory for resolving assets.
    #[error("path has no parent: {path}")]
    MissingParent {
        /// Path that did not have a parent directory.
        path: String,
    },
    /// Drawable meshes could not be built from the parsed model data.
    #[error("failed to build drawable meshes")]
    DrawableMeshes,
}

#[derive(Debug, Clone)]
/// A Cubism 2 model with its decoded textures and default-pose meshes.
pub struct LoadedMoc2Model {
    model: Moc2Model,
    settings: Moc2Settings,
    textures: Vec<DecodedTexture>,
    model_dir: Option<PathBuf>,
}

impl LoadedMoc2Model {
    /// Returns parsed MOC2 data and renderer-ready drawables.
    pub fn model(&self) -> &Moc2Model {
        &self.model
    }
    /// Returns the parsed `.model.json` settings.
    pub fn settings(&self) -> &Moc2Settings {
        &self.settings
    }
    /// Returns renderer-ready drawables in model order.
    pub fn drawables(&self) -> &[Moc3DrawableMesh] {
        self.model.drawables()
    }
    /// Alias matching the naming used by the MOC3 loader.
    pub fn meshes(&self) -> &[Moc3DrawableMesh] {
        self.model.drawables()
    }
    /// Returns decoded RGBA8 textures in manifest order.
    pub fn textures(&self) -> &[DecodedTexture] {
        &self.textures
    }
    /// Returns the folder containing the model settings JSON.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }
}

#[derive(Debug)]
/// A Cubism 2 model loaded for animation.
///
/// The runtime model owns the decoded textures, the parsed settings, the
/// animation runtime, and a cache of loaded motions and expressions. Use
/// [`play_motion`](Self::play_motion) and [`play_expression`](Self::play_expression)
/// to start animation, then call [`update`](Self::update) every frame.
pub struct Moc2RuntimeModel {
    runtime: Moc2Runtime,
    settings: Moc2Settings,
    textures: Vec<DecodedTexture>,
    model_dir: Option<PathBuf>,
    motions: HashMap<String, Arc<Moc2Motion>>,
    expressions: HashMap<String, Arc<Moc2Expression>>,
}

impl Moc2RuntimeModel {
    /// Returns the immutable runtime state.
    pub fn runtime(&self) -> &Moc2Runtime {
        &self.runtime
    }

    /// Returns the mutable runtime state.
    pub fn runtime_mut(&mut self) -> &mut Moc2Runtime {
        &mut self.runtime
    }

    /// Returns the parsed `.model.json` settings.
    pub fn settings(&self) -> &Moc2Settings {
        &self.settings
    }

    /// Returns decoded RGBA8 textures in manifest order.
    pub fn textures(&self) -> &[DecodedTexture] {
        &self.textures
    }

    /// Transfers decoded pixels to the caller after GPU upload. Animation does
    /// not use them; subsequent calls to `textures()` return an empty slice.
    pub fn take_textures(&mut self) -> Vec<DecodedTexture> {
        std::mem::take(&mut self.textures)
    }

    /// Returns the folder containing the model settings JSON.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Returns the model-to-clip-space matrix declared by the model layout.
    pub fn model_matrix(&self) -> Matrix44 {
        self.runtime.model_matrix()
    }

    /// Advances the model by `delta_seconds`.
    pub fn update(&mut self, delta_seconds: f32) -> Option<()> {
        self.runtime.update(delta_seconds)
    }

    /// Returns the drawable meshes in model order.
    pub fn meshes(&self) -> &[Moc3DrawableMesh] {
        self.runtime.meshes()
    }

    /// Starts one of the motions declared by the settings file.
    ///
    /// Motions are cached by group and index. Returns `Ok(false)` when the
    /// reference does not name a motion file.
    pub fn play_motion(&mut self, group: &str, index: usize) -> Result<bool, AssetLoadError> {
        let reference = self
            .settings
            .motions(group)
            .and_then(|motions| motions.get(index))
            .ok_or_else(|| AssetLoadError::Moc2Json {
                message: format!("motion group '{group}' has no entry {index}"),
            })?;
        let Some(file) = reference.file() else {
            return Ok(false);
        };
        let key = format!("{group}#{index}");

        let motion = match self.motions.get(&key) {
            Some(motion) => Arc::clone(motion),
            None => {
                let path = self.resolve(file);
                let bytes = read_bytes(&path)?;
                let motion = Arc::new(Moc2Motion::from_bytes(&bytes).map_err(|source| {
                    AssetLoadError::Moc2File {
                        format: "mtn",
                        source,
                    }
                })?);
                self.motions.insert(key, Arc::clone(&motion));
                motion
            }
        };

        self.runtime
            .start_motion(motion, reference.fade_in_ms(), reference.fade_out_ms());
        Ok(true)
    }

    /// Starts the first motion of a group.
    pub fn play_motion_group(&mut self, group: &str) -> Result<bool, AssetLoadError> {
        self.play_motion(group, 0)
    }

    /// Returns the motion group names declared by the settings file.
    pub fn motion_groups(&self) -> impl Iterator<Item = &str> {
        self.settings.motion_groups().keys().map(String::as_str)
    }

    /// Starts an expression declared by the settings file.
    ///
    /// Expressions are cached by name. Returns `Ok(false)` when no expression
    /// with that name exists.
    pub fn play_expression(&mut self, name: &str) -> Result<bool, AssetLoadError> {
        let Some(file) = self.settings.expression_file(name) else {
            return Ok(false);
        };

        let expression = match self.expressions.get(name) {
            Some(expression) => Arc::clone(expression),
            None => {
                let path = self.resolve(file);
                let source = read_text(&path)?;
                let expression =
                    Arc::new(Moc2Expression::from_json_str(&source).map_err(|source| {
                        AssetLoadError::Moc2File {
                            format: "exp",
                            source,
                        }
                    })?);
                self.expressions
                    .insert(name.to_owned(), Arc::clone(&expression));
                expression
            }
        };

        self.runtime.start_expression(expression);
        Ok(true)
    }

    /// Returns the expression names declared by the settings file.
    pub fn expression_names(&self) -> impl Iterator<Item = &str> {
        self.settings
            .expressions()
            .iter()
            .map(crate::moc2::Moc2ExpressionReference::name)
    }

    /// Returns the hit area name containing a model-space point.
    ///
    /// When the drawable is not referenced by a named hit area the drawable id is
    /// returned instead.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<&str> {
        let id = self.runtime.hit_test(x, y)?;
        Some(
            self.settings
                .hit_areas()
                .iter()
                .find(|area| area.id() == id)
                .map_or(id, |area| area.name()),
        )
    }

    fn resolve(&self, file: &str) -> PathBuf {
        let mut path = self.model_dir.clone().unwrap_or_default();
        for component in file.replace('\\', "/").split('/') {
            if !component.is_empty() {
                path.push(component);
            }
        }
        path
    }
}

fn read_moc2_settings(path: &Path) -> Result<Moc2Settings, AssetLoadError> {
    let source = read_text(path)?;
    Moc2Settings::from_json_str(&source).map_err(|source| AssetLoadError::Moc2File {
        format: "model.json",
        source,
    })
}

/// Loads a Cubism 2 model settings JSON, its referenced MOC file, and textures.
///
/// The generated drawables use the model default parameter values and can be
/// uploaded through the existing wgpu renderer.
pub fn load_moc2_model(path: impl AsRef<Path>) -> Result<LoadedMoc2Model, AssetLoadError> {
    let path = path.as_ref();
    let model_dir = path.parent().ok_or_else(|| AssetLoadError::MissingParent {
        path: path.display().to_string(),
    })?;
    let settings = read_moc2_settings(path)?;
    let model_file = settings
        .model_file()
        .ok_or_else(|| AssetLoadError::Moc2Json {
            message: "model settings have no model file".to_owned(),
        })?;
    let moc_bytes = read_bytes(&model_dir.join(model_file.replace('\\', "/")))?;
    let model = Moc2Model::from_bytes(&moc_bytes).map_err(AssetLoadError::Moc2)?;
    let textures = decode_textures(model_dir, settings.texture_files())?;
    Ok(LoadedMoc2Model {
        model,
        settings,
        textures,
        model_dir: Some(model_dir.to_path_buf()),
    })
}

/// Loads a Cubism 2 model with its physics, pose, and animation runtime.
///
/// The loader applies `init_params`, `init_parts_visible`, physics, and pose
/// data declared by the settings file, then stores the resulting state as the
/// animation base. Motions and expressions are loaded lazily by
/// [`Moc2RuntimeModel::play_motion`] and [`Moc2RuntimeModel::play_expression`].
pub fn load_moc2_runtime(path: impl AsRef<Path>) -> Result<Moc2RuntimeModel, AssetLoadError> {
    let path = path.as_ref();
    let model_dir = path.parent().ok_or_else(|| AssetLoadError::MissingParent {
        path: path.display().to_string(),
    })?;
    let settings = read_moc2_settings(path)?;
    let model_file = settings
        .model_file()
        .ok_or_else(|| AssetLoadError::Moc2Json {
            message: "model settings have no model file".to_owned(),
        })?;
    let moc_bytes = read_bytes(&model_dir.join(model_file.replace('\\', "/")))?;
    let model = Moc2Model::from_bytes(&moc_bytes).map_err(AssetLoadError::Moc2)?;

    let mut runtime = Moc2Runtime::new(model.into_scene()).ok_or(AssetLoadError::DrawableMeshes)?;

    for parameter in settings.init_params() {
        runtime.set_parameter(parameter.id(), parameter.value());
    }
    for part in settings.init_parts_visible() {
        runtime.set_part_opacity(part.id(), part.value());
    }

    if let Some(physics_file) = settings.physics_file() {
        let source = read_text(&model_dir.join(physics_file.replace('\\', "/")))?;
        let physics =
            Moc2Physics::from_json_str(&source).map_err(|source| AssetLoadError::Moc2File {
                format: "physics",
                source,
            })?;
        runtime.set_physics(physics);
    }

    if let Some(pose_file) = settings.pose_file() {
        let source = read_text(&model_dir.join(pose_file.replace('\\', "/")))?;
        let pose = Moc2Pose::from_json_str(&source).map_err(|source| AssetLoadError::Moc2File {
            format: "pose",
            source,
        })?;
        runtime.set_pose(pose);
    }

    runtime.set_layout(settings.layout().clone());
    runtime.save_parameters();

    let textures = decode_textures(model_dir, settings.texture_files())?;

    Ok(Moc2RuntimeModel {
        runtime,
        settings,
        textures,
        model_dir: Some(model_dir.to_path_buf()),
        motions: HashMap::new(),
        expressions: HashMap::new(),
    })
}

/// Loads a model as a static default-pose snapshot.
///
/// The `path` should point to a `.model3.json` file. Neocari reads the referenced
/// `.moc3` file and textures from the same model directory, then builds drawable
/// meshes using the model's default parameter values.
pub fn load_model(path: impl AsRef<Path>) -> Result<DefaultModel, AssetLoadError> {
    parse_model(path)?.into_default_model()
}

/// Loads a model into a mutable runtime.
///
/// This is the main entry point for interactive applications. The returned
/// [`RuntimeModel`] keeps decoded textures next to a [`ModelRuntime`], so a render
/// loop can update parameters, apply motions and expressions, call
/// [`ModelRuntime::update_meshes`], and draw the resulting meshes.
pub fn load_model_runtime(path: impl AsRef<Path>) -> Result<RuntimeModel, AssetLoadError> {
    let path = path.as_ref();
    let model_dir = path.parent().map(Path::to_path_buf);
    parse_model(path)?.into_runtime_model(model_dir)
}

#[derive(Debug, Clone)]
/// A loaded model with mutable runtime state and decoded textures.
pub struct RuntimeModel {
    runtime: ModelRuntime,
    textures: Vec<DecodedTexture>,
    model_dir: Option<PathBuf>,
}

impl RuntimeModel {
    /// Returns the immutable runtime state.
    pub fn runtime(&self) -> &ModelRuntime {
        &self.runtime
    }

    /// Returns the mutable runtime state.
    ///
    /// Use this in a frame loop to edit parameters, apply motions or expressions,
    /// and rebuild drawable meshes.
    pub fn runtime_mut(&mut self) -> &mut ModelRuntime {
        &mut self.runtime
    }

    /// Returns decoded textures in the order used by drawable texture indices.
    pub fn textures(&self) -> &[DecodedTexture] {
        &self.textures
    }

    /// Transfers decoded pixels to the caller after GPU upload. Animation does
    /// not use them; subsequent calls to `textures()` return an empty slice.
    pub fn take_textures(&mut self) -> Vec<DecodedTexture> {
        std::mem::take(&mut self.textures)
    }

    /// Returns the directory that contained the loaded `.model3.json` file.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }
}

struct ParsedModel {
    model: Model3,
    canvas: Moc3CanvasInfo,
    art_meshes: Moc3ArtMeshes,
    art_mesh_keyforms: Moc3ArtMeshKeyforms,
    deformers: Moc3Deformers,
    bindings: Moc3KeyformBindings,
    ids: Moc3Ids,
    offscreen: Moc3OffscreenInfo,
    glues: Moc3Glues,
    parts: Moc3Parts,
    draw_order_groups: Option<Moc3DrawOrderGroups>,
    physics: Option<Physics3>,
    pose: Option<Pose3>,
    textures: Vec<DecodedTexture>,
}

impl ParsedModel {
    fn into_default_model(self) -> Result<DefaultModel, AssetLoadError> {
        let mut meshes = build_moc3_drawable_meshes_for_default_pose_with_offscreen_state(
            &self.art_meshes,
            &self.art_mesh_keyforms,
            &self.deformers,
            &self.bindings,
            &self.ids,
            &self.offscreen,
        )
        .ok_or(AssetLoadError::DrawableMeshes)?;
        self.glues
            .apply(
                &mut meshes,
                &self.bindings,
                self.bindings.parameter_default_values(),
            )
            .ok_or(AssetLoadError::DrawableMeshes)?;

        Ok(DefaultModel {
            model: self.model,
            canvas: self.canvas,
            meshes,
            textures: self.textures,
        })
    }

    fn into_runtime_model(
        self,
        model_dir: Option<PathBuf>,
    ) -> Result<RuntimeModel, AssetLoadError> {
        let mut runtime = ModelRuntime::new(
            self.model,
            self.canvas,
            self.art_meshes,
            self.art_mesh_keyforms,
            self.deformers,
            self.bindings,
            self.ids,
            self.offscreen,
            self.glues,
            self.parts,
            self.draw_order_groups,
            self.pose,
        )
        .ok_or(AssetLoadError::DrawableMeshes)?;
        if let Some(physics) = self.physics {
            runtime.set_physics(physics);
        }

        Ok(RuntimeModel {
            runtime,
            textures: self.textures,
            model_dir,
        })
    }
}

fn parse_model(path: impl AsRef<Path>) -> Result<ParsedModel, AssetLoadError> {
    let path = path.as_ref();
    let model_source = read_text(path)?;
    let model = Model3::from_json_str(&model_source).map_err(AssetLoadError::Json)?;
    let model_dir = path.parent().ok_or_else(|| AssetLoadError::MissingParent {
        path: path.display().to_string(),
    })?;
    let moc_path = model_dir.join(model.moc());
    let moc = read_bytes(&moc_path)?;

    let art_meshes = Moc3ArtMeshes::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let art_mesh_keyforms = Moc3ArtMeshKeyforms::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let deformers = Moc3Deformers::parse_with_positions(
        &moc,
        Some(art_mesh_keyforms.shared_position_xys()),
    )
    .map_err(AssetLoadError::Moc3)?;
    let bindings = Moc3KeyformBindings::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let ids = Moc3Ids::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let offscreen = Moc3OffscreenInfo::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let glues = Moc3Glues::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let parts = Moc3Parts::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let canvas = Moc3CanvasInfo::parse(&moc).map_err(AssetLoadError::Moc3)?;
    let draw_order_groups = Moc3DrawOrderGroups::parse(&moc).map_err(AssetLoadError::Moc3)?;
    // Parsed runtime arrays own their data. Release the source file before PNG
    // decoding and upload add their temporary allocations to the loading peak.
    drop(moc);
    let physics = match model.physics() {
        Some(physics_file) => {
            let physics_source = read_text(&model_dir.join(physics_file))?;
            Some(Physics3::from_json_str(&physics_source).map_err(AssetLoadError::Json)?)
        }
        None => None,
    };
    let pose = match model.pose() {
        Some(pose_file) => {
            let pose_source = read_text(&model_dir.join(pose_file))?;
            Some(Pose3::from_json_str(&pose_source).map_err(AssetLoadError::Json)?)
        }
        None => None,
    };
    let textures = decode_textures(model_dir, model.textures())?;

    Ok(ParsedModel {
        model,
        canvas,
        art_meshes,
        art_mesh_keyforms,
        deformers,
        bindings,
        ids,
        offscreen,
        glues,
        parts,
        draw_order_groups,
        physics,
        pose,
        textures,
    })
}

fn read_text(path: &Path) -> Result<String, AssetLoadError> {
    fs::read_to_string(path).map_err(|source| AssetLoadError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, AssetLoadError> {
    fs::read(path).map_err(|source| AssetLoadError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn decode_textures(
    model_dir: &Path,
    textures: &[String],
) -> Result<Vec<DecodedTexture>, AssetLoadError> {
    let paths = textures
        .iter()
        .map(|texture| model_dir.join(texture))
        .collect::<Vec<_>>();
    if paths.len() <= 1 {
        return paths.into_iter().map(decode_texture).collect();
    }

    thread::scope(|scope| {
        let handles = paths
            .iter()
            .map(|path| scope.spawn(move || decode_texture(path)))
            .collect::<Vec<_>>();

        handles
            .into_iter()
            .map(|handle| handle.join().expect("texture decode thread panicked"))
            .collect()
    })
}

fn decode_texture(path: impl AsRef<Path>) -> Result<DecodedTexture, AssetLoadError> {
    let path = path.as_ref();
    let image = image::open(path).map_err(|source| AssetLoadError::Image {
        path: path.display().to_string(),
        source,
    })?;
    let rgba = image.into_rgba8();
    Ok(DecodedTexture::new(
        rgba.width(),
        rgba.height(),
        rgba.into_raw(),
    ))
}
