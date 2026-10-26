//! Cubism 2 (`.moc`) model support.
//!
//! Cubism 2 models use a `.moc` binary, a `.model.json` settings file, `.mtn`
//! motions, `.exp` expressions, `.physics` chains, and `.pose` groups. This
//! module reproduces the Cubism 2 runtime:
//!
//! - [`Moc2Model`](crate::moc2::Moc2Model) parses a `.moc` file and generates
//!   its default-pose meshes.
//! - [`Moc2Runtime`](crate::moc2::Moc2Runtime) evaluates parameters, deformers,
//!   physics, motions, expressions, pose fading, eye blinking, and the built-in
//!   breathing motion, then rebuilds renderer-ready
//!   [`crate::moc3::Moc3DrawableMesh`] data.
//! - [`Moc2Settings`](crate::moc2::Moc2Settings),
//!   [`Moc2Motion`](crate::moc2::Moc2Motion),
//!   [`Moc2Expression`](crate::moc2::Moc2Expression),
//!   [`Moc2Physics`](crate::moc2::Moc2Physics), and
//!   [`Moc2Pose`](crate::moc2::Moc2Pose) parse the JSON and text sidecar files
//!   referenced by a model.
//!
//! ```no_run
//! use neocari::assets::load_moc2_runtime;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut model = load_moc2_runtime("path/to/model.model.json")?;
//! model.play_motion("idle", 0)?;
//! model.update(1.0 / 60.0);
//!
//! for mesh in model.runtime().meshes() {
//!     // Upload mesh.vertices() and mesh.indices() to a renderer.
//! }
//! # Ok(())
//! # }
//! ```

mod behaviors;
mod deformers;
mod expression;
mod json;
mod layout;
mod motion;
mod parse;
mod physics;
mod pose;
mod runtime;
mod scene;

use crate::moc3::Moc3DrawableMesh;

pub use behaviors::Moc2EyeBlink;
pub use json::{
    Moc2Expression, Moc2ExpressionBlend, Moc2ExpressionParameter, Moc2ExpressionReference,
    Moc2HitArea, Moc2InitialParameter, Moc2InitialPartVisible, Moc2MotionReference, Moc2Physics,
    Moc2PhysicsHair, Moc2PhysicsSource, Moc2PhysicsSourceKind, Moc2PhysicsTarget,
    Moc2PhysicsTargetKind, Moc2Pose, Moc2PosePart, Moc2Settings,
};
pub use layout::Moc2ModelMatrix;
pub use motion::{
    Moc2Motion, Moc2MotionChannel, Moc2MotionChannelKind, Moc2MotionLayout, Moc2MotionManager,
    priority as motion_priority,
};
pub use physics::Moc2PhysicsRuntime;
pub use pose::Moc2PoseRuntime;
pub use runtime::{Moc2Runtime, Moc2RuntimeOptions};
pub use scene::{
    Moc2Affine, Moc2Canvas, Moc2Deformer, Moc2DeformerKind, Moc2Drawable, Moc2Parameter, Moc2Part,
    Moc2PivotDimension, Moc2PivotSample, Moc2PivotTable, Moc2Scene,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors produced while parsing Cubism 2 model data and sidecar files.
pub enum Moc2Error {
    /// The file does not start with the `moc` magic bytes.
    #[error("invalid moc2 signature")]
    InvalidMagic,
    /// The `.moc` serialization version is not supported.
    #[error("unsupported moc2 format version {0}")]
    UnsupportedVersion(u8),
    /// The file ended before a field could be read.
    #[error("unexpected end of moc2 data at byte {0}")]
    UnexpectedEof(usize),
    /// A variable-length integer was malformed.
    #[error("invalid moc2 integer encoding at byte {0}")]
    InvalidInteger(usize),
    /// An object reference pointed outside the parsed object table.
    #[error("invalid moc2 object reference {0}")]
    InvalidReference(i32),
    /// The `.moc` file uses an object type this parser does not know.
    #[error("unsupported moc2 object type {0}")]
    UnsupportedObjectType(u32),
    /// The `.moc` file is internally inconsistent.
    #[error("invalid moc2 model data: {0}")]
    InvalidData(&'static str),
    /// The model uses a texture option this runtime does not support.
    #[error("unsupported moc2 texture option {0:#x}")]
    UnsupportedTextureOption(i32),
    /// A Cubism 2 JSON sidecar file was malformed.
    #[error("invalid {format}: {message}")]
    InvalidJson {
        /// Human-readable format name, such as `model.json`.
        format: &'static str,
        /// Specific validation failure.
        message: String,
    },
}

/// A parsed Cubism 2 model with meshes evaluated at its default parameters.
///
/// Use [`Moc2Model::into_runtime`] to animate the model. The generated meshes
/// use the shared [`Moc3DrawableMesh`] representation, so the same renderers
/// handle Cubism 2 and Cubism 3 models.
#[derive(Debug, Clone)]
pub struct Moc2Model {
    scene: Moc2Scene,
    drawables: Vec<Moc3DrawableMesh>,
}

impl Moc2Model {
    /// Parses a `.moc` file and generates meshes in the default pose.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Moc2Error> {
        let scene = parse::parse_scene(bytes)?;
        let mut runtime = Moc2Runtime::new(scene)
            .ok_or(Moc2Error::InvalidData("failed to build drawable meshes"))?;
        runtime
            .update_meshes()
            .ok_or(Moc2Error::InvalidData("failed to build drawable meshes"))?;
        let (scene, drawables) = runtime.into_scene_and_meshes();
        Ok(Self { scene, drawables })
    }

    /// Returns the parsed scene.
    pub fn scene(&self) -> &Moc2Scene {
        &self.scene
    }

    /// Returns the `.moc` serialization version.
    pub fn format_version(&self) -> u8 {
        self.scene.format_version()
    }

    /// Returns the canvas width in model pixels.
    pub fn canvas_width(&self) -> u32 {
        self.scene.canvas_width()
    }

    /// Returns the canvas height in model pixels.
    pub fn canvas_height(&self) -> u32 {
        self.scene.canvas_height()
    }

    /// Returns drawables in model order, evaluated at the default parameters.
    pub fn drawables(&self) -> &[Moc3DrawableMesh] {
        &self.drawables
    }

    /// Consumes the model and returns a mutable runtime.
    pub fn into_runtime(self) -> Option<Moc2Runtime> {
        Moc2Runtime::new(self.scene)
    }

    /// Consumes the model and returns its parsed scene.
    pub fn into_scene(self) -> Moc2Scene {
        self.scene
    }
}
