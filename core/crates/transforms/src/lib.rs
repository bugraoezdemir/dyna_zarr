//! Coordinate transformations for dyna-zarr's engine (Workplan WP15, task DR2).
//!
//! Pure maths, no I/O and no Python, mirroring ome_zarr_pyramid's spatial model
//! (`core/spatial.py`) so both packages give the same answers:
//!
//! - [`Transformation`]: an NGFF transformation as read (type, ends, name, parameters),
//!   its homogeneous [`Matrix`] on demand; [`compose`], [`Transformation::invert`],
//!   [`Transformation::from_matrix`], [`Transformation::scale_translation`].
//!   Modelled: identity, translation, scale, affine, sequence (ozp's), rotation and
//!   mapAxis (NGFF 0.6; ozp follows). Any other type is carried as read.
//! - [`Matrix::embed`]: a transformation over some axes acting on an array's full axes.
//! - [`bounding_box`], [`input_region`]: what part of an input an output region needs
//!   (pixel centres at integer indices, as NGFF 0.6), for resampling (DR4) and viewing.
//! - [`unit_factor`]: ozp's unit table.

mod matrix;
mod region;
mod transform;
mod units;

pub use matrix::Matrix;
pub use region::{bounding_box, input_region, snap, IndexBox, Interpolation};
pub use transform::{compose, ScaleTranslation, Transformation, MODELLED_TYPES};
pub use units::unit_factor;

/// This crate's version, for the bindings' build information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A transformation cannot be modelled or applied as asked; the message says why.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformError {
    pub message: String,
}

impl TransformError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TransformError {}
