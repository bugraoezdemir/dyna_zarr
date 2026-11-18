//! Coordinate transformations for dyna-zarr's engine (Workplan WP15, task DR2).
//!
//! Pure maths, no I/O and no Python: scale, translation, affine, sequences and
//! inverses; composition and inversion; mapping regions and bounding boxes
//! between coordinate spaces. The arrays crate (resampling), the OME-Zarr layer
//! (DR9) and the viewer (WP14) all build on it.
//!
//! DR1 sets up the crate only; the types come with DR2.

/// This crate's version, for the bindings' build information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn has_a_version() {
        assert!(!super::VERSION.is_empty());
    }
}
