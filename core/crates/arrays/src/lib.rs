//! Lazy arrays over zarrs for dyna-zarr's engine (Workplan WP15, tasks DR3-DR5).
//!
//! DR1 sets up the crate and proves the zarrs build on every platform with one
//! small function, [`array_info`]: open a Zarr v2 or v3 array in a local folder
//! and describe it. Regions, views, resampling and downscaling come with DR3-DR5.

use std::path::Path;
use std::sync::Arc;

use zarrs::array::{Array, ArrayMetadata};
use zarrs::filesystem::FilesystemStore;
use zarrs::plugin::{ExtensionName, ZarrVersion};

/// This crate's version, for the bindings' build information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What an array is: its shape, the shape of its first chunk, its data type and
/// its Zarr format (2 or 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayInfo {
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub dtype: String,
    pub zarr_format: u8,
}

/// Open the Zarr array stored in the local folder `path` (v2 or v3) and describe it.
pub fn array_info(path: &Path) -> Result<ArrayInfo, String> {
    let store =
        FilesystemStore::new(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let array = Array::open(Arc::new(store), "/")
        .map_err(|e| format!("{} is not a Zarr array zarrs can read: {e}", path.display()))?;
    let shape = array.shape().to_vec();
    let origin = vec![0u64; shape.len()];
    let chunks = array
        .chunk_shape(&origin)
        .map_err(|e| format!("{}: no chunk shape: {e}", path.display()))?
        .iter()
        .map(|c| c.get())
        .collect();
    let zarr_format = match array.metadata() {
        ArrayMetadata::V2(_) => 2,
        ArrayMetadata::V3(_) => 3,
    };
    Ok(ArrayInfo {
        shape,
        chunks,
        // the Zarr v3 name ("uint16"), which is also NumPy's for the numeric types;
        // Display would give "uint16 / <u2" (v3 / v2)
        dtype: array
            .data_type()
            .name(ZarrVersion::V3)
            .map_or_else(|| array.data_type().to_string(), |n| n.into_owned()),
        zarr_format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zarrs::array::{data_type, ArrayBuilder};

    #[test]
    fn describes_a_v3_array() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FilesystemStore::new(dir.path()).unwrap());
        let array = ArrayBuilder::new(vec![10, 20], vec![4, 8], data_type::uint16(), 0u16)
            .build(store, "/")
            .unwrap();
        array.store_metadata().unwrap();
        let info = array_info(dir.path()).unwrap();
        assert_eq!(info.shape, vec![10, 20]);
        assert_eq!(info.chunks, vec![4, 8]);
        assert_eq!(info.dtype, "uint16");
        assert_eq!(info.zarr_format, 3);
    }

    #[test]
    fn a_folder_without_an_array_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(array_info(dir.path()).is_err());
    }
}
