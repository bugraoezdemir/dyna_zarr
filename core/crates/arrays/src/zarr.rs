//! Zarr v2 / v3 arrays read with zarrs: in a local folder, or inside a finished zip.
//! A region is decoded straight into the destination (zarrs' decode-into), chunks in
//! parallel on the read's thread pool.

use std::path::Path;
use std::sync::Arc;

use zarrs::array::{Array, ArrayBytesFixedDisjointView, ArrayMetadata, ArraySubset, CodecOptions};
use zarrs::filesystem::FilesystemStore;
use zarrs::plugin::{ExtensionName, ZarrVersion};
use zarrs::storage::{ReadableStorage, StoreKey};
use zarrs_zip::ZipStorageAdapter;

use crate::dtype::DType;
use crate::out::Out;
use crate::pool::pool;
use crate::source::{region, Source};
use crate::ReadError;

/// A Zarr array, opened read-only.
pub struct ZarrArray {
    array: Array<dyn zarrs::storage::ReadableStorageTraits>,
    label: String,
    shape: Vec<u64>,
    dtype: DType,
    chunks: Vec<u64>,
    zarr_format: u8,
}

impl ZarrArray {
    /// The array stored in the local folder `path`.
    pub fn open(path: &Path) -> Result<Self, ReadError> {
        let store = FilesystemStore::new(path)
            .map_err(|e| ReadError::new(format!("cannot open {}: {e}", path.display())))?;
        Self::from_storage(Arc::new(store), "/", path.display().to_string())
    }

    /// The array at `inner` (a path inside the zip, "" for its root) in the zip file
    /// `zip`, as zarr-zip and dyna's io.open_zip write them.
    pub fn open_zip(zip: &Path, inner: &str) -> Result<Self, ReadError> {
        let label = format!("{}::{inner}", zip.display());
        let dir = zip
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = zip
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| ReadError::new(format!("{}: not a file name", zip.display())))?;
        let store = FilesystemStore::new(dir)
            .map_err(|e| ReadError::new(format!("cannot open {}: {e}", dir.display())))?;
        let key =
            StoreKey::new(name).map_err(|e| ReadError::new(format!("{}: {e}", zip.display())))?;
        let adapter = ZipStorageAdapter::new(Arc::new(store), key).map_err(|e| {
            ReadError::new(format!(
                "{} is not a zip zarrs can read: {e}",
                zip.display()
            ))
        })?;
        let inner = inner.trim_matches('/');
        let path = if inner.is_empty() {
            "/".to_string()
        } else {
            format!("/{inner}")
        };
        Self::from_storage(Arc::new(adapter), &path, label)
    }

    fn from_storage(
        storage: ReadableStorage,
        path: &str,
        label: String,
    ) -> Result<Self, ReadError> {
        let array = Array::open(storage, path).map_err(|e| {
            ReadError::new(format!("{label} is not a Zarr array zarrs can read: {e}"))
        })?;
        let shape = array.shape().to_vec();
        let name = array
            .data_type()
            .name(ZarrVersion::V3)
            .map_or_else(|| array.data_type().to_string(), |n| n.into_owned());
        let dtype =
            DType::parse(&name).map_err(|e| ReadError::new(format!("{label}: {}", e.message)))?;
        let chunks = array
            .chunk_shape(&vec![0u64; shape.len()])
            .map_err(|e| ReadError::new(format!("{label}: no chunk shape: {e}")))?
            .iter()
            .map(|c| c.get())
            .collect();
        let zarr_format = match array.metadata() {
            ArrayMetadata::V2(_) => 2,
            ArrayMetadata::V3(_) => 3,
        };
        Ok(Self {
            array,
            label,
            shape,
            dtype,
            chunks,
            zarr_format,
        })
    }

    /// The shape of the first chunk (the read grid; with sharding, the shard).
    pub fn chunks(&self) -> &[u64] {
        &self.chunks
    }

    /// 2 or 3.
    pub fn zarr_format(&self) -> u8 {
        self.zarr_format
    }

    /// The array's metadata as stored (JSON text): codecs, fill value, dimension names.
    pub fn metadata_json(&self) -> String {
        serde_json::to_string(self.array.metadata()).unwrap_or_default()
    }
}

impl Source for ZarrArray {
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn grid(&self) -> Option<Vec<u64>> {
        Some(self.chunks.clone())
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = region(&self.shape, start, stop)?;
        out.check(&r, self.dtype)?;
        if r.contains(&0) {
            return Ok(());
        }
        let subset = ArraySubset::new_with_start_shape(start.to_vec(), r.clone())
            .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
        let target_subset = ArraySubset::new_with_start_shape(out.at.clone(), r)
            .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
        // SAFETY: the view covers only the region `out` was given, which no other
        // concurrent read targets (the Out contract).
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(out.buf, self.dtype.size(), &out.shape, target_subset)
        }
        .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
        let options = CodecOptions::default().with_concurrent_target(threads.max(1));
        pool(threads)?
            .install(|| {
                self.array
                    .retrieve_array_subset_into_opt(&subset, (&mut view).into(), &options)
            })
            .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))
    }
}
