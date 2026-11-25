//! Zarr arrays written with zarrs (Workplan WP15, task DR8; scope decided 2026-11-25:
//! local folders, Zarr v2 / v3 / sharded v3, the codecs dyna writes).
//!
//! The array is created by the caller (dyna creates it with TensorStore, so its
//! metadata is exactly what dyna wrote before); a writer opens it and stores regions:
//! each chunk the region covers is encoded with the array's codecs and written, at most
//! `threads` chunks at once (and, sharded, the shard's inner chunks in parallel).
//!
//! A region is stored unit by unit (chunks, or shards when sharded). A unit it covers
//! whole (an edge unit up to the array's end) is encoded and written as it is; a unit
//! it covers in part is read, updated and written back under that unit's lock, so
//! regions that share a unit (bands cut mid-chunk) may be stored at the same time, as
//! with TensorStore.
//! Chunks equal to the fill value are not written (zarrs' default, as TensorStore's).
//!
//! Files are written without an fsync each ([`LocalStore`]): zarrs' own store syncs every
//! chunk file before it returns, which on Windows made a copy slower than TensorStore's
//! (which does not sync either). A write is on disk when the OS writes it back.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use zarrs::array::{Array, ArrayBytes, ArraySubset, CodecOptions};
use zarrs::filesystem::FilesystemStore;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{
    Bytes, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits, StorageError,
    StoreKey, StorePrefix, WritableStorageTraits,
};

use crate::blocks::Blocks;
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::{new_pool, pieces};
use crate::source::{c_strides, region};
use crate::ReadError;

/// A local folder: zarrs' file store, but whole values written without an fsync.
pub struct LocalStore {
    files: FilesystemStore,
    base: PathBuf,
}

impl LocalStore {
    fn path(&self, key: &StoreKey) -> PathBuf {
        let mut p = self.base.clone();
        let k = key.as_str();
        if !k.is_empty() {
            p.push(k.strip_prefix('/').unwrap_or(k));
        }
        p
    }
}

impl ReadableStorageTraits for LocalStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.files.get_partial_many(key, byte_ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.files.size_key(key)
    }
    fn supports_get_partial(&self) -> bool {
        self.files.supports_get_partial()
    }
}

impl WritableStorageTraits for LocalStore {
    /// The value as the whole file, written and closed (no fsync).
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let path = self.path(key);
        if let Some(dir) = path.parent() {
            if !dir.exists() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let mut f = std::fs::File::create(&path)?;
        f.write_all(&value)?;
        Ok(())
    }
    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        self.files.set_partial_many(key, offset_values)
    }
    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.files.erase(key)
    }
    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.files.erase_prefix(prefix)
    }
    fn supports_set_partial(&self) -> bool {
        self.files.supports_set_partial()
    }
}

/// A Zarr array in a local folder, opened for writing.
pub struct ZarrWriter {
    array: Array<LocalStore>,
    label: String,
    shape: Vec<u64>,
    dtype: DType,
    /// What one store writes at once: the shard when sharded, else the chunk.
    unit: Vec<u64>,
    /// Locks for units stored in part, by a hash of the unit's index.
    stripes: Vec<std::sync::Mutex<()>>,
}

/// Locks for units stored in part (striped: a fixed number, whatever the array's size).
const STRIPES: usize = 256;

impl ZarrWriter {
    /// The array in the local folder `path` (created already, its metadata stored).
    pub fn open(path: &Path) -> Result<Self, ReadError> {
        let label = path.display().to_string();
        let files = FilesystemStore::new(path)
            .map_err(|e| ReadError::new(format!("cannot open {label}: {e}")))?;
        let store = LocalStore {
            files,
            base: path.to_path_buf(),
        };
        let array = Array::open(Arc::new(store), "/").map_err(|e| {
            ReadError::new(format!("{label} is not a Zarr array zarrs can write: {e}"))
        })?;
        let shape = array.shape().to_vec();
        let name = array
            .data_type()
            .name(ZarrVersion::V3)
            .map_or_else(|| array.data_type().to_string(), |n| n.into_owned());
        let dtype =
            DType::parse(&name).map_err(|e| ReadError::new(format!("{label}: {}", e.message)))?;
        let unit = array
            .chunk_shape(&vec![0u64; shape.len()])
            .map_err(|e| ReadError::new(format!("{label}: no chunk shape: {e}")))?
            .iter()
            .map(|c| c.get())
            .collect();
        Ok(Self {
            array,
            label,
            shape,
            dtype,
            unit,
            stripes: (0..STRIPES).map(|_| std::sync::Mutex::new(())).collect(),
        })
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The write unit: the shard when sharded, else the chunk.
    pub fn unit(&self) -> &[u64] {
        &self.unit
    }

    fn options(threads: usize) -> CodecOptions {
        // as for reads (zarr.rs): made once on a pool's thread, so that it starts no
        // global pool; exactly `threads` as the target, no minimum of chunks at once
        static BASE: OnceLock<CodecOptions> = OnceLock::new();
        let base = *BASE.get_or_init(|| {
            if rayon::current_thread_index().is_some() {
                CodecOptions::default()
            } else {
                new_pool(1)
                    .map(|p| p.install(CodecOptions::default))
                    .unwrap_or_default()
            }
        });
        base.with_concurrent_target(threads.max(1))
            .with_chunk_concurrent_minimum(1)
    }

    /// Store `bytes` (C order, the array's element type in native byte order) as the
    /// region `start .. start + shape`, with `threads`.
    pub fn store(
        &self,
        start: &[u64],
        shape: &[u64],
        bytes: &[u8],
        threads: usize,
    ) -> Result<(), ReadError> {
        let stop: Vec<u64> = start.iter().zip(shape).map(|(a, r)| a + r).collect();
        let r = region(&self.shape, start, &stop)?;
        let want = r.iter().product::<u64>() as usize * self.dtype.size();
        if bytes.len() != want {
            return Err(ReadError::new(format!(
                "{}: a region of {r:?} {} needs {want} bytes, not {}",
                self.label,
                self.dtype.name(),
                bytes.len()
            )));
        }
        if r.contains(&0) {
            return Ok(());
        }
        let n = r.len();
        let item = self.dtype.size();
        let parts = Blocks::aligned(start, &r, &self.unit);
        let rstrides = c_strides(&r, item);
        let err = |e: &dyn std::fmt::Display| ReadError::new(format!("{}: {e}", self.label));
        pieces(parts.len(), threads, |k, each| {
            let b = parts.get(k);
            let first: Vec<u64> = (0..n).map(|i| start[i] + b[i].0).collect();
            let index: Vec<u64> = (0..n).map(|i| first[i] / self.unit[i]).collect();
            let origin: Vec<u64> = (0..n).map(|i| index[i] * self.unit[i]).collect();
            let sub: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
            let whole =
                (0..n).all(|i| first[i] == origin[i] && sub[i] == self.unit[i].min(self.shape[i] - origin[i]));
            // the piece's elements, C-ordered (the region itself when it is one piece)
            let own;
            let data: &[u8] = if parts.len() == 1 {
                bytes
            } else {
                let mut v = vec![0u8; sub.iter().product::<u64>() as usize * item];
                {
                    let o = Out::new(&mut v, &sub, &vec![0; n], self.dtype)?;
                    let offset: usize = (0..n).map(|i| b[i].0 as usize * rstrides[i]).sum();
                    o.copy_from(&sub, &bytes[offset..], &rstrides, item, None);
                }
                own = v;
                &own
            };
            let at: Vec<u64> = (0..n).map(|i| first[i] - origin[i]).collect();
            let in_unit = ArraySubset::new_with_start_shape(at, sub).map_err(|e| err(&e))?;
            let options = Self::options(each);
            let store = || {
                self.array
                    .store_chunk_subset_opt(&index, &in_unit, ArrayBytes::from(data), &options)
                    .map_err(|e| err(&e))
            };
            if whole {
                store()
            } else {
                // read, updated and written back: one region at a time per unit
                let h = index
                    .iter()
                    .fold(0xcbf29ce484222325u64, |h, &v| (h ^ v).wrapping_mul(0x100000001b3));
                let _held = self.stripes[(h % STRIPES as u64) as usize]
                    .lock()
                    .map_err(|_| ReadError::new("a unit's lock is poisoned".to_string()))?;
                store()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::read;
    use crate::zarr::ZarrArray;
    use zarrs::array::{data_type, ArrayBuilder};

    fn create(path: &Path, shards: bool) {
        let store = Arc::new(FilesystemStore::new(path).unwrap());
        let mut b = ArrayBuilder::new(
            vec![5, 70, 90],
            if shards { vec![5, 32, 32] } else { vec![2, 16, 16] },
            data_type::uint16(),
            0u16,
        );
        if shards {
            b.subchunk_shape(vec![5, 16, 16]);
        }
        b.build(store, "/").unwrap().store_metadata().unwrap();
    }

    #[test]
    fn stores_whole_units_and_edges() {
        for shards in [false, true] {
            let dir = std::env::temp_dir().join(format!(
                "dyna_writer_{}_{shards}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            create(&dir, shards);
            let w = ZarrWriter::open(&dir).unwrap();
            let all: Vec<u16> = (0..5 * 70 * 90).map(|i| (i % 5000) as u16).collect();
            let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_ne_bytes()).collect();
            // in two parts cut on the write unit along y
            let cut = w.unit()[1];
            let row = 90 * 2;
            let per_plane = 70 * row;
            let mut top = Vec::new();
            let mut bottom = Vec::new();
            for z in 0..5 {
                let p = &bytes[z * per_plane..(z + 1) * per_plane];
                top.extend_from_slice(&p[..cut as usize * row]);
                bottom.extend_from_slice(&p[cut as usize * row..]);
            }
            w.store(&[0, 0, 0], &[5, cut, 90], &top, 4).unwrap();
            w.store(&[0, cut, 0], &[5, 70 - cut, 90], &bottom, 1).unwrap();
            let back = read(&ZarrArray::open(&dir).unwrap(), &[0, 0, 0], &[5, 70, 90], 2).unwrap();
            assert_eq!(back, bytes, "sharded: {shards}");
            assert!(w.store(&[0, 0, 0], &[5, 70, 90], &bytes[1..], 1).is_err());
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn bands_cut_mid_chunk_stored_at_once_lose_nothing() {
        for shards in [false, true] {
            let dir = std::env::temp_dir().join(format!(
                "dyna_writer_bands_{}_{shards}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            create(&dir, shards);
            let w = ZarrWriter::open(&dir).unwrap();
            // bands of 7 rows along y, each in its own thread, chunks of 16 / 32 rows
            std::thread::scope(|sc| {
                for y0 in (0..70u64).step_by(7) {
                    let w = &w;
                    sc.spawn(move || {
                        let rows = 7.min(70 - y0);
                        let v: Vec<u8> = (0..5 * rows * 90)
                            .flat_map(|i| {
                                let (z, rest) = (i / (rows * 90), i % (rows * 90));
                                let (y, x) = (y0 + rest / 90, rest % 90);
                                (((z * 70 + y) * 90 + x) as u16 + 1).to_ne_bytes()
                            })
                            .collect();
                        w.store(&[0, y0, 0], &[5, rows, 90], &v, 2).unwrap();
                    });
                }
            });
            let back = read(&ZarrArray::open(&dir).unwrap(), &[0, 0, 0], &[5, 70, 90], 2).unwrap();
            let want: Vec<u8> = (0..5 * 70 * 90u64)
                .flat_map(|i| (i as u16 + 1).to_ne_bytes())
                .collect();
            assert_eq!(back, want, "sharded: {shards}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
