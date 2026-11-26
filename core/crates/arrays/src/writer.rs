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
//! with TensorStore. In a sharded array that update appends the region's inner chunks to
//! the shard's file in place and rewrites its index after them (zarrs' partial encoding),
//! so a shard written inner chunk by inner chunk holds only the chunks being written.
//!
//! write_levels' bounded mode (shards larger than a block) appends its inner chunks one at
//! a time with [`ZarrWriter::append`] instead: each is encoded with the shard's inner
//! codecs and appended to the shard's file, the shard's index kept in memory and written
//! once, when its last inner chunk is in (re-reading and rewriting the index for every
//! inner chunk made a sharded copy on one thread 1.65x the Python path's time,
//! 2026-11-26). The shard's memory is its index, 16 bytes per inner chunk.
//! Chunks equal to the fill value are not written (zarrs' default, as TensorStore's).
//!
//! Files are written without an fsync each ([`LocalStore`]): zarrs' own store syncs every
//! chunk file before it returns, which on Windows made a copy slower than TensorStore's
//! (which does not sync either). A write is on disk when the OS writes it back.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use std::collections::HashMap;
use std::num::NonZeroU64;

use zarrs::array::codec::api::ArrayToBytesCodecTraits;
use zarrs::array::{
    data_type, Array, ArrayBytes, ArrayShardedExt, ArraySubset, CodecChain, CodecOptions, FillValue,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::v3::MetadataV3;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{
    Bytes, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
    StorePrefix, WritableStorageTraits,
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
    pub(crate) fn path(&self, key: &StoreKey) -> PathBuf {
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
    /// Each value written at its offset, in place (the file grows as needed; no fsync).
    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        let path = self.path(key);
        if let Some(dir) = path.parent() {
            if !dir.exists() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        for (offset, value) in offset_values {
            write_all_at(&f, &value, offset)?;
        }
        Ok(())
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

/// `buf` written at `offset` of `f`.
fn write_all_at(f: &std::fs::File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_write(f, buf, offset)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::write_at(f, buf, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "a write of 0 bytes",
            ));
        }
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// A shard being appended to: its file once it has data, where the next inner chunk goes,
/// its index ((offset, size) per inner chunk, u64::MAX for none) and how many of its inner
/// chunks are still to come.
struct OpenShard {
    file: Option<std::fs::File>,
    offset: u64,
    index: Vec<u64>,
    left: u64,
}

/// What appending inner chunks to shards needs: the shard's inner codecs and index codecs
/// (from the array's metadata), the inner chunk's shape and the shards being written.
struct Appender {
    inner: CodecChain,
    index_codecs: CodecChain,
    inner_shape: Vec<u64>,
    per_shard: Vec<u64>,
    open: std::sync::Mutex<HashMap<Vec<u64>, Arc<std::sync::Mutex<OpenShard>>>>,
}

impl Appender {
    /// From a sharded array's metadata: one sharding_indexed codec with its index at the
    /// end (dyna's and TensorStore's layout); None for anything else (then `store`'s
    /// partial encoding writes it).
    fn of(meta: &serde_json::Value, shard: &[u64]) -> Option<Self> {
        let codecs = meta.get("codecs")?.as_array()?;
        if codecs.len() != 1 || codecs[0].get("name")?.as_str()? != "sharding_indexed" {
            return None;
        }
        let conf = codecs[0].get("configuration")?;
        if conf
            .get("index_location")
            .and_then(|v| v.as_str())
            .unwrap_or("end")
            != "end"
        {
            return None;
        }
        let inner_shape: Vec<u64> =
            serde_json::from_value(conf.get("chunk_shape")?.clone()).ok()?;
        let chain = |key: &str| -> Option<CodecChain> {
            let m: Vec<MetadataV3> = serde_json::from_value(conf.get(key)?.clone()).ok()?;
            CodecChain::from_metadata(&m).ok()
        };
        if inner_shape.len() != shard.len() || inner_shape.contains(&0) {
            return None;
        }
        let per_shard: Vec<u64> = shard
            .iter()
            .zip(&inner_shape)
            .map(|(s, c)| s.div_ceil(*c))
            .collect();
        Some(Self {
            inner: chain("codecs")?,
            index_codecs: chain("index_codecs")?,
            inner_shape,
            per_shard,
            open: std::sync::Mutex::new(HashMap::new()),
        })
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
    sharded: bool,
    /// Appending inner chunks to shards (sharded arrays with the index at the end).
    appender: Option<Appender>,
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
        let array: Array<LocalStore> = Array::open(Arc::new(store), "/").map_err(|e| {
            ReadError::new(format!("{label} is not a Zarr array zarrs can write: {e}"))
        })?;
        let shape = array.shape().to_vec();
        let name = array
            .data_type()
            .name(ZarrVersion::V3)
            .map_or_else(|| array.data_type().to_string(), |n| n.into_owned());
        let dtype =
            DType::parse(&name).map_err(|e| ReadError::new(format!("{label}: {}", e.message)))?;
        let unit: Vec<u64> = array
            .chunk_shape(&vec![0u64; shape.len()])
            .map_err(|e| ReadError::new(format!("{label}: no chunk shape: {e}")))?
            .iter()
            .map(|c| c.get())
            .collect();
        let sharded = array.is_sharded();
        let appender = if sharded {
            serde_json::to_value(array.metadata())
                .ok()
                .and_then(|m| Appender::of(&m, &unit))
        } else {
            None
        };
        Ok(Self {
            array,
            label,
            shape,
            dtype,
            unit,
            sharded,
            appender,
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

    /// The unit piece `k` of `parts` lies in, and whether the piece covers it whole (an
    /// edge unit up to the array's end): (first element, unit index, unit origin, extent,
    /// whole).
    fn piece(
        &self,
        parts: &Blocks,
        start: &[u64],
        k: usize,
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, bool) {
        let n = start.len();
        let b = parts.get(k);
        let first: Vec<u64> = (0..n).map(|i| start[i] + b[i].0).collect();
        let index: Vec<u64> = (0..n).map(|i| first[i] / self.unit[i]).collect();
        let origin: Vec<u64> = (0..n).map(|i| index[i] * self.unit[i]).collect();
        let sub: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
        let whole = (0..n).all(|i| {
            first[i] == origin[i] && sub[i] == self.unit[i].min(self.shape[i] - origin[i])
        });
        (first, index, origin, sub, whole)
    }

    /// Whether [`Self::append`] can append inner chunks to this array's shards.
    pub fn appends(&self) -> bool {
        self.appender.is_some()
    }

    /// Append the inner chunk at `start` (`shape`: the inner chunk, cut at the array's
    /// end) to its shard: encoded with the shard's inner codecs and written after the
    /// shard's data so far; its index entry kept in memory. The shard's index is written
    /// when its last inner chunk is in (or by [`Self::finish_shards`]). A chunk equal to
    /// the fill value is not written; a shard with no data gets no file. Each inner chunk
    /// is appended once; the caller appends to one shard from one thread at a time (or
    /// waits: the shard's lock is taken here).
    pub fn append(&self, start: &[u64], shape: &[u64], bytes: &[u8]) -> Result<(), ReadError> {
        let ap = self
            .appender
            .as_ref()
            .ok_or_else(|| ReadError::new(format!("{}: not appended by shards", self.label)))?;
        let n = self.shape.len();
        let stop: Vec<u64> = start.iter().zip(shape).map(|(a, r)| a + r).collect();
        let r = region(&self.shape, start, &stop)?;
        let item = self.dtype.size();
        let inner = &ap.inner_shape;
        let whole: Vec<u64> = (0..n)
            .map(|i| inner[i].min(self.shape[i] - start[i]))
            .collect();
        if (0..n).any(|i| !start[i].is_multiple_of(inner[i])) || r != whole {
            return Err(ReadError::new(format!(
                "{}: {start:?} + {shape:?} is not one inner chunk of {inner:?}",
                self.label
            )));
        }
        if bytes.len() != r.iter().product::<u64>() as usize * item {
            return Err(ReadError::new(format!(
                "{}: {} bytes for {r:?}",
                self.label,
                bytes.len()
            )));
        }
        let fill = self.array.fill_value();
        // the inner chunk at its full shape (cut at the array's end: padded with the fill)
        let padded;
        let data: &[u8] = if r == *inner {
            bytes
        } else {
            let fv = fill.as_ne_bytes();
            let count = inner.iter().product::<u64>() as usize;
            let mut v: Vec<u8> = Vec::with_capacity(count * item);
            for _ in 0..count {
                v.extend_from_slice(fv);
            }
            {
                let o = Out::new(&mut v, inner, &vec![0; n], self.dtype)?;
                o.copy_from(&r, bytes, &c_strides(&r, item), item, None);
            }
            padded = v;
            &padded
        };
        let ab = ArrayBytes::from(data);
        let encoded = if ab.is_fill_value(fill) {
            None
        } else {
            let shape_nz: Vec<NonZeroU64> = inner
                .iter()
                .map(|&c| NonZeroU64::new(c).expect("inner chunks are > 0"))
                .collect();
            let e = ap
                .inner
                .encode(
                    ab,
                    &shape_nz,
                    self.array.data_type(),
                    fill,
                    &Self::options(1),
                )
                .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
            Some(e.into_owned())
        };
        // the shard and the inner chunk's place in it
        let shard: Vec<u64> = (0..n).map(|i| start[i] / self.unit[i]).collect();
        let k = (0..n).fold(0u64, |acc, i| {
            acc * ap.per_shard[i] + (start[i] - shard[i] * self.unit[i]) / inner[i]
        }) as usize;
        let state = {
            let mut open = ap
                .open
                .lock()
                .map_err(|_| ReadError::new("the open shards' lock is poisoned".to_string()))?;
            open.entry(shard.clone())
                .or_insert_with(|| {
                    // inner chunks of this shard inside the array
                    let left = (0..n)
                        .map(|i| {
                            let o = shard[i] * self.unit[i];
                            self.unit[i].min(self.shape[i] - o).div_ceil(inner[i])
                        })
                        .product();
                    let slots = ap.per_shard.iter().product::<u64>() as usize;
                    Arc::new(std::sync::Mutex::new(OpenShard {
                        file: None,
                        offset: 0,
                        index: vec![u64::MAX; 2 * slots],
                        left,
                    }))
                })
                .clone()
        };
        let mut st = state
            .lock()
            .map_err(|_| ReadError::new("a shard's lock is poisoned".to_string()))?;
        if st.index[2 * k] != u64::MAX || st.left == 0 {
            return Err(ReadError::new(format!(
                "{}: the inner chunk at {start:?} is appended twice",
                self.label
            )));
        }
        if let Some(enc) = encoded {
            if st.file.is_none() {
                let path = self.array.storage().path(&self.array.chunk_key(&shard));
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)
                        .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
                }
                st.file = Some(
                    std::fs::File::create(&path)
                        .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?,
                );
            }
            let at = st.offset;
            write_all_at(st.file.as_ref().expect("opened above"), &enc, at)
                .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
            st.index[2 * k] = at;
            st.index[2 * k + 1] = enc.len() as u64;
            st.offset += enc.len() as u64;
        }
        st.left -= 1;
        if st.left == 0 {
            self.close_shard(ap, &mut st)?;
            drop(st);
            if let Ok(mut open) = ap.open.lock() {
                open.remove(&shard);
            }
        }
        Ok(())
    }

    /// Write a shard's index after its data and close its file (no file: no data).
    fn close_shard(&self, ap: &Appender, st: &mut OpenShard) -> Result<(), ReadError> {
        let Some(file) = st.file.take() else {
            return Ok(());
        };
        let raw: Vec<u8> = st.index.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut index_shape: Vec<NonZeroU64> = ap
            .per_shard
            .iter()
            .map(|&c| NonZeroU64::new(c).expect("inner chunks per shard are > 0"))
            .collect();
        index_shape.push(NonZeroU64::new(2).expect("2 > 0"));
        let enc = ap
            .index_codecs
            .encode(
                ArrayBytes::from(raw.as_slice()),
                &index_shape,
                &data_type::uint64(),
                &FillValue::from(u64::MAX),
                &Self::options(1),
            )
            .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
        write_all_at(&file, &enc, st.offset)
            .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
        Ok(())
    }

    /// Close every shard still open (inner chunks never appended read as the fill value).
    pub fn finish_shards(&self) -> Result<(), ReadError> {
        let Some(ap) = self.appender.as_ref() else {
            return Ok(());
        };
        let open: Vec<_> = ap
            .open
            .lock()
            .map_err(|_| ReadError::new("the open shards' lock is poisoned".to_string()))?
            .drain()
            .collect();
        for (_, state) in open {
            let mut st = state
                .lock()
                .map_err(|_| ReadError::new("a shard's lock is poisoned".to_string()))?;
            self.close_shard(ap, &mut st)?;
        }
        Ok(())
    }

    fn stripe(index: &[u64]) -> usize {
        let h = index.iter().fold(0xcbf29ce484222325u64, |h, &v| {
            (h ^ v).wrapping_mul(0x100000001b3)
        });
        (h % STRIPES as u64) as usize
    }

    /// Store `bytes` (C order, the array's element type in native byte order) as the
    /// region `start .. start + shape`, with `threads`, on `pool` when given (else as
    /// [`crate::pool::enter`]).
    ///
    /// The locks of the units the region covers in part are taken here, on the calling
    /// thread, in order, before any work starts on the pool, and held until the region is
    /// stored. Taken inside the pool, a thread holding one could, while zarrs' parallel
    /// work waits, pick up another piece that waits for the same lock (it did: a test
    /// hung).
    pub fn store(
        &self,
        start: &[u64],
        shape: &[u64],
        bytes: &[u8],
        threads: usize,
        pool: Option<&rayon::ThreadPool>,
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
        let mut stripes: Vec<usize> = (0..parts.len())
            .filter_map(|k| {
                let (_, index, _, _, whole) = self.piece(&parts, start, k);
                (!whole).then(|| Self::stripe(&index))
            })
            .collect();
        stripes.sort_unstable();
        stripes.dedup();
        let _held = stripes
            .iter()
            .map(|&i| self.stripes[i].lock())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ReadError::new("a unit's lock is poisoned".to_string()))?;
        let rstrides = c_strides(&r, item);
        let err = |e: &dyn std::fmt::Display| ReadError::new(format!("{}: {e}", self.label));
        let run = || {
            pieces(parts.len(), threads, |k, each| {
                let b = parts.get(k);
                let (first, index, origin, sub, whole) = self.piece(&parts, start, k);
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
                // part of a shard: its inner chunks appended in place (partial encoding)
                let options =
                    Self::options(each).with_experimental_partial_encoding(self.sharded && !whole);
                self.array
                    .store_chunk_subset_opt(&index, &in_unit, ArrayBytes::from(data), &options)
                    .map_err(|e| err(&e))
            })
        };
        crate::pool::enter(threads, pool, run)?
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
            if shards {
                vec![5, 32, 32]
            } else {
                vec![2, 16, 16]
            },
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
            let dir =
                std::env::temp_dir().join(format!("dyna_writer_{}_{shards}", std::process::id()));
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
            w.store(&[0, 0, 0], &[5, cut, 90], &top, 4, None).unwrap();
            w.store(&[0, cut, 0], &[5, 70 - cut, 90], &bottom, 1, None)
                .unwrap();
            let back = read(&ZarrArray::open(&dir).unwrap(), &[0, 0, 0], &[5, 70, 90], 2).unwrap();
            assert_eq!(back, bytes, "sharded: {shards}");
            assert!(w
                .store(&[0, 0, 0], &[5, 70, 90], &bytes[1..], 1, None)
                .is_err());
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn bands_cut_mid_chunk_stored_at_once_lose_nothing() {
        for shards in [false, true] {
            let dir = std::env::temp_dir()
                .join(format!("dyna_writer_bands_{}_{shards}", std::process::id()));
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
                        w.store(&[0, y0, 0], &[5, rows, 90], &v, 2, None).unwrap();
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

    #[test]
    fn a_shard_written_inner_chunk_by_chunk_has_no_dead_space() {
        let dir = std::env::temp_dir().join(format!("dyna_writer_inner_{}", std::process::id()));
        let whole =
            std::env::temp_dir().join(format!("dyna_writer_inner_w_{}", std::process::id()));
        for d in [&dir, &whole] {
            let _ = std::fs::remove_dir_all(d);
            std::fs::create_dir_all(d).unwrap();
            create(d, true);
        }
        let all: Vec<u8> = (0..5 * 70 * 90u64)
            .flat_map(|i| ((i % 3000) as u16 + 1).to_ne_bytes())
            .collect();
        ZarrWriter::open(&whole)
            .unwrap()
            .store(&[0, 0, 0], &[5, 70, 90], &all, 2, None)
            .unwrap();
        let w = ZarrWriter::open(&dir).unwrap();
        // inner chunks of (5, 16, 16), one at a time, in any order
        let mut boxes = Vec::new();
        for y in (0..70u64).step_by(16) {
            for x in (0..90u64).step_by(16) {
                boxes.push((y, x));
            }
        }
        boxes.reverse();
        for (y, x) in boxes {
            let (h, wd) = (16.min(70 - y), 16.min(90 - x));
            let mut v = Vec::new();
            for z in 0..5u64 {
                for yy in y..y + h {
                    let row = ((z * 70 + yy) * 90 + x) as usize * 2;
                    v.extend_from_slice(&all[row..row + wd as usize * 2]);
                }
            }
            w.store(&[0, y, x], &[5, h, wd], &v, 1, None).unwrap();
        }
        let back = read(&ZarrArray::open(&dir).unwrap(), &[0, 0, 0], &[5, 70, 90], 2).unwrap();
        assert_eq!(back, all);
        let size = |d: &std::path::Path| -> u64 {
            fn walk(p: &std::path::Path) -> u64 {
                std::fs::read_dir(p)
                    .unwrap()
                    .map(|e| {
                        let e = e.unwrap();
                        if e.file_type().unwrap().is_dir() {
                            walk(&e.path())
                        } else {
                            e.metadata().unwrap().len()
                        }
                    })
                    .sum()
            }
            walk(d)
        };
        let (a, b) = (size(&dir), size(&whole));
        assert!(a <= b + b / 50, "{a} bytes chunk by chunk, {b} whole");
        for d in [&dir, &whole] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn appended_shards_read_back_and_are_no_larger() {
        let dir = std::env::temp_dir().join(format!("dyna_writer_app_{}", std::process::id()));
        let whole = std::env::temp_dir().join(format!("dyna_writer_app_w_{}", std::process::id()));
        for d in [&dir, &whole] {
            let _ = std::fs::remove_dir_all(d);
            std::fs::create_dir_all(d).unwrap();
            create(d, true);
        }
        // zeros (the fill) in the first rows: inner chunks there are not written
        let all: Vec<u8> = (0..5 * 70 * 90u64)
            .flat_map(|i| {
                let y = (i / 90) % 70;
                (if y < 16 { 0u16 } else { (i % 3000) as u16 + 1 }).to_ne_bytes()
            })
            .collect();
        ZarrWriter::open(&whole)
            .unwrap()
            .store(&[0, 0, 0], &[5, 70, 90], &all, 2, None)
            .unwrap();
        let w = ZarrWriter::open(&dir).unwrap();
        assert!(w.appends());
        let mut boxes = Vec::new();
        for y in (0..70u64).step_by(16) {
            for x in (0..90u64).step_by(16) {
                boxes.push((y, x));
            }
        }
        boxes.reverse();
        for (y, x) in boxes {
            let (h, wd) = (16.min(70 - y), 16.min(90 - x));
            let mut v = Vec::new();
            for z in 0..5u64 {
                for yy in y..y + h {
                    let row = ((z * 70 + yy) * 90 + x) as usize * 2;
                    v.extend_from_slice(&all[row..row + wd as usize * 2]);
                }
            }
            w.append(&[0, y, x], &[5, h, wd], &v).unwrap();
        }
        w.finish_shards().unwrap();
        let back = read(&ZarrArray::open(&dir).unwrap(), &[0, 0, 0], &[5, 70, 90], 2).unwrap();
        assert_eq!(back, all);
        let size = |d: &std::path::Path| -> u64 {
            fn walk(p: &std::path::Path) -> u64 {
                std::fs::read_dir(p)
                    .unwrap()
                    .map(|e| {
                        let e = e.unwrap();
                        if e.file_type().unwrap().is_dir() {
                            walk(&e.path())
                        } else {
                            e.metadata().unwrap().len()
                        }
                    })
                    .sum()
            }
            walk(d)
        };
        assert!(
            size(&dir) <= size(&whole) + size(&whole) / 50,
            "{} vs {}",
            size(&dir),
            size(&whole)
        );
        // not one inner chunk, or twice: refused
        let w2 = ZarrWriter::open(&dir).unwrap();
        assert!(w2
            .append(&[0, 8, 0], &[5, 16, 16], &vec![0u8; 5 * 16 * 16 * 2])
            .is_err());
        for d in [&dir, &whole] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
