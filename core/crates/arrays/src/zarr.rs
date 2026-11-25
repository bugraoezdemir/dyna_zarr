//! Zarr v2 / v3 arrays read with zarrs: in a local folder, or inside a finished zip.
//! A region is decoded straight into the destination (zarrs' decode-into), at most
//! `threads` chunks (inner chunks, when sharded) at once.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use zarrs::array::concurrency::concurrency_chunks_and_codec;
use zarrs::array::{
    Array, ArrayBytesFixedDisjointView, ArrayCodecTraits, ArrayMetadata, ArrayShardedExt,
    ArraySubset, CodecOptions,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::plugin::{ExtensionName, ZarrVersion};
use zarrs::storage::{ReadableStorage, StoreKey};
use zarrs_zip::ZipStorageAdapter;

use crate::dtype::DType;
use crate::out::Out;
use crate::pool::{install, new_pool};
use crate::blocks::Blocks;
use crate::cache::ChunkCache;
use crate::source::{bytes_of, c_strides, region, Source, SMALL};
use crate::ReadError;

/// Small allocations per chunk decoded (keys, subsets, codec state), on top of its
/// encoded and decoded bytes.
const PER_CHUNK_EXTRA: u64 = 16 << 10;

/// A Zarr array, opened read-only.
pub struct ZarrArray {
    array: Array<dyn zarrs::storage::ReadableStorageTraits>,
    label: String,
    shape: Vec<u64>,
    dtype: DType,
    chunks: Vec<u64>,
    /// What is decoded at once: the inner chunk when sharded, else the chunk.
    unit: Vec<u64>,
    sharded: bool,
    zarr_format: u8,
    /// Its id in a write's chunk cache.
    id: u64,
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
        let chunks: Vec<u64> = array
            .chunk_shape(&vec![0u64; shape.len()])
            .map_err(|e| ReadError::new(format!("{label}: no chunk shape: {e}")))?
            .iter()
            .map(|c| c.get())
            .collect();
        let zarr_format = match array.metadata() {
            ArrayMetadata::V2(_) => 2,
            ArrayMetadata::V3(_) => 3,
        };
        let sharded = array.is_sharded();
        let unit = match array.effective_subchunk_shape() {
            Some(s) if sharded => s.iter().map(|c| c.get()).collect(),
            _ => chunks.clone(),
        };
        Ok(Self {
            array,
            label,
            shape,
            dtype,
            chunks,
            unit,
            sharded,
            zarr_format,
            id: crate::cache::new_id(),
        })
    }

    /// zarrs' options for a read with `threads`: exactly `threads` as the target, and
    /// no minimum of chunks at once (zarrs' default of 4 would run 4 chunks, on rayon's
    /// global pool, even for one thread).
    fn options(threads: usize) -> CodecOptions {
        // CodecOptions::default() asks rayon for its thread count, which outside a pool
        // starts rayon's global pool: made once, on a pool's thread
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

    /// Whether the region `start .. start + r` contains a whole chunk (an edge chunk
    /// counts as whole up to the array's end).
    fn covers_a_chunk(&self, start: &[u64], r: &[u64]) -> bool {
        (0..r.len()).all(|i| {
            let (a, b, c) = (start[i], start[i] + r[i], self.chunks[i]);
            let first = a.div_ceil(c) * c;
            first < self.shape[i] && (first + c).min(self.shape[i]) <= b
        })
    }

    /// Chunks of `grid` the region `start .. start + r` touches.
    fn touched(start: &[u64], r: &[u64], grid: &[u64]) -> u64 {
        (0..r.len())
            .map(|i| (start[i] + r[i] - 1) / grid[i] - start[i] / grid[i] + 1)
            .product()
    }

    /// The region `start .. start + r` decoded straight into `out` (zarrs'
    /// decode-into), at most `threads` units at once.
    fn read_direct(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = r.to_vec();
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
        let options = Self::options(threads);
        // one thread: on the calling thread, with no rayon work at all
        let result = if threads <= 1 {
            self.array
                .retrieve_array_subset_into_opt(&subset, (&mut view).into(), &options)
        } else {
            install(threads, || {
                self.array
                    .retrieve_array_subset_into_opt(&subset, (&mut view).into(), &options)
            })?
        };
        result.map_err(|e| ReadError::new(format!("{}: {e}", self.label)))
    }


    /// The region through `cache`: per unit it touches (the inner chunk when sharded),
    /// the whole unit from the cache or decoded into it, and the region's part copied
    /// out. At most `threads` units decoded at once, each with one thread.
    fn read_cached(
        &self,
        cache: &ChunkCache,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let n = r.len();
        let item = self.dtype.size();
        let parts = Blocks::aligned(start, r, &self.unit);
        crate::pool::pieces(parts.len(), threads, |k, _| {
            let b = parts.get(k);
            let first: Vec<u64> = (0..n).map(|i| start[i] + b[i].0).collect();
            let index: Vec<u64> = (0..n).map(|i| first[i] / self.unit[i]).collect();
            let origin: Vec<u64> = (0..n).map(|i| index[i] * self.unit[i]).collect();
            let ushape: Vec<u64> = (0..n)
                .map(|i| self.unit[i].min(self.shape[i] - origin[i]))
                .collect();
            let bytes = bytes_of(&ushape, item);
            let data = cache.get_or_decode((self.id, index), bytes, || {
                let mut buf = vec![0u8; bytes as usize];
                {
                    let o = Out::new(&mut buf, &ushape, &vec![0; n], self.dtype)?;
                    self.read_direct(&origin, &ushape, &o, 1)?;
                }
                Ok(buf)
            })?;
            let strides = c_strides(&ushape, item);
            let offset: usize = (0..n)
                .map(|i| (first[i] - origin[i]) as usize * strides[i])
                .sum();
            let sub: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
            let at: Vec<u64> = (0..n).map(|i| out.at[i] + b[i].0).collect();
            out.moved(at)
                .copy_from(&sub, &data[offset..], &strides, item, None);
            Ok(())
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

    /// What one decode yields: the inner chunk when sharded, else the chunk. Views above
    /// read in blocks of about one such unit, at most `threads` at once, so their scratch
    /// stays near a unit per thread. With the shard here, a stride or mean over a
    /// sharded array held a whole shard's input per thread (16 threads of 8 MiB shard
    /// spans: 142 MiB where the Python path held ~100 MiB in all, 2026-11-25).
    fn grid(&self) -> Option<Vec<u64>> {
        Some(self.unit.clone())
    }

    /// Inside a write whose pool has a chunk cache ([`crate::cache`]): unit by unit
    /// through it, each unit decoded whole once and its part copied. Otherwise decoded
    /// straight into the destination.
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
        match crate::cache::current() {
            Some(cache) => self.read_cached(&cache, start, &r, out, threads),
            None => self.read_direct(start, &r, out, threads),
        }
    }
    /// Per unit decoded at once: its encoded bytes (at most about its decoded size), its
    /// decoded bytes and, for a partly read unit, the part taken out of it (measured:
    /// zarrs' partial decoders extract into a new buffer). Sharded: per shard open its
    /// index and, when the region covers a whole shard, the whole encoded shard (zarrs
    /// then reads it at once). How many are open at once is zarrs' own split of
    /// `threads` into chunks x codec.
    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        if r.contains(&0) {
            return Ok(0);
        }
        let n_units = Self::touched(start, &r, &self.unit);
        let n_chunks = Self::touched(start, &r, &self.chunks);
        let t = threads.max(1);
        let (shards_open, units_open) = if t <= 1 {
            (1, 1)
        } else if n_chunks == 1 {
            (1, if self.sharded { t as u64 } else { 1 })
        } else {
            let first = vec![0u64; r.len()];
            let chunk_shape = self
                .array
                .chunk_shape(&first)
                .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
            let codec = self
                .array
                .codecs()
                .recommended_concurrency(&chunk_shape, self.array.data_type())
                .map_err(|e| ReadError::new(format!("{}: {e}", self.label)))?;
            let (outer, inner) =
                concurrency_chunks_and_codec(t, n_chunks as usize, &Self::options(t), &codec);
            let outer = outer.max(1) as u64;
            let inner = inner.concurrent_target().max(1) as u64;
            (outer, if self.sharded { outer * inner } else { outer })
        };
        let units_open = units_open.min(n_units);
        let unit_bytes = bytes_of(&self.unit, self.dtype.size());
        let mut cost = units_open * (3 * unit_bytes + PER_CHUNK_EXTRA);
        // the index: two u64 per inner chunk, read and decoded
        let per_shard: u64 = (0..r.len())
            .map(|i| self.chunks[i].div_ceil(self.unit[i]))
            .product();
        let index = if self.sharded { per_shard * 16 * 2 } else { 0 };
        // through a write's chunk cache: up to `threads` units decoded at once, each on
        // its own (a sharded unit reads its shard's index again); the cached units
        // themselves are the write's, counted once in its budget
        let cached = (t as u64).min(n_units) * (3 * unit_bytes + PER_CHUNK_EXTRA + index);
        if self.sharded {
            let shards_open = shards_open.min(n_chunks);
            cost += shards_open * index;
            if self.covers_a_chunk(start, &r) {
                // encoded, at most about its decoded size plus a header per inner chunk
                let shard = bytes_of(&self.chunks, self.dtype.size());
                cost += shards_open * (shard + shard / 64 + per_shard * 64);
            }
        }
        Ok(cost.max(cached) + SMALL)
    }
}
