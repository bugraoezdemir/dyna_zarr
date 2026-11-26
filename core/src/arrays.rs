//! Python bindings of `dyna-zarr-arrays` (DR3): `dyna_zarr_core.Array`, a lazy array
//! owned by Rust (a Zarr array or a view over arrays), read into NumPy arrays.
//!
//! A read writes into the memory of a NumPy array (given, or made for it): Python
//! hands its buffer over for the call, the GIL is released and the Rust reader writes the
//! region in place. Errors raise `dyna_zarr_core.ReadError`, a ValueError.

use std::path::PathBuf;
use std::sync::Arc;

use dyna_zarr_arrays as ar;
use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyEllipsis, PySlice, PyTuple};

create_exception!(dyna_zarr_core, ReadError, PyValueError);
// The engine cannot give dyna's Python-path values for this read (they depend on where
// the read starts): dyna reads it with the Python path.
create_exception!(dyna_zarr_core, UsePythonPath, ReadError);

fn err(e: ar::ReadError) -> PyErr {
    if e.python_path {
        UsePythonPath::new_err(e.message)
    } else {
        ReadError::new_err(e.message)
    }
}

/// A fixed set of threads that reads run on (Workplan decision DR-5). A write makes one
/// with its max_threads and runs every region's read on it: the threads busy at once
/// never exceed `threads`, however many reads share it. A read's own `threads` caps its
/// memory (how many of its pieces are open at once), not the pool.
///
/// `cache_mb` > 0 gives the pool a cache of decoded chunks of at most that many MiB,
/// shared by every read on it (a write over a filter: neighbouring regions' halos are
/// then decoded once). The write counts it once in its memory budget.
#[pyclass(frozen, module = "dyna_zarr_core")]
pub struct Pool {
    pool: Arc<ar::pool::ThreadPool>,
    threads: usize,
    cache: Option<Arc<ar::cache::ChunkCache>>,
}

#[pymethods]
impl Pool {
    #[new]
    #[pyo3(signature = (threads, cache_mb = 0.0))]
    fn new(threads: usize, cache_mb: f64) -> PyResult<Self> {
        if threads < 1 {
            return Err(PyValueError::new_err("a pool needs at least 1 thread"));
        }
        if cache_mb.is_nan() || cache_mb < 0.0 {
            return Err(PyValueError::new_err("cache_mb must be 0 or more"));
        }
        let bytes = (cache_mb * 1048576.0) as u64;
        let cache = (bytes > 0).then(|| ar::cache::ChunkCache::new(bytes));
        Ok(Self {
            pool: ar::pool::new_pool_cached(threads, cache.clone()).map_err(err)?,
            threads,
            cache,
        })
    }

    #[getter]
    fn threads(&self) -> usize {
        self.threads
    }

    /// The chunk cache's limit in bytes (0: no cache).
    #[getter]
    fn cache_bytes(&self) -> u64 {
        self.cache.as_ref().map_or(0, |c| c.limit())
    }

    /// (chunks served from the cache, chunks decoded) by reads on this pool so far.
    fn cache_counts(&self) -> (u64, u64) {
        self.cache.as_ref().map_or((0, 0), |c| c.counts())
    }

    fn __repr__(&self) -> String {
        match &self.cache {
            Some(c) => format!(
                "<dyna_zarr_core.Pool {} threads, cache {} MiB>",
                self.threads,
                c.limit() >> 20
            ),
            None => format!("<dyna_zarr_core.Pool {} threads>", self.threads),
        }
    }
}

/// A Zarr array in a local folder, opened for writing with zarrs (Workplan DR8): dyna
/// creates the array (its metadata), the writer encodes and stores regions of it.
#[pyclass(frozen, module = "dyna_zarr_core")]
pub struct Writer {
    writer: Arc<ar::ZarrWriter>,
}

#[pymethods]
impl Writer {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let w = py.detach(|| ar::ZarrWriter::open(&path)).map_err(err)?;
        Ok(Self {
            writer: Arc::new(w),
        })
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.writer.shape())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.writer.dtype().name()
    }

    /// The write unit: the shard when sharded, else the chunk.
    #[getter]
    fn unit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.writer.unit())
    }

    /// Store `data` (a C-contiguous NumPy array of the array's dtype) at `start`, with
    /// `threads` (on `pool` when given), without the GIL. Regions should cover whole
    /// write units: one covering a unit in part rewrites it, which is safe only while no
    /// other region writes that unit at the same time.
    #[pyo3(signature = (start, data, threads = 1, pool = None))]
    fn store(
        &self,
        py: Python<'_>,
        start: Vec<u64>,
        data: Bound<'_, PyAny>,
        threads: usize,
        pool: Option<PyRef<'_, Pool>>,
    ) -> PyResult<()> {
        if threads < 1 {
            return Err(PyValueError::new_err("threads must be >= 1"));
        }
        let dtype = self.writer.dtype();
        let name: String = data.getattr("dtype")?.getattr("name")?.extract()?;
        if name != dtype.name() {
            return Err(PyTypeError::new_err(format!(
                "data is {name}; the array is {}",
                dtype.name()
            )));
        }
        let native: bool = data.getattr("dtype")?.getattr("isnative")?.extract()?;
        let contiguous: bool = data.getattr("flags")?.getattr("c_contiguous")?.extract()?;
        if !native || !contiguous {
            return Err(PyValueError::new_err(
                "data must be a C-contiguous NumPy array in native byte order",
            ));
        }
        let shape: Vec<u64> = data.getattr("shape")?.extract()?;
        if shape.len() != start.len() {
            return Err(PyValueError::new_err(format!(
                "data has {} axes, start {}",
                shape.len(),
                start.len()
            )));
        }
        // the array's bytes, kept alive (and unresizable) while they are stored
        let bytes = data
            .call_method1("reshape", (-1,))?
            .call_method1("view", ("u1",))?;
        let buf: PyBuffer<u8> = PyBuffer::get(&bytes)?;
        let (ptr, len) = (buf.buf_ptr() as usize, buf.len_bytes());
        let w = self.writer.clone();
        let pool = pool.map(|p| p.pool.clone());
        py.detach(move || {
            // SAFETY: the buffer export above keeps the memory alive and unchanged in
            // size for the call; it is only read
            let b = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
            // on this (Python) thread: the writer takes its unit locks before the pool
            w.store(&start, &shape, b, threads, pool.as_deref())
        })
        .map_err(err)?;
        drop(buf);
        Ok(())
    }

    /// Whether `append` can append inner chunks to this array's shards (a sharded array
    /// with its index at the end).
    #[getter]
    fn appends(&self) -> bool {
        self.writer.appends()
    }

    /// Append the inner chunk `data` (cut at the array's end) at `start` to its shard;
    /// the shard's index is written when its last inner chunk is in (or by
    /// `finish_shards`). Without the GIL.
    fn append(&self, py: Python<'_>, start: Vec<u64>, data: Bound<'_, PyAny>) -> PyResult<()> {
        let dtype = self.writer.dtype();
        let name: String = data.getattr("dtype")?.getattr("name")?.extract()?;
        let native: bool = data.getattr("dtype")?.getattr("isnative")?.extract()?;
        let contiguous: bool = data.getattr("flags")?.getattr("c_contiguous")?.extract()?;
        if name != dtype.name() || !native || !contiguous {
            return Err(PyValueError::new_err(format!(
                "data must be a C-contiguous {} NumPy array in native byte order",
                dtype.name()
            )));
        }
        let shape: Vec<u64> = data.getattr("shape")?.extract()?;
        let bytes = data
            .call_method1("reshape", (-1,))?
            .call_method1("view", ("u1",))?;
        let buf: PyBuffer<u8> = PyBuffer::get(&bytes)?;
        let (ptr, len) = (buf.buf_ptr() as usize, buf.len_bytes());
        let w = self.writer.clone();
        py.detach(move || {
            // SAFETY: the buffer export above keeps the memory alive and unchanged in
            // size for the call; it is only read
            let b = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
            w.append(&start, &shape, b)
        })
        .map_err(err)?;
        drop(buf);
        Ok(())
    }

    /// Close every shard still open (its index written).
    fn finish_shards(&self, py: Python<'_>) -> PyResult<()> {
        let w = self.writer.clone();
        py.detach(move || w.finish_shards()).map_err(err)
    }

    fn __repr__(&self) -> String {
        format!(
            "<dyna_zarr_core.Writer {:?} {}>",
            self.writer.shape(),
            self.writer.dtype().name()
        )
    }
}

/// Bytes the engine has allocated now (Rust's heap; NumPy arrays are not included).
#[pyfunction]
pub fn memory_in_use() -> usize {
    ar::memory::in_use()
}

/// The most bytes the engine had allocated at once since `reset_memory_peak()`.
#[pyfunction]
pub fn memory_peak() -> usize {
    ar::memory::peak()
}

/// Start a new peak from what is allocated now; returns that. The counts are
/// process-wide: measure one read at a time.
#[pyfunction]
pub fn reset_memory_peak() -> usize {
    ar::memory::reset_peak()
}

/// A lazy array read by dyna's engine: a Zarr array, or a view over arrays.
#[pyclass(frozen, module = "dyna_zarr_core")]
pub struct Array {
    src: ar::SourceRef,
    zarr: Option<Arc<ar::ZarrArray>>,
}

impl Array {
    fn view(src: impl ar::Source + 'static) -> Self {
        Self {
            src: Arc::new(src),
            zarr: None,
        }
    }

    fn zarr(a: ar::ZarrArray) -> Self {
        let a = Arc::new(a);
        Self {
            src: a.clone(),
            zarr: Some(a),
        }
    }

    /// Read `start .. stop` into `out` (or a new array) at `at`.
    #[allow(clippy::too_many_arguments)]
    fn read_region(
        &self,
        py: Python<'_>,
        start: Vec<u64>,
        stop: Vec<u64>,
        out: Option<Bound<'_, PyAny>>,
        at: Option<Vec<u64>>,
        threads: usize,
        pool: Option<Arc<ar::pool::ThreadPool>>,
    ) -> PyResult<Py<PyAny>> {
        if threads < 1 {
            return Err(PyValueError::new_err("threads must be >= 1"));
        }
        let region = ar::region(self.src.shape(), &start, &stop).map_err(err)?;
        let dtype = self.src.dtype();
        let np = py.import("numpy")?;
        let out = match out {
            Some(o) => o,
            None => np.call_method1("empty", (region.clone(), dtype.name()))?,
        };
        let name: String = out.getattr("dtype")?.getattr("name")?.extract()?;
        if name != dtype.name() {
            return Err(PyTypeError::new_err(format!(
                "out is {name}; this array is {}",
                dtype.name()
            )));
        }
        let shape: Vec<u64> = out.getattr("shape")?.extract()?;
        let flags = out.getattr("flags")?;
        if !flags.getattr("c_contiguous")?.extract::<bool>()?
            || !flags.getattr("writeable")?.extract::<bool>()?
        {
            return Err(PyValueError::new_err(
                "out must be a writable C-contiguous NumPy array",
            ));
        }
        let at = at.unwrap_or_else(|| vec![0; shape.len()]);
        // the array's bytes, as a buffer kept alive (and unresizable) during the read;
        // flattened first: NumPy cannot view a 0-d array as bytes, and for a C-contiguous
        // array the flat array is a view, not a copy
        let bytes = out
            .call_method1("reshape", (-1,))?
            .call_method1("view", ("u1",))?;
        let buf: PyBuffer<u8> = PyBuffer::get(&bytes)?;
        if buf.readonly() {
            return Err(PyValueError::new_err("out is read-only"));
        }
        let (ptr, len) = (buf.buf_ptr() as usize, buf.len_bytes());
        let src = self.src.clone();
        py.detach(move || {
            // the read's pieces share the files they open (local arrays: ar::ReadStore)
            let mut stores = Vec::new();
            src.stores(&mut stores);
            let _held: Vec<_> = stores.iter().map(|s| s.reading()).collect();
            let run = move || {
                // SAFETY: the buffer export above keeps the memory alive and in place for
                // the call; the region written lies inside it (Out::new and read_into
                // check).
                let bytes = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
                let dst = ar::Out::new(bytes, &shape, &at, dtype)?;
                src.read_into(&start, &stop, &dst, threads)
            };
            // the given pool; else the shared pool of `threads`; one thread: right here
            ar::pool::enter(threads, pool.as_deref(), run)?
        })
        .map_err(err)?;
        drop(buf);
        Ok(out.unbind())
    }
}

#[pymethods]
impl Array {
    /// The Zarr array (v2 or v3) in the local folder `path`.
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Ok(Self::zarr(
            py.detach(|| ar::ZarrArray::open(&path)).map_err(err)?,
        ))
    }

    /// The Zarr array at `inner` ("" for the root) inside the finished zip file `zip`.
    #[staticmethod]
    #[pyo3(signature = (zip, inner = ""))]
    fn open_zip(py: Python<'_>, zip: PathBuf, inner: &str) -> PyResult<Self> {
        let inner = inner.to_string();
        Ok(Self::zarr(
            py.detach(|| ar::ZarrArray::open_zip(&zip, &inner))
                .map_err(err)?,
        ))
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.src.shape())
    }

    /// Whether reading it reads a halo around every region (a filter or a resampling
    /// in the chain): a write over it gives its pool a chunk cache.
    #[getter]
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.src.shape().len()
    }

    /// The NumPy name of the element type.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.src.dtype().name()
    }

    /// The storage grid underneath (the chunk or shard shape); for a view, that grid in
    /// the view's elements (a downsampled level: divided by the factors), or None.
    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.src.grid().map(|g| PyTuple::new(py, g)).transpose()
    }

    /// 2 or 3 for a Zarr array, None for a view.
    #[getter]
    fn zarr_format(&self) -> Option<u8> {
        self.zarr.as_ref().map(|z| z.zarr_format())
    }

    /// The stored metadata (JSON text) of a Zarr array, None for a view.
    #[getter]
    fn metadata(&self) -> Option<String> {
        self.zarr.as_ref().map(|z| z.metadata_json())
    }

    fn __len__(&self) -> PyResult<usize> {
        self.src
            .shape()
            .first()
            .map(|n| *n as usize)
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d array"))
    }

    fn __repr__(&self) -> String {
        let kind = if self.zarr.is_some() { "zarr" } else { "view" };
        format!(
            "<dyna_zarr_core.Array {kind} {:?} {}>",
            self.src.shape(),
            self.src.dtype().name()
        )
    }

    /// Read the region `start .. stop` into `out` at `at` (a writable C-contiguous NumPy
    /// array of this dtype; a new one of the region's shape when None), without the GIL.
    /// Returns `out`.
    ///
    /// `threads` caps the read's memory: at most that many of its buffer-holding pieces
    /// at once, which `cost(start, stop, threads)` bounds. It runs on `pool` (a
    /// `Pool`: a write's threads, shared by all its reads); without one, on the shared
    /// pool of `threads` threads, and with `threads=1` on the calling thread.
    #[pyo3(signature = (start, stop, out = None, at = None, threads = 1, pool = None))]
    #[allow(clippy::too_many_arguments)]
    fn read(
        &self,
        py: Python<'_>,
        start: Vec<u64>,
        stop: Vec<u64>,
        out: Option<Bound<'_, PyAny>>,
        at: Option<Vec<u64>>,
        threads: usize,
        pool: Option<PyRef<'_, Pool>>,
    ) -> PyResult<Py<PyAny>> {
        let pool = pool.map(|p| p.pool.clone());
        self.read_region(py, start, stop, out, at, threads, pool)
    }

    /// The most bytes reading `start .. stop` with `threads` holds at once, beyond the
    /// destination: an upper bound for planning memory (Workplan decision DR-5).
    #[pyo3(signature = (start, stop, threads = 1))]
    fn cost(&self, start: Vec<u64>, stop: Vec<u64>, threads: usize) -> PyResult<u64> {
        if threads < 1 {
            return Err(PyValueError::new_err("threads must be >= 1"));
        }
        ar::read_cost(self.src.as_ref(), &start, &stop, threads).map_err(err)
    }

    /// NumPy-style basic indexing (ints, slices with a positive step, Ellipsis, None),
    /// read at once on one thread.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let view = self.index(key)?;
        let shape = view.src.shape().to_vec();
        view.read_region(py, vec![0; shape.len()], shape, None, None, 1, None)
    }

    /// The lazy view `self[key]` (basic indexing as __getitem__, nothing read).
    fn index(&self, key: &Bound<'_, PyAny>) -> PyResult<Array> {
        let items: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
            Ok(t) => t.iter().collect(),
            Err(_) => vec![key.clone()],
        };
        let shape = self.src.shape().to_vec();
        let n = shape.len();
        let used = items
            .iter()
            .filter(|k| !k.is_none() && !k.is_instance_of::<PyEllipsis>())
            .count();
        if used > n {
            return Err(PyIndexError::new_err(format!(
                "{used} indices for an array of {n} axes"
            )));
        }
        let (mut start, mut stop, mut step) = (vec![0u64; n], shape.clone(), vec![1u64; n]);
        let mut squeeze = Vec::new(); // source axes indexed by an int
        let mut new_axes = Vec::new(); // positions (in the result) of None
        let mut axis = 0usize;
        let mut out_axis = 0usize;
        let mut seen_ellipsis = false;
        for k in &items {
            if k.is_instance_of::<PyEllipsis>() {
                if seen_ellipsis {
                    return Err(PyIndexError::new_err("only one Ellipsis is allowed"));
                }
                seen_ellipsis = true;
                let skip = n - used;
                axis += skip;
                out_axis += skip;
            } else if k.is_none() {
                new_axes.push(out_axis);
                out_axis += 1;
            } else if let Ok(s) = k.cast::<PySlice>() {
                let len = shape[axis] as isize;
                let ind = s.indices(len)?;
                if ind.step <= 0 {
                    return Err(PyIndexError::new_err(
                        "only positive slice steps are supported",
                    ));
                }
                start[axis] = ind.start as u64;
                step[axis] = ind.step as u64;
                stop[axis] = if ind.slicelength == 0 {
                    start[axis].min(shape[axis])
                } else {
                    start[axis] + (ind.slicelength as u64 - 1) * step[axis] + 1
                };
                if ind.slicelength == 0 {
                    start[axis] = stop[axis];
                }
                axis += 1;
                out_axis += 1;
            } else if let Ok(i) = k.extract::<i64>() {
                let len = shape[axis] as i64;
                let j = if i < 0 { i + len } else { i };
                if j < 0 || j >= len {
                    return Err(PyIndexError::new_err(format!(
                        "index {i} is out of bounds for axis {axis} of size {len}"
                    )));
                }
                start[axis] = j as u64;
                stop[axis] = j as u64 + 1;
                squeeze.push(axis);
                axis += 1;
            } else {
                return Err(PyTypeError::new_err(format!(
                    "index {k}: only ints, slices, Ellipsis and None are supported"
                )));
            }
        }
        // a filter under a stepped slice computes only the kept elements (ar::sliced)
        let mut src: ar::SourceRef =
            ar::sliced(self.src.clone(), &start, &stop, &step).map_err(err)?;
        for &a in squeeze.iter().rev() {
            src = Arc::new(ar::Squeeze::new(src, a).map_err(err)?);
        }
        for &a in &new_axes {
            src = Arc::new(ar::ExpandDims::new(src, a).map_err(err)?);
        }
        Ok(Array { src, zarr: None })
    }

    /// Output axis i is axis `perm[i]` (default: reversed).
    #[pyo3(signature = (perm = None))]
    fn transpose(&self, perm: Option<Vec<usize>>) -> PyResult<Array> {
        let perm = perm.unwrap_or_else(|| (0..self.src.shape().len()).rev().collect());
        Ok(Self::view(
            ar::Transpose::new(self.src.clone(), &perm).map_err(err)?,
        ))
    }

    /// A new axis of length 1 at `axis`.
    fn expand_dims(&self, axis: usize) -> PyResult<Array> {
        Ok(Self::view(
            ar::ExpandDims::new(self.src.clone(), axis).map_err(err)?,
        ))
    }

    /// Axis `axis`, of length 1, removed.
    fn squeeze(&self, axis: usize) -> PyResult<Array> {
        Ok(Self::view(
            ar::Squeeze::new(self.src.clone(), axis).map_err(err)?,
        ))
    }

    /// The elements converted to `dtype` (a NumPy name).
    fn astype(&self, dtype: &str) -> PyResult<Array> {
        let to = ar::DType::parse(dtype).map_err(err)?;
        Ok(Self::view(
            ar::Cast::new(self.src.clone(), to).map_err(err)?,
        ))
    }

    /// The array downsampled by integer `factors` (one per axis), ceil-sized as ozp's
    /// levels. `method`: "mean" (exactly tensorstore.downsample's), "stride" (every
    /// f-th element; ozp's "simple") or "gaussian" (a Gaussian low-pass, then every
    /// f-th sample; `sigma` per axis, default f / 3 on downscaled axes; `mode`
    /// "reflect", "mirror", "nearest" or "constant" with `cval`; `truncate` in sigmas,
    /// as scipy.ndimage.gaussian_filter). Lazy: nothing is read until asked.
    #[pyo3(signature = (factors, method = "mean", sigma = None, mode = "reflect", cval = 0.0, truncate = 4.0, max_read_mb = None))]
    #[allow(clippy::too_many_arguments)]
    fn downsample(
        &self,
        factors: Vec<u64>,
        method: &str,
        sigma: Option<Vec<f64>>,
        mode: &str,
        cval: f64,
        truncate: f64,
        max_read_mb: Option<f64>,
    ) -> PyResult<Array> {
        let bytes = max_read_mb.map(|mb| (mb * 1048576.0) as u64);
        match method {
            "stride" | "simple" => {
                ar::stride(self.src.clone(), &factors).map_err(err)?; // checks the factors
                let n = factors.len();
                let src = ar::sliced(self.src.clone(), &vec![0; n], self.src.shape(), &factors)
                    .map_err(err)?;
                Ok(Self { src, zarr: None })
            }
            "mean" => {
                let mut m = ar::Mean::new(self.src.clone(), &factors).map_err(err)?;
                if let Some(b) = bytes {
                    m = m.with_max_read(b);
                }
                Ok(Self::view(m))
            }
            "gaussian" => {
                let boundary = ar::Boundary::parse(mode, cval).ok_or_else(|| {
                    ReadError::new_err(format!(
                        "mode {mode:?}: use \"reflect\", \"mirror\", \"nearest\" or \"constant\""
                    ))
                })?;
                let mut g = ar::Gaussian::new(
                    self.src.clone(),
                    &factors,
                    sigma.as_deref(),
                    boundary,
                    truncate,
                )
                .map_err(err)?;
                if let Some(b) = bytes {
                    g = g.with_max_read(b);
                }
                Ok(Self::view(g))
            }
            other => Err(ReadError::new_err(format!(
                "method {other:?}: use \"mean\", \"stride\" or \"gaussian\""
            ))),
        }
    }

    /// A separable filter (Workplan DR7): `passes` applied in order, each
    /// `(axis, weights, mode)` a scipy.ndimage.correlate1d along one axis with scipy's
    /// values, in this array's element type after every pass (as scipy.ndimage). `mode`
    /// is reflect, mirror, nearest, wrap or constant (filled with `cval`). Any region
    /// gives the values of the whole filtered array. Lazy: nothing is read until asked.
    #[pyo3(signature = (passes, cval = 0.0, max_read_mb = None))]
    fn separable(
        &self,
        passes: Vec<(usize, Vec<f64>, String)>,
        cval: f64,
        max_read_mb: Option<f64>,
    ) -> PyResult<Array> {
        let passes = passes
            .into_iter()
            .map(|(axis, weights, mode)| {
                let mode = ar::FilterMode::parse(&mode, cval).ok_or_else(|| {
                    ReadError::new_err(format!(
                        "mode {mode:?}: use \"reflect\", \"mirror\", \"nearest\", \"wrap\" or \"constant\""
                    ))
                })?;
                Ok(ar::Pass::correlate(axis, weights, mode))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut f = ar::Separable::new(self.src.clone(), passes).map_err(err)?;
        if let Some(mb) = max_read_mb {
            f = f.with_max_read((mb * 1048576.0) as u64);
        }
        Ok(Self::view(f))
    }

    /// A filter of one or more chains of passes (Workplan DR7), each pass
    /// `(axis, kind, values, origin, mode)`: kind "correlate" (values: the weights),
    /// "uniform", "minimum" or "maximum" (values: [size]). Each chain runs on this array
    /// as scipy.ndimage runs a separable filter; several chains are combined by `combine`:
    /// "sum" (scipy's generic_laplace) or "hypot" (generic_gradient_magnitude, floats
    /// only). Any region gives the values of the whole filtered array. Lazy.
    #[pyo3(signature = (chains, combine = None, cval = 0.0, max_read_mb = None))]
    fn filter(
        &self,
        chains: Vec<Vec<crate::filters::PassSpec>>,
        combine: Option<&str>,
        cval: f64,
        max_read_mb: Option<f64>,
    ) -> PyResult<Array> {
        let chains = crate::filters::chains_of(chains, cval).map_err(ReadError::new_err)?;
        let combine = crate::filters::combine_of(combine).map_err(ReadError::new_err)?;
        let mut f = ar::Separable::terms(self.src.clone(), chains, combine).map_err(err)?;
        if let Some(mb) = max_read_mb {
            f = f.with_max_read((mb * 1048576.0) as u64);
        }
        Ok(Self::view(f))
    }

    /// scipy.ndimage's n-D rank filter (Workplan DR7 step 4): the element of rank `rank`
    /// among the positions of a footprint of `shape` (all of it, or those `mask` marks,
    /// C order) shifted by `origin`, the array extended at its edges by `mode`. What
    /// scipy's rank_filter, median_filter and percentile_filter run for arrays of two or
    /// more dimensions and ranks other than the first and the last. Any region gives the
    /// values of the whole filtered array. Lazy.
    #[pyo3(signature = (shape, rank, mask = None, origin = None, mode = "reflect", cval = 0.0, max_read_mb = None))]
    #[allow(clippy::too_many_arguments)]
    fn rank_filter(
        &self,
        shape: Vec<usize>,
        rank: usize,
        mask: Option<Vec<bool>>,
        origin: Option<Vec<isize>>,
        mode: &str,
        cval: f64,
        max_read_mb: Option<f64>,
    ) -> PyResult<Array> {
        let mode = ar::FilterMode::parse(mode, cval).ok_or_else(|| {
            ReadError::new_err(format!(
                "mode {mode:?}: use \"reflect\", \"mirror\", \"nearest\", \"wrap\" or \"constant\""
            ))
        })?;
        let origin = origin.unwrap_or_else(|| vec![0; shape.len()]);
        let fp = ar::Footprint {
            shape,
            mask,
            origin,
        };
        let mut f = ar::Window::rank(self.src.clone(), fp, rank, mode).map_err(err)?;
        if let Some(mb) = max_read_mb {
            f = f.with_max_read((mb * 1048576.0) as u64);
        }
        Ok(Self::view(f))
    }

    /// scipy.ndimage.correlate with an n-D kernel (Workplan DR7 step 5): `weights` of
    /// `shape` (C order) shifted by `origin`, the array extended at its edges by `mode`;
    /// `convolve=True` is scipy.ndimage.convolve (the kernel reversed, origins negated).
    /// Any region gives the values of the whole filtered array. Lazy.
    #[pyo3(signature = (shape, weights, origin = None, mode = "reflect", cval = 0.0, convolve = false, max_read_mb = None))]
    #[allow(clippy::too_many_arguments)]
    fn correlate(
        &self,
        shape: Vec<usize>,
        weights: Vec<f64>,
        origin: Option<Vec<isize>>,
        mode: &str,
        cval: f64,
        convolve: bool,
        max_read_mb: Option<f64>,
    ) -> PyResult<Array> {
        let mode = ar::FilterMode::parse(mode, cval).ok_or_else(|| {
            ReadError::new_err(format!(
                "mode {mode:?}: use \"reflect\", \"mirror\", \"nearest\", \"wrap\" or \"constant\""
            ))
        })?;
        let origin = origin.unwrap_or_else(|| vec![0; shape.len()]);
        let mut f =
            ar::Window::correlate(self.src.clone(), shape, &weights, origin, mode, convolve)
                .map_err(err)?;
        if let Some(mb) = max_read_mb {
            f = f.with_max_read((mb * 1048576.0) as u64);
        }
        Ok(Self::view(f))
    }

    /// The final labels of this mask in tiles of `tile` (tilewise-ccl's Phase B, Workplan
    /// DR7 step 8): each tile's foreground (`v > 0`) labeled as scipy.ndimage.label does
    /// with `generate_binary_structure(ndim, connectivity)`, its local label l mapped to
    /// `tables[offsets[k] + l]` for tile k (tiles in C order of the grid). `tables` and
    /// `offsets` are the bytes of int64 arrays; `dtype` "int32" or "int64". Lazy; a read
    /// labels every tile it overlaps whole.
    #[pyo3(signature = (tile, connectivity, tables, offsets, dtype = "int32"))]
    fn label_tiles(
        &self,
        tile: Vec<u64>,
        connectivity: usize,
        tables: &Bound<'_, PyBytes>,
        offsets: &Bound<'_, PyBytes>,
        dtype: &str,
    ) -> PyResult<Array> {
        let ints = |b: &Bound<'_, PyBytes>, what: &str| -> PyResult<Vec<i64>> {
            let b = b.as_bytes();
            if b.len() % 8 != 0 {
                return Err(ReadError::new_err(format!(
                    "{what}: not the bytes of int64s"
                )));
            }
            Ok(b.as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_ne_bytes(*c))
                .collect())
        };
        let tables = ints(tables, "tables")?;
        let offsets = ints(offsets, "offsets")?
            .into_iter()
            .map(|v| usize::try_from(v).map_err(|_| ReadError::new_err("a negative offset")))
            .collect::<PyResult<Vec<usize>>>()?;
        let t = ar::DType::parse(dtype).map_err(err)?;
        Ok(Self::view(
            ar::TileLabels::new(self.src.clone(), &tile, connectivity, tables, offsets, t)
                .map_err(err)?,
        ))
    }

    /// The array resampled onto a grid of `shape`: output index o takes the value at
    /// input index `matrix · o` (scipy.ndimage.affine_transform's convention: the matrix
    /// maps output to input). `matrix`: n rows of n + 1 (the last column the offset) or
    /// the full (n + 1) x (n + 1). `interpolation`: "nearest" or "linear"; `mode`:
    /// "constant" or "grid-constant" (as scipy); `fill`: the value outside the input.
    /// A request whose input would exceed `max_read_mb` is split first. Lazy: nothing is
    /// read until asked.
    #[pyo3(signature = (matrix, shape, interpolation = "linear", mode = "constant", fill = 0.0, max_read_mb = None))]
    fn resample(
        &self,
        matrix: Vec<Vec<f64>>,
        shape: Vec<u64>,
        interpolation: &str,
        mode: &str,
        fill: f64,
        max_read_mb: Option<f64>,
    ) -> PyResult<Array> {
        let n = matrix.len();
        let rows = if n > 0 && matrix.iter().all(|r| r.len() == n) {
            matrix[..n - 1].to_vec() // the full form: its last row is implied
        } else {
            matrix
        };
        let m = dyna_zarr_transforms::Matrix::from_rows(&rows)
            .map_err(|e| ReadError::new_err(e.message))?;
        let interp =
            dyna_zarr_transforms::Interpolation::parse(interpolation).ok_or_else(|| {
                ReadError::new_err(format!(
                    "interpolation {interpolation:?}: use \"nearest\" or \"linear\""
                ))
            })?;
        let edge = ar::EdgeMode::parse(mode).ok_or_else(|| {
            ReadError::new_err(format!(
                "mode {mode:?}: use \"constant\" or \"grid-constant\""
            ))
        })?;
        let mut r =
            ar::Resample::new(self.src.clone(), &shape, m, interp, edge, fill).map_err(err)?;
        if let Some(mb) = max_read_mb {
            r = r.with_max_read((mb * 1048576.0) as u64);
        }
        Ok(Self::view(r))
    }
}

fn sources(arrays: Vec<PyRef<'_, Array>>) -> Vec<ar::SourceRef> {
    arrays.iter().map(|a| a.src.clone()).collect()
}

/// The arrays joined along `axis`.
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn concatenate(arrays: Vec<PyRef<'_, Array>>, axis: usize) -> PyResult<Array> {
    Ok(Array::view(
        ar::Concat::new(sources(arrays), axis).map_err(err)?,
    ))
}

/// The arrays stacked along a new axis `axis`.
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn stack(arrays: Vec<PyRef<'_, Array>>, axis: usize) -> PyResult<Array> {
    Ok(Array::view(
        ar::Concat::stack(sources(arrays), axis).map_err(err)?,
    ))
}

/// One operand of a pointwise operation: an Array, or the bytes of one element.
fn operand(o: &Bound<'_, PyAny>) -> PyResult<ar::Operand> {
    if let Ok(a) = o.cast::<Array>() {
        return Ok(ar::Operand::Array(a.borrow().src.clone()));
    }
    if let Ok(b) = o.cast::<PyBytes>() {
        return Ok(ar::Operand::Scalar(b.as_bytes().to_vec()));
    }
    Err(PyTypeError::new_err(
        "a pointwise operand is an Array or the bytes of one element",
    ))
}

/// `op` (a NumPy ufunc name, "where" or "clip") in the loop type `loop_dtype` on
/// `operands`, with NumPy's values (Workplan DR7 step 7): each operand an Array (all of
/// one shape; converted to the loop type when read, `where`'s condition to bool) or the
/// bytes of one element of the loop type. The result has the loop type, or bool for the
/// comparisons, the logical operations and the tests. A read above `max_read_mb` of
/// operands and result is split first. Lazy.
#[pyfunction]
#[pyo3(signature = (op, operands, loop_dtype, max_read_mb = None))]
pub fn pointwise(
    op: &str,
    operands: Vec<Bound<'_, PyAny>>,
    loop_dtype: &str,
    max_read_mb: Option<f64>,
) -> PyResult<Array> {
    let t = ar::DType::parse(loop_dtype).map_err(err)?;
    let ops = operands.iter().map(operand).collect::<PyResult<Vec<_>>>()?;
    let mut p = ar::Pointwise::new(op, ops, t).map_err(err)?;
    if let Some(mb) = max_read_mb {
        p = p.with_max_read((mb * 1048576.0) as u64);
    }
    Ok(Array::view(p))
}

/// The pointwise kernel on bytes, without arrays: `op` in `loop_dtype` on `n`
/// elements, each operand `(bytes, scalar)` (n elements of its type, or one when
/// `scalar`). The result's bytes. What dyna checks the engine with against NumPy.
#[pyfunction]
pub fn pointwise_apply<'py>(
    py: Python<'py>,
    op: &str,
    operands: Vec<(Bound<'py, PyBytes>, bool)>,
    n: usize,
    loop_dtype: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let t = ar::DType::parse(loop_dtype).map_err(err)?;
    let ops: Vec<(&[u8], bool)> = operands.iter().map(|(b, s)| (b.as_bytes(), *s)).collect();
    let out = ar::pointwise_apply(op, t, &ops, n).map_err(err)?;
    Ok(PyBytes::new(py, &out))
}
