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
use pyo3::types::{PyEllipsis, PySlice, PyTuple};

create_exception!(dyna_zarr_core, ReadError, PyValueError);

fn err(e: ar::ReadError) -> PyErr {
    ReadError::new_err(e.message)
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
    fn read_region(
        &self,
        py: Python<'_>,
        start: Vec<u64>,
        stop: Vec<u64>,
        out: Option<Bound<'_, PyAny>>,
        at: Option<Vec<u64>>,
        threads: usize,
    ) -> PyResult<Py<PyAny>> {
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
            // SAFETY: the buffer export above keeps the memory alive and in place for the
            // call; the region written lies inside it (Out::new and read_into check).
            let bytes = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
            let dst = ar::Out::new(bytes, &shape, &at, dtype)?;
            src.read_into(&start, &stop, &dst, threads)
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

    #[getter]
    fn ndim(&self) -> usize {
        self.src.shape().len()
    }

    /// The NumPy name of the element type.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.src.dtype().name()
    }

    /// The storage grid underneath (the chunk or shard shape), or None.
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
    /// array of this dtype; a new one of the region's shape when None), on `threads`
    /// threads, without the GIL. Returns `out`.
    #[pyo3(signature = (start, stop, out = None, at = None, threads = 1))]
    fn read(
        &self,
        py: Python<'_>,
        start: Vec<u64>,
        stop: Vec<u64>,
        out: Option<Bound<'_, PyAny>>,
        at: Option<Vec<u64>>,
        threads: usize,
    ) -> PyResult<Py<PyAny>> {
        self.read_region(py, start, stop, out, at, threads)
    }

    /// NumPy-style basic indexing (ints, slices with a positive step, Ellipsis, None),
    /// read at once on one thread.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let view = self.index(key)?;
        let shape = view.src.shape().to_vec();
        view.read_region(py, vec![0; shape.len()], shape, None, None, 1)
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
        let mut src: ar::SourceRef =
            Arc::new(ar::Slice::new(self.src.clone(), &start, &stop, &step).map_err(err)?);
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
