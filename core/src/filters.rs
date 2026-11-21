//! Python bindings of `dyna-zarr-filters` (DR7), the submodule `dyna_zarr_core.filters`:
//! scipy.ndimage's filters on whole NumPy arrays, with scipy's results. The lazy filter
//! views over Zarr arrays come later (DR7 step 2); these functions are what the tests
//! compare with scipy.
//!
//! Input and output are C-contiguous NumPy arrays of the same shape that do not share
//! memory. The output's type is the input's, float32 or float64 (as scipy's `output=`).
//! Errors raise `dyna_zarr_core.filters.FilterError`, a ValueError.

use dyna_zarr_arrays as ar;
use dyna_zarr_filters as fl;
use fl::Element;
use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(filters, FilterError, PyValueError);

fn err(e: fl::FilterError) -> PyErr {
    FilterError::new_err(e.0)
}

/// A NumPy array's memory, kept exported (alive, in place, not resizable) while held.
struct Mem {
    buf: PyBuffer<u8>,
    dtype: String,
    shape: Vec<usize>,
}

impl Mem {
    fn of(a: &Bound<'_, PyAny>, writable: bool) -> PyResult<Self> {
        let flags = a.getattr("flags")?;
        if !flags.getattr("c_contiguous")?.extract::<bool>()? {
            return Err(FilterError::new_err("arrays must be C-contiguous"));
        }
        if writable && !flags.getattr("writeable")?.extract::<bool>()? {
            return Err(FilterError::new_err("output is read-only"));
        }
        let dtype: String = a.getattr("dtype")?.getattr("name")?.extract()?;
        let shape: Vec<usize> = a.getattr("shape")?.extract()?;
        // flat bytes: NumPy cannot view a 0-d array as bytes; C-contiguous, so a view
        let bytes = a
            .call_method1("reshape", (-1,))?
            .call_method1("view", ("u1",))?;
        Ok(Self {
            buf: PyBuffer::get(&bytes)?,
            dtype,
            shape,
        })
    }

    fn range(&self) -> (usize, usize) {
        let p = self.buf.buf_ptr() as usize;
        (p, p + self.buf.len_bytes())
    }
}

/// The memory at `addr` as `n` elements of `T`, after checking its alignment.
///
/// # Safety
/// `addr .. addr + n * size_of::<T>()` must be live memory of `T`s for the lifetime used.
unsafe fn elements<'a, T>(addr: usize, bytes: usize) -> Result<&'a mut [T], fl::FilterError> {
    if !addr.is_multiple_of(std::mem::align_of::<T>()) {
        return Err(fl::FilterError::new("arrays must be aligned"));
    }
    let n = bytes / std::mem::size_of::<T>();
    Ok(std::slice::from_raw_parts_mut(addr as *mut T, n))
}

/// Run `$body` with `$x: &[T]` and `$y: &mut [U]` for the arrays' types: U is T,
/// float32 or float64.
macro_rules! typed {
    ($input:expr, $output:expr, |$x:ident, $y:ident| $body:expr) => {{
        let (ia, ib) = $input.range();
        let (oa, ob) = $output.range();
        macro_rules! out_as {
            ($t:ty) => {
                match $output.dtype.as_str() {
                    "float32" => run::<$t, f32>(ia, ib - ia, oa, ob - oa, |$x, $y| $body),
                    "float64" => run::<$t, f64>(ia, ib - ia, oa, ob - oa, |$x, $y| $body),
                    o if o == <$t as Element>::NAME => {
                        run::<$t, $t>(ia, ib - ia, oa, ob - oa, |$x, $y| $body)
                    }
                    o => Err(fl::FilterError::new(format!(
                        "output {o} for input {}: the input's type, float32 or float64",
                        <$t as Element>::NAME
                    ))),
                }
            };
        }
        match $input.dtype.as_str() {
            "uint8" => out_as!(u8),
            "int8" => out_as!(i8),
            "uint16" => out_as!(u16),
            "int16" => out_as!(i16),
            "uint32" => out_as!(u32),
            "int32" => out_as!(i32),
            "uint64" => out_as!(u64),
            "int64" => out_as!(i64),
            "float32" => out_as!(f32),
            "float64" => out_as!(f64),
            t => Err(fl::FilterError::new(format!("{t} is not supported"))),
        }
    }};
}

fn run<T: Element, U: Element + Default>(
    ia: usize,
    ilen: usize,
    oa: usize,
    olen: usize,
    f: impl FnOnce(&[T], &mut [U]) -> Result<(), fl::FilterError>,
) -> Result<(), fl::FilterError> {
    // SAFETY: both ranges are exported NumPy buffers held by the caller for the call;
    // they do not overlap (checked by `checked`)
    let (x, y) = unsafe { (elements::<T>(ia, ilen)?, elements::<U>(oa, olen)?) };
    f(x, y)
}

fn checked(input: &Mem, output: &Mem) -> PyResult<()> {
    if input.shape != output.shape {
        return Err(FilterError::new_err(format!(
            "output shape {:?} for input shape {:?}",
            output.shape, input.shape
        )));
    }
    let ((a0, a1), (b0, b1)) = (input.range(), output.range());
    if a0 < b1 && b0 < a1 {
        return Err(FilterError::new_err(
            "output must not share memory with input",
        ));
    }
    Ok(())
}

fn axis_of(axis: isize, ndim: usize) -> PyResult<usize> {
    let a = if axis < 0 { axis + ndim as isize } else { axis };
    if (0..ndim as isize).contains(&a) {
        Ok(a as usize)
    } else {
        Err(FilterError::new_err(format!(
            "axis {axis} for {ndim} dimensions"
        )))
    }
}

fn mode_of(name: &str, cval: f64) -> PyResult<fl::Mode> {
    fl::Mode::parse(name, cval).ok_or_else(|| {
        FilterError::new_err(format!(
            "mode {name:?}: reflect, mirror, nearest, wrap or constant"
        ))
    })
}

/// Run `f` on `threads` threads (one: right here), the GIL released.
fn threaded(
    py: Python<'_>,
    threads: usize,
    f: impl FnOnce() -> Result<(), fl::FilterError> + Send,
) -> PyResult<()> {
    py.detach(|| ar::pool::enter(threads.max(1), None, f))
        .map_err(|e| FilterError::new_err(e.message))?
        .map_err(err)
}

/// scipy.ndimage.correlate1d(input, weights, axis, output, mode, cval, origin) into
/// `output`.
#[pyfunction]
#[pyo3(signature = (input, weights, axis, output, mode = "reflect", cval = 0.0, origin = 0, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn correlate1d(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    weights: Vec<f64>,
    axis: isize,
    output: &Bound<'_, PyAny>,
    mode: &str,
    cval: f64,
    origin: isize,
    threads: usize,
) -> PyResult<()> {
    let (i, o) = (Mem::of(input, false)?, Mem::of(output, true)?);
    checked(&i, &o)?;
    if i.shape.is_empty() {
        return Err(FilterError::new_err("a 0-d array has no axis"));
    }
    let axis = axis_of(axis, i.shape.len())?;
    let mode = mode_of(mode, cval)?;
    let span = fl::Span::whole(i.shape[axis]);
    let shape = i.shape.clone();
    threaded(py, threads, || {
        typed!(i, o, |x, y| fl::correlate1d(
            x, &shape, axis, &weights, origin, mode, span, y
        ))
    })
}

/// scipy.ndimage.gaussian_filter(input, sigma, order, output, mode, cval, truncate)
/// into `output`, with the crate's own kernels; `sigma`, `order` and `mode` one per
/// axis.
#[pyfunction]
#[pyo3(signature = (input, sigma, output, order, mode, cval = 0.0, truncate = 4.0, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn gaussian_filter(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    sigma: Vec<f64>,
    output: &Bound<'_, PyAny>,
    order: Vec<usize>,
    mode: Vec<String>,
    cval: f64,
    truncate: f64,
    threads: usize,
) -> PyResult<()> {
    let (i, o) = (Mem::of(input, false)?, Mem::of(output, true)?);
    checked(&i, &o)?;
    let modes = mode
        .iter()
        .map(|m| mode_of(m, cval))
        .collect::<PyResult<Vec<_>>>()?;
    let shape = i.shape.clone();
    threaded(py, threads, || {
        typed!(i, o, |x, y| fl::gaussian_filter(
            x, &shape, &sigma, &order, &modes, truncate, y
        ))
    })
}

/// One pass as Python gives it: (axis, kind, values, origin, mode).
pub type PassSpec = (usize, String, Vec<f64>, isize, String);

/// The passes of `chains` (see `Array.filter`), with `cval` for constant mode.
pub fn chains_of(chains: Vec<Vec<PassSpec>>, cval: f64) -> Result<Vec<Vec<fl::Pass>>, String> {
    chains
        .into_iter()
        .map(|c| c.into_iter().map(|p| pass_of(p, cval)).collect())
        .collect()
}

fn pass_of((axis, kind, values, origin, mode): PassSpec, cval: f64) -> Result<fl::Pass, String> {
    let mode = fl::Mode::parse(&mode, cval)
        .ok_or_else(|| format!("mode {mode:?}: reflect, mirror, nearest, wrap or constant"))?;
    let size = || match values.as_slice() {
        [s] if s.fract() == 0.0 && *s >= 1.0 => Ok(*s as usize),
        _ => Err(format!("a {kind} pass takes [size], not {values:?}")),
    };
    let line = match kind.as_str() {
        "correlate" => fl::Line::Correlate {
            weights: values.clone(),
            origin,
        },
        "uniform" => fl::Line::Uniform {
            size: size()?,
            origin,
        },
        "minimum" | "maximum" => fl::Line::MinMax {
            size: size()?,
            origin,
            minimum: kind == "minimum",
        },
        other => {
            return Err(format!(
                "pass kind {other:?}: correlate, uniform, minimum or maximum"
            ))
        }
    };
    Ok(fl::Pass { axis, line, mode })
}

/// `combine` by name: None, "sum" or "hypot".
pub fn combine_of(combine: Option<&str>) -> Result<Option<fl::Combine>, String> {
    combine
        .map(|c| fl::Combine::parse(c).ok_or_else(|| format!("combine {c:?}: sum or hypot")))
        .transpose()
}

/// A filter of chains of passes (see `Array.filter`) on a whole NumPy array: scipy's
/// separable filters (one chain) and generic_laplace / generic_gradient_magnitude
/// (several, combined by "sum" / "hypot").
#[pyfunction]
#[pyo3(signature = (input, chains, output, combine = None, cval = 0.0, threads = 1))]
fn chains(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    chains: Vec<Vec<PassSpec>>,
    output: &Bound<'_, PyAny>,
    combine: Option<&str>,
    cval: f64,
    threads: usize,
) -> PyResult<()> {
    let (i, o) = (Mem::of(input, false)?, Mem::of(output, true)?);
    checked(&i, &o)?;
    let chains = chains_of(chains, cval).map_err(FilterError::new_err)?;
    let combine = combine_of(combine).map_err(FilterError::new_err)?;
    let n = i.shape.len();
    if let Some(p) = chains.iter().flatten().find(|p| p.axis >= n) {
        return Err(FilterError::new_err(format!(
            "a pass along axis {} of a {n}-d array",
            p.axis
        )));
    }
    let shape = i.shape.clone();
    threaded(py, threads, || {
        typed!(i, o, |x, y| match (chains.len(), combine) {
            (1, None) => fl::separable(x, &shape, &chains[0], y),
            (_, Some(c)) => fl::terms(x, &shape, &chains, c, y),
            _ => Err(fl::FilterError::new(
                "several chains need combine (sum or hypot)",
            )),
        })
    })
}

/// scipy.ndimage's n-D rank filter (`NI_RankFilter`) into `output` (the input's type):
/// the element of rank `rank` among the positions of a footprint of `shape` (all of it,
/// or those `mask` marks, C order) shifted by `origin`. scipy's own dispatch (rank 0 and
/// the last rank to minimum_filter / maximum_filter, 1-D arrays to another routine) is
/// the caller's.
#[pyfunction]
#[pyo3(signature = (input, shape, rank, output, mask = None, origin = None, mode = "reflect", cval = 0.0, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn rank_filter(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    rank: usize,
    output: &Bound<'_, PyAny>,
    mask: Option<Vec<bool>>,
    origin: Option<Vec<isize>>,
    mode: &str,
    cval: f64,
    threads: usize,
) -> PyResult<()> {
    let (i, o) = (Mem::of(input, false)?, Mem::of(output, true)?);
    checked(&i, &o)?;
    if i.dtype != o.dtype {
        return Err(FilterError::new_err(format!(
            "output {} for input {}: the input's type",
            o.dtype, i.dtype
        )));
    }
    let origin = origin.unwrap_or_else(|| vec![0; shape.len()]);
    let fp = fl::Footprint {
        shape,
        mask,
        origin,
    };
    let mode = mode_of(mode, cval)?;
    let dims = i.shape.clone();
    let (ia, ib) = i.range();
    let (oa, ob) = o.range();
    threaded(py, threads, || {
        macro_rules! go {
            ($t:ty) => {
                run::<$t, $t>(ia, ib - ia, oa, ob - oa, |x, y| {
                    fl::rank_filter_on(threads > 1, x, &dims, fp, rank, mode, y)
                })
            };
        }
        match i.dtype.as_str() {
            "uint8" => go!(u8),
            "int8" => go!(i8),
            "uint16" => go!(u16),
            "int16" => go!(i16),
            "uint32" => go!(u32),
            "int32" => go!(i32),
            "uint64" => go!(u64),
            "int64" => go!(i64),
            "float32" => go!(f32),
            "float64" => go!(f64),
            t => Err(fl::FilterError::new(format!("{t} is not supported"))),
        }
    })
}

/// scipy.ndimage.correlate (`convolve=True`: convolve) with an n-D kernel `weights` of
/// `shape` (C order) into `output` (the input's type, float32 or float64).
#[pyfunction]
#[pyo3(signature = (input, shape, weights, output, origin = None, mode = "reflect", cval = 0.0, convolve = false, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn correlate(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    weights: Vec<f64>,
    output: &Bound<'_, PyAny>,
    origin: Option<Vec<isize>>,
    mode: &str,
    cval: f64,
    convolve: bool,
    threads: usize,
) -> PyResult<()> {
    let (i, o) = (Mem::of(input, false)?, Mem::of(output, true)?);
    checked(&i, &o)?;
    let origin = origin.unwrap_or_else(|| vec![0; shape.len()]);
    let mode = mode_of(mode, cval)?;
    let dims = i.shape.clone();
    threaded(py, threads, || {
        typed!(i, o, |x, y| fl::correlate_on(
            threads > 1,
            x,
            &dims,
            shape.clone(),
            &weights,
            origin.clone(),
            mode,
            convolve,
            y
        ))
    })
}

/// The correlation weights of scipy.ndimage.gaussian_filter1d (its kernel reversed).
#[pyfunction]
fn gaussian_weights(sigma: f64, order: usize, radius: usize) -> Vec<f64> {
    fl::gaussian_weights(sigma, order, radius)
}

/// numpy's sum of a float64 sequence, in numpy's order (for the tests).
#[pyfunction]
fn numpy_sum(values: Vec<f64>) -> f64 {
    fl::numpy_sum(&values)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "filters")?;
    m.add("FilterError", py.get_type::<FilterError>())?;
    m.add("VERSION", fl::VERSION)?;
    m.add_function(wrap_pyfunction!(correlate1d, &m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(chains, &m)?)?;
    m.add_function(wrap_pyfunction!(rank_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(correlate, &m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_weights, &m)?)?;
    m.add_function(wrap_pyfunction!(numpy_sum, &m)?)?;
    Ok(m)
}
