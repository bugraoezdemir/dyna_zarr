//! dyna-zarr's native engine, shipped as the package `dyna-zarr-core` (Python module
//! `dyna_zarr_core`); Workplan WP15, decision DR-1 (the Rust array library is
//! dyna-zarr's engine, as micro-reader-core is micro-reader's).
//!
//! The work lives in two crates that know nothing about Python:
//! `dyna-zarr-transforms` (coordinate transformations, DR2) and `dyna-zarr-arrays`
//! (lazy arrays over zarrs, DR3-DR5). This crate only binds them. dyna keeps its
//! Python path as the reference the tests compare with.

mod arrays;
mod filters;
mod transforms;

/// mimalloc for everything Rust allocates in this module (Python's own memory is not
/// affected). Decoding a chunk allocates a buffer of the chunk's size; the system
/// allocators map such blocks fresh each time (page faults on every chunk), which made
/// reads 2-3.5x slower on Windows (Cargo.toml). Counted (decision DR-5), so tests and
/// writes can compare a read's real peak with its cost.
#[global_allocator]
static GLOBAL: dyna_zarr_arrays::memory::Counting<mimalloc::MiMalloc> =
    dyna_zarr_arrays::memory::Counting(mimalloc::MiMalloc);

/// mimalloc hands memory a thread freed back to the system this many milliseconds later
/// (mimalloc's own default is 10). A read thread frees a chunk's buffers every few
/// milliseconds, so with 10 ms each thread held several freed buffers at once: a float32
/// write with 4 engine threads at a 32 MB budget grew by 32-34 MB where the engine had
/// planned 26 (its own live memory 7.7 MB), and macOS CI went over the peak test's limit.
/// Measured on Windows (2026-11-20): 1 ms -> 29-30 MB, reads as fast as with 10 ms;
/// 0 -> 26 MB, but reads 25-50% slower (every chunk's buffer mapped fresh again).
/// `MIMALLOC_PURGE_DELAY` in the environment wins.
const PURGE_DELAY_MS: std::os::raw::c_long = 1;
/// `mi_option_purge_delay` in mimalloc v2's mimalloc.h (the enum's 16th member); the
/// binding names no constant for it. tests/test_engine_memory.py checks that this option
/// reads mimalloc's default of 10 when the environment sets it.
const MI_OPTION_PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;

fn set_purge_delay() {
    if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
        // SAFETY: mi_option_set only stores the value; mimalloc reads it on later purges
        unsafe { libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, PURGE_DELAY_MS) }
    }
}

/// mimalloc's purge delay in milliseconds, as the engine's allocator uses it.
#[pyfunction]
fn allocator_purge_delay_ms() -> i64 {
    // SAFETY: reads an option
    unsafe { libmimalloc_sys::mi_option_get(MI_OPTION_PURGE_DELAY) as i64 }
}

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Describe the Zarr array (v2 or v3) in the local folder `path`: a dict with
/// `shape`, `chunks` (the first chunk's shape), `dtype` (zarrs' name) and
/// `zarr_format`. The GIL is released while zarrs reads the metadata.
#[pyfunction]
fn array_info<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let info = py
        .detach(|| dyna_zarr_arrays::array_info(&path))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("shape", info.shape)?;
    out.set_item("chunks", info.chunks)?;
    out.set_item("dtype", info.dtype)?;
    out.set_item("zarr_format", info.zarr_format)?;
    Ok(out)
}

#[pymodule]
fn dyna_zarr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    set_purge_delay();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TRANSFORMS_VERSION", dyna_zarr_transforms::VERSION)?;
    m.add("ARRAYS_VERSION", dyna_zarr_arrays::VERSION)?;
    m.add_function(wrap_pyfunction!(array_info, m)?)?;
    // lazy arrays (DR3)
    m.add_class::<arrays::Array>()?;
    m.add_class::<arrays::Pool>()?;
    m.add_class::<arrays::Writer>()?;
    m.add_function(wrap_pyfunction!(arrays::memory_in_use, m)?)?;
    m.add_function(wrap_pyfunction!(arrays::memory_peak, m)?)?;
    m.add_function(wrap_pyfunction!(arrays::reset_memory_peak, m)?)?;
    m.add_function(wrap_pyfunction!(allocator_purge_delay_ms, m)?)?;
    m.add("ReadError", m.py().get_type::<arrays::ReadError>())?;
    m.add("UsePythonPath", m.py().get_type::<arrays::UsePythonPath>())?;
    m.add_function(wrap_pyfunction!(arrays::concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(arrays::stack, m)?)?;
    // pointwise operations (DR7 step 7)
    m.add_function(wrap_pyfunction!(arrays::pointwise, m)?)?;
    m.add_function(wrap_pyfunction!(arrays::pointwise_apply, m)?)?;
    // dyna_zarr_core.transforms (DR2): an attribute, so `from dyna_zarr_core import
    // transforms` works; registered in sys.modules too, for `import dyna_zarr_core.transforms`
    let sub = transforms::module(m.py())?;
    m.add_submodule(&sub)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("dyna_zarr_core.transforms", &sub)?;
    // dyna_zarr_core.filters (DR7), registered the same way
    let sub = filters::module(m.py())?;
    m.add_submodule(&sub)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("dyna_zarr_core.filters", &sub)?;
    m.add("FILTERS_VERSION", dyna_zarr_filters::VERSION)?;
    Ok(())
}
