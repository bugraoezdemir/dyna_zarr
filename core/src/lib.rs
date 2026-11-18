//! dyna-zarr's native engine, shipped as the package `dyna-zarr-core` (Python module
//! `dyna_zarr_core`); Workplan WP15, decision DR-1 (the Rust array library is
//! dyna-zarr's engine, as micro-reader-core is micro-reader's).
//!
//! The work lives in two crates that know nothing about Python:
//! `dyna-zarr-transforms` (coordinate transformations, DR2) and `dyna-zarr-arrays`
//! (lazy arrays over zarrs, DR3-DR5). This crate only binds them. dyna keeps its
//! Python path as the reference the tests compare with.

mod transforms;

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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TRANSFORMS_VERSION", dyna_zarr_transforms::VERSION)?;
    m.add("ARRAYS_VERSION", dyna_zarr_arrays::VERSION)?;
    m.add_function(wrap_pyfunction!(array_info, m)?)?;
    // dyna_zarr_core.transforms (DR2): an attribute, so `from dyna_zarr_core import
    // transforms` works; registered in sys.modules too, for `import dyna_zarr_core.transforms`
    let sub = transforms::module(m.py())?;
    m.add_submodule(&sub)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("dyna_zarr_core.transforms", &sub)?;
    Ok(())
}
