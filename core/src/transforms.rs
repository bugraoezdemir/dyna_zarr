//! Python bindings of `dyna-zarr-transforms` (DR2), the submodule
//! `dyna_zarr_core.transforms`. Transformations go in and out as NGFF 0.6 JSON text;
//! matrices as lists of rows (n rows of n + 1, the homogeneous last row implied, or the
//! full (n + 1) x (n + 1)). Errors raise `dyna_zarr_core.transforms.TransformError`, a
//! ValueError.

use dyna_zarr_transforms as tf;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(transforms, TransformError, PyValueError);

fn err(e: tf::TransformError) -> PyErr {
    TransformError::new_err(e.message)
}

fn parse(text: &str) -> PyResult<tf::Transformation> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| TransformError::new_err(format!("not JSON: {e}")))?;
    tf::Transformation::from_json(&value).map_err(err)
}

fn dump(t: &tf::Transformation) -> String {
    t.to_json().to_string()
}

fn to_matrix(rows: Vec<Vec<f64>>) -> PyResult<tf::Matrix> {
    let n = rows.len();
    let rows = if n > 0 && rows.iter().all(|r| r.len() == n) {
        rows[..n - 1].to_vec() // full (n + 1) x (n + 1): its last row is implied
    } else {
        rows
    };
    tf::Matrix::from_rows(&rows).map_err(err)
}

fn full_rows(m: &tf::Matrix) -> Vec<Vec<f64>> {
    let n = m.ndim();
    (0..=n)
        .map(|i| (0..=n).map(|j| m.get(i, j)).collect())
        .collect()
}

/// The homogeneous (ndim + 1) x (ndim + 1) matrix of a transformation (JSON text).
#[pyfunction]
fn matrix(transformation: &str, ndim: usize) -> PyResult<Vec<Vec<f64>>> {
    Ok(full_rows(
        &parse(transformation)?.matrix(ndim).map_err(err)?,
    ))
}

/// Whether the transformation is modelled (a sequence: all its steps).
#[pyfunction]
fn modelled(transformation: &str) -> PyResult<bool> {
    Ok(parse(transformation)?.modelled())
}

/// The transformation read and written again (JSON text): what is written back.
#[pyfunction]
fn round_trip(transformation: &str) -> PyResult<String> {
    Ok(dump(&parse(transformation)?))
}

/// The transformations applied one after another (the first first), simplified.
#[pyfunction]
fn compose(transformations: Vec<String>, ndim: usize) -> PyResult<String> {
    let ts = transformations
        .iter()
        .map(|t| parse(t))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(dump(&tf::compose(&ts, ndim).map_err(err)?))
}

/// The inverse, from the output to the input; singular raises TransformError.
#[pyfunction]
fn invert(transformation: &str, ndim: usize) -> PyResult<String> {
    Ok(dump(&parse(transformation)?.invert(ndim).map_err(err)?))
}

/// The simplest transformation with this matrix (identity, translation, scale,
/// sequence[scale, translation] or affine).
#[pyfunction]
fn from_matrix(rows: Vec<Vec<f64>>) -> PyResult<String> {
    Ok(dump(&tf::Transformation::from_matrix(&to_matrix(rows)?)))
}

/// (scale, translation) when the transformation is one, else None.
#[pyfunction]
fn scale_translation(transformation: &str, ndim: usize) -> PyResult<Option<tf::ScaleTranslation>> {
    parse(transformation)?.scale_translation(ndim).map_err(err)
}

/// A matrix over the `sub` axes acting on the `full` axes (identity on the others).
#[pyfunction]
fn embed(rows: Vec<Vec<f64>>, sub: Vec<String>, full: Vec<String>) -> PyResult<Vec<Vec<f64>>> {
    let sub: Vec<&str> = sub.iter().map(String::as_str).collect();
    let full: Vec<&str> = full.iter().map(String::as_str).collect();
    Ok(full_rows(
        &to_matrix(rows)?.embed(&sub, &full).map_err(err)?,
    ))
}

/// The box [lo, hi] of the image of the box [lo, hi] under the matrix.
#[pyfunction]
fn bounding_box(rows: Vec<Vec<f64>>, lo: Vec<f64>, hi: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    tf::bounding_box(&to_matrix(rows)?, &lo, &hi).map_err(err)
}

/// The input index box (start, stop) an output region needs, or None when it needs
/// nothing; `rows` maps output indices to input indices; interpolation "nearest" or
/// "linear".
#[pyfunction]
fn input_region(
    rows: Vec<Vec<f64>>,
    start: Vec<u64>,
    stop: Vec<u64>,
    interpolation: &str,
    input_shape: Vec<u64>,
) -> PyResult<Option<tf::IndexBox>> {
    let interp = tf::Interpolation::parse(interpolation).ok_or_else(|| {
        TransformError::new_err(format!(
            "interpolation {interpolation:?}: use \"nearest\" or \"linear\""
        ))
    })?;
    tf::input_region(&to_matrix(rows)?, &start, &stop, interp, &input_shape).map_err(err)
}

/// How many `into` one `unit` is, or None when they cannot be converted.
#[pyfunction]
#[pyo3(signature = (unit, into))]
fn unit_factor(unit: Option<&str>, into: Option<&str>) -> Option<f64> {
    tf::unit_factor(unit, into)
}

/// The submodule `dyna_zarr_core.transforms`.
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "transforms")?;
    m.add("TransformError", py.get_type::<TransformError>())?;
    m.add("MODELLED_TYPES", tf::MODELLED_TYPES.to_vec())?;
    m.add_function(wrap_pyfunction!(matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(modelled, &m)?)?;
    m.add_function(wrap_pyfunction!(round_trip, &m)?)?;
    m.add_function(wrap_pyfunction!(compose, &m)?)?;
    m.add_function(wrap_pyfunction!(invert, &m)?)?;
    m.add_function(wrap_pyfunction!(from_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(scale_translation, &m)?)?;
    m.add_function(wrap_pyfunction!(embed, &m)?)?;
    m.add_function(wrap_pyfunction!(bounding_box, &m)?)?;
    m.add_function(wrap_pyfunction!(input_region, &m)?)?;
    m.add_function(wrap_pyfunction!(unit_factor, &m)?)?;
    Ok(m)
}
