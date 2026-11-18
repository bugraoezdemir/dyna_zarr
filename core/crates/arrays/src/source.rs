//! What every array of the engine answers: its shape and element type, and reading a
//! region into a destination ([`Out`]). Zarr arrays and lazy views over them are
//! sources; a view reads from the sources below it only when asked.

use std::sync::Arc;

use crate::dtype::DType;
use crate::out::Out;
use crate::ReadError;

/// An array the engine can read.
pub trait Source: Send + Sync {
    /// The shape.
    fn shape(&self) -> &[u64];

    /// The element type.
    fn dtype(&self) -> DType;

    /// The storage grid (chunk shape) of the data underneath, if any: views use it to
    /// read in blocks that match the storage.
    fn grid(&self) -> Option<Vec<u64>> {
        None
    }

    /// Read the region `start .. stop` into `out` (its region shape is `stop - start`),
    /// using up to `threads` threads.
    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError>;
}

/// A shared source: views hold their sources this way.
pub type SourceRef = Arc<dyn Source>;

/// The region's extent per axis, after checking it lies inside `shape`.
pub fn region(shape: &[u64], start: &[u64], stop: &[u64]) -> Result<Vec<u64>, ReadError> {
    if start.len() != shape.len() || stop.len() != shape.len() {
        return Err(ReadError::new(format!(
            "a region of {} / {} values for an array of {} axes",
            start.len(),
            stop.len(),
            shape.len()
        )));
    }
    let mut out = Vec::with_capacity(shape.len());
    for i in 0..shape.len() {
        if start[i] > stop[i] || stop[i] > shape[i] {
            return Err(ReadError::new(format!(
                "region {start:?} .. {stop:?} is not inside the array's shape {shape:?}"
            )));
        }
        out.push(stop[i] - start[i]);
    }
    Ok(out)
}

/// Read the region `start .. stop` of `src` into a new C-ordered buffer.
pub fn read(
    src: &dyn Source,
    start: &[u64],
    stop: &[u64],
    threads: usize,
) -> Result<Vec<u8>, ReadError> {
    let r = region(src.shape(), start, stop)?;
    let mut buf = vec![0u8; r.iter().product::<u64>() as usize * src.dtype().size()];
    {
        let out = Out::new(&mut buf, &r, &vec![0; r.len()], src.dtype())?;
        src.read_into(start, stop, &out, threads)?;
    }
    Ok(buf)
}

/// C strides of `shape` in bytes for items of `item` bytes.
pub(crate) fn c_strides(shape: &[u64], item: usize) -> Vec<usize> {
    let mut s = vec![item; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1] as usize;
    }
    s
}
