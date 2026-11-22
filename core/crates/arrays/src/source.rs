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

    /// This array as a pointwise view, if it is one: a pointwise view over it merges it
    /// into its own expression.
    fn as_pointwise(&self) -> Option<&crate::pointwise::Pointwise> {
        None
    }

    /// Read the region `start .. stop` into `out` (its region shape is `stop - start`),
    /// with `threads` (see [`crate::pool`]: at most `threads` pieces holding memory at
    /// once; 1 runs on the calling thread).
    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError>;

    /// The most heap bytes reading `start .. stop` with `threads` holds at once, beyond
    /// the destination (Workplan decision DR-5): this view's buffers plus the cost of
    /// what it reads from its sources, following the same decisions `read_into` makes.
    /// An upper bound, checked against measured peaks in the tests.
    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError>;
}

/// Small allocations of one piece of a read (index vectors, kernel tables, a region's
/// bookkeeping), counted in every cost on top of its buffers.
pub(crate) const SMALL: u64 = 16 << 10;

/// The cost of a whole read of `start .. stop` with `threads`, as a caller asks for it:
/// the source's cost and the call's own bookkeeping.
pub fn read_cost(
    src: &dyn Source,
    start: &[u64],
    stop: &[u64],
    threads: usize,
) -> Result<u64, ReadError> {
    Ok(src.cost(start, stop, threads)? + SMALL)
}

/// Bytes of a region of `r` elements of `item` bytes.
pub(crate) fn bytes_of(r: &[u64], item: usize) -> u64 {
    r.iter().product::<u64>() * item as u64
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

/// The region `start .. stop` of `src` read straight into a vector of `T`: the element
/// type, or any type of the element's size (an unsigned integer to move bits).
pub(crate) fn read_vec<T: Copy + Default>(
    src: &dyn Source,
    start: &[u64],
    stop: &[u64],
    threads: usize,
) -> Result<Vec<T>, ReadError> {
    let r = region(src.shape(), start, stop)?;
    if std::mem::size_of::<T>() != src.dtype().size() {
        return Err(ReadError::new(format!(
            "reading {} into elements of {} bytes",
            src.dtype().name(),
            std::mem::size_of::<T>()
        )));
    }
    let count = r.iter().product::<u64>() as usize;
    let mut v = vec![T::default(); count];
    {
        // SAFETY: the vector's memory as bytes, of the element's size, written by the
        // read and not used otherwise until it returns
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                v.as_mut_ptr() as *mut u8,
                count * std::mem::size_of::<T>(),
            )
        };
        let out = Out::new(bytes, &r, &vec![0; r.len()], src.dtype())?;
        src.read_into(start, stop, &out, threads)?;
    }
    Ok(v)
}

/// C strides of `shape` in bytes for items of `item` bytes.
pub(crate) fn c_strides(shape: &[u64], item: usize) -> Vec<usize> {
    let mut s = vec![item; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1] as usize;
    }
    s
}
