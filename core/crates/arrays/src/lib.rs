//! Lazy arrays over zarrs for dyna-zarr's engine (Workplan WP15, task DR3).
//!
//! The arrays belong to Rust: a [`Source`] is any array the engine can read (a
//! [`ZarrArray`], or a lazy view over sources: [`Slice`], [`Transpose`], [`ExpandDims`],
//! [`Squeeze`], [`Concat`], [`Cast`]). Reading a region writes it into a destination the
//! caller owns ([`Out`]: a Rust buffer, NumPy's memory through the Python bindings, a GPU
//! staging buffer), so no layer copies what it does not have to; [`read`] gives a new
//! buffer.
//!
//! Memory and threads (Workplan decision DR-5): a read runs on one thread pool
//! ([`pool`]), its `threads` cap how many of its buffer-holding pieces are open at once,
//! and every source reports the most it holds for a region ([`Source::cost`]), which
//! [`memory`] measures against.

mod blocks;
mod cast;
mod downsample;
mod dtype;
mod filter;
pub mod memory;
mod out;
mod pointwise;
pub mod pool;
mod resample;
mod source;
mod views;
mod window;
mod zarr;

use std::path::Path;

pub use cast::converter;
pub use downsample::{stride, Boundary, Gaussian, Mean};
pub use dtype::DType;
pub use filter::{Combine, FilterLine, FilterMode, Pass, Separable};
pub use out::Out;
pub use pointwise::{apply as pointwise_apply, Op, Operand, Pointwise};
pub use resample::{EdgeMode, Resample, DEFAULT_MAX_READ};
pub use source::{read, read_cost, region, Source, SourceRef};
pub use views::{Cast, Concat, ExpandDims, Slice, Squeeze, Transpose};
pub use window::{Footprint, Window};
pub use zarr::ZarrArray;

/// This crate's version, for the bindings' build information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A read cannot be done as asked; the message says why.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadError {
    pub message: String,
    /// The engine cannot give the values dyna's Python path gives for this read (it
    /// depends on where the read starts, as scipy's sliding minimum through a NaN does):
    /// read it with the Python path.
    pub python_path: bool,
}

impl ReadError {
    pub fn new(message: String) -> Self {
        Self {
            message,
            python_path: false,
        }
    }

    /// A filter's error: [`ReadError::python_path`] when its values depend on where its
    /// line starts.
    pub fn filter(e: dyna_zarr_filters::FilterError) -> Self {
        Self {
            python_path: e.from_line_start(),
            message: e.0,
        }
    }
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ReadError {}

/// What an array is: its shape, the shape of its first chunk, its data type and its
/// Zarr format (2 or 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayInfo {
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub dtype: String,
    pub zarr_format: u8,
}

/// Open the Zarr array stored in the local folder `path` (v2 or v3) and describe it.
pub fn array_info(path: &Path) -> Result<ArrayInfo, String> {
    let a = ZarrArray::open(path).map_err(|e| e.message)?;
    Ok(ArrayInfo {
        shape: a.shape().to_vec(),
        chunks: a.chunks().to_vec(),
        dtype: a.dtype().name().to_string(),
        zarr_format: a.zarr_format(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use zarrs::array::{data_type, ArrayBuilder};
    use zarrs::filesystem::FilesystemStore;

    /// A 3-D uint16 array whose element (z, y, x) is z * 10000 + y * 100 + x.
    fn make(dir: &Path, shape: [u64; 3], chunks: [u64; 3]) -> ZarrArray {
        let store = Arc::new(FilesystemStore::new(dir).unwrap());
        let array = ArrayBuilder::new(shape.to_vec(), chunks.to_vec(), data_type::uint16(), 0u16)
            .build(store, "/")
            .unwrap();
        array.store_metadata().unwrap();
        let mut data = Vec::new();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    data.push((z * 10000 + y * 100 + x) as u16);
                }
            }
        }
        let all = zarrs::array::ArraySubset::new_with_shape(shape.to_vec());
        array.store_array_subset(&all, data).unwrap();
        ZarrArray::open(dir).unwrap()
    }

    fn value(z: u64, y: u64, x: u64) -> u16 {
        (z * 10000 + y * 100 + x) as u16
    }

    fn as_u16(b: &[u8]) -> Vec<u16> {
        b.chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn describes_a_v3_array() {
        let dir = tempfile::tempdir().unwrap();
        make(dir.path(), [3, 10, 20], [2, 4, 8]);
        let info = array_info(dir.path()).unwrap();
        assert_eq!(info.shape, vec![3, 10, 20]);
        assert_eq!(info.chunks, vec![2, 4, 8]);
        assert_eq!(info.dtype, "uint16");
        assert_eq!(info.zarr_format, 3);
    }

    #[test]
    fn a_folder_without_an_array_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(array_info(dir.path()).is_err());
    }

    #[test]
    fn a_region_across_chunks_on_any_thread_count() {
        let dir = tempfile::tempdir().unwrap();
        let a = make(dir.path(), [3, 10, 20], [2, 4, 8]);
        for threads in [1, 4] {
            let got = as_u16(&read(&a, &[1, 3, 5], &[3, 9, 17], threads).unwrap());
            let mut want = Vec::new();
            for z in 1..3 {
                for y in 3..9 {
                    for x in 5..17 {
                        want.push(value(z, y, x));
                    }
                }
            }
            assert_eq!(got, want);
        }
        assert!(read(&a, &[0, 0, 0], &[4, 1, 1], 1).is_err());
    }

    #[test]
    fn a_read_lands_at_its_corner_of_a_larger_destination() {
        let dir = tempfile::tempdir().unwrap();
        let a = make(dir.path(), [3, 10, 20], [2, 4, 8]);
        let mut buf = vec![0u8; 4 * 6 * 6 * 2];
        let out = Out::new(&mut buf, &[4, 6, 6], &[1, 2, 3], DType::UInt16).unwrap();
        a.read_into(&[0, 0, 0], &[2, 3, 3], &out, 2).unwrap();
        let got = as_u16(&buf);
        let at = |z: usize, y: usize, x: usize| got[(z * 6 + y) * 6 + x];
        assert_eq!(at(1, 2, 3), value(0, 0, 0));
        assert_eq!(at(2, 4, 5), value(1, 2, 2));
        assert_eq!(at(0, 0, 0), 0);
        assert_eq!(at(3, 5, 5), 0);
    }

    #[test]
    fn views_read_what_numpy_would() {
        let dir = tempfile::tempdir().unwrap();
        let a: SourceRef = Arc::new(make(dir.path(), [3, 10, 20], [2, 4, 8]));
        // stepped slice a[0:3:2, 1:10:3, 2:20:5]
        let s = Slice::new(a.clone(), &[0, 1, 2], &[3, 10, 20], &[2, 3, 5]).unwrap();
        assert_eq!(s.shape(), &[2, 3, 4]);
        let got = as_u16(&read(&s, &[0, 0, 0], &[2, 3, 4], 2).unwrap());
        let mut want = Vec::new();
        for z in [0u64, 2] {
            for y in [1u64, 4, 7] {
                for x in [2u64, 7, 12, 17] {
                    want.push(value(z, y, x));
                }
            }
        }
        assert_eq!(got, want);
        // transpose (x, z, y)
        let t = Transpose::new(a.clone(), &[2, 0, 1]).unwrap();
        assert_eq!(t.shape(), &[20, 3, 10]);
        let got = as_u16(&read(&t, &[4, 1, 2], &[6, 3, 4], 1).unwrap());
        let mut want = Vec::new();
        for x in 4..6 {
            for z in 1..3 {
                for y in 2..4 {
                    want.push(value(z, y, x));
                }
            }
        }
        assert_eq!(got, want);
        // squeeze after a slice of one plane, expand back, stack two planes, cast
        let plane: SourceRef =
            Arc::new(Slice::new(a.clone(), &[1, 0, 0], &[2, 10, 20], &[1, 1, 1]).unwrap());
        let sq: SourceRef = Arc::new(Squeeze::new(plane, 0).unwrap());
        assert_eq!(sq.shape(), &[10, 20]);
        let stacked = Concat::stack(vec![sq.clone(), sq.clone()], 2).unwrap();
        assert_eq!(stacked.shape(), &[10, 20, 2]);
        let got = as_u16(&read(&stacked, &[3, 4, 0], &[4, 6, 2], 4).unwrap());
        assert_eq!(
            got,
            vec![
                value(1, 3, 4),
                value(1, 3, 4),
                value(1, 3, 5),
                value(1, 3, 5)
            ]
        );
        let cast = Cast::new(sq.clone(), DType::Float32).unwrap();
        let got = read(&cast, &[2, 3], &[3, 5], 1).unwrap();
        let f: Vec<f32> = got
            .chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(f, vec![value(1, 2, 3) as f32, value(1, 2, 4) as f32]);
    }

    #[test]
    fn concatenated_parts_land_in_their_places() {
        let d1 = tempfile::tempdir().unwrap();
        let d2 = tempfile::tempdir().unwrap();
        let a: SourceRef = Arc::new(make(d1.path(), [2, 5, 6], [1, 2, 3]));
        let b: SourceRef = Arc::new(make(d2.path(), [3, 5, 6], [2, 5, 6]));
        let c = Concat::new(vec![a, b], 0).unwrap();
        assert_eq!(c.shape(), &[5, 5, 6]);
        for threads in [1, 3] {
            let got = as_u16(&read(&c, &[1, 2, 1], &[4, 4, 3], threads).unwrap());
            let want = vec![
                value(1, 2, 1),
                value(1, 2, 2),
                value(1, 3, 1),
                value(1, 3, 2),
                value(0, 2, 1),
                value(0, 2, 2),
                value(0, 3, 1),
                value(0, 3, 2),
                value(1, 2, 1),
                value(1, 2, 2),
                value(1, 3, 1),
                value(1, 3, 2),
            ];
            assert_eq!(got, want);
        }
    }
}
