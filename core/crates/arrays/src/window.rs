//! Filters with an n-D window as lazy views (Workplan WP15, task DR7 steps 4 and 5),
//! from `dyna-zarr-filters`, with scipy.ndimage's values: the rank filter (what
//! rank_filter, median_filter and percentile_filter run on arrays of two or more
//! dimensions) and the correlation with an n-D kernel (correlate, convolve).
//!
//! A region is read as its input box: the region grown by the window's reach, cut back
//! at the array's edges, plus the positions the mode maps the reach past an edge to. The
//! kernel extends that box at the edges as scipy does, so any region gives the values of
//! the whole filtered array.

use dyna_zarr_filters as fl;
use fl::Select;

use crate::downsample::{split, split_cost, DEFAULT_MAX_READ};
use crate::dtype::DType;
use crate::filter::mapped_range;
use crate::out::Out;
use crate::pool::install;
use crate::source::{read_vec, region, Source, SourceRef, SMALL};
use crate::ReadError;

pub use fl::Footprint;

/// A box: start and stop per axis.
type Box_ = (Vec<u64>, Vec<u64>);

/// The filter a [`Window`] runs.
enum Kernel {
    Rank(fl::RankFilter),
    Correlate(fl::Correlate),
}

impl Kernel {
    fn reach(&self, axis: usize) -> (usize, usize) {
        match self {
            Self::Rank(f) => f.reach(axis),
            Self::Correlate(f) => f.reach(axis),
        }
    }

    fn bytes(&self, in_len: &[usize], out_len: &[usize], item: usize) -> (u64, u64) {
        match self {
            Self::Rank(f) => f.bytes(in_len, out_len, item),
            Self::Correlate(f) => f.bytes(out_len),
        }
    }

    fn run<T: Select>(
        &self,
        parallel: bool,
        input: &[T],
        shape: &[usize],
        spans: &[fl::Span],
        output: &mut [T],
    ) -> Result<(), fl::FilterError> {
        match self {
            Self::Rank(f) => f.run_on(parallel, input, shape, spans, output),
            Self::Correlate(f) => f.run_on(parallel, input, shape, spans, output),
        }
    }
}

/// A filter with an n-D window around each element of `src`, in its type.
pub struct Window {
    src: SourceRef,
    kernel: Kernel,
    mode: fl::Mode,
    max_read: u64,
}

impl Window {
    /// The element of rank `rank` among a footprint's positions (scipy's n-D rank
    /// filter).
    pub fn rank(
        src: SourceRef,
        footprint: Footprint,
        rank: usize,
        mode: fl::Mode,
    ) -> Result<Self, ReadError> {
        Self::check(&src, footprint.shape.len(), "a rank filter")?;
        let filter = fl::RankFilter::new(footprint, rank, mode).map_err(ReadError::filter)?;
        Ok(Self::of(src, Kernel::Rank(filter), mode))
    }

    /// scipy.ndimage.correlate with `weights` of `shape` (C order) shifted by `origin`;
    /// `convolve`: scipy.ndimage.convolve.
    pub fn correlate(
        src: SourceRef,
        shape: Vec<usize>,
        weights: &[f64],
        origin: Vec<isize>,
        mode: fl::Mode,
        convolve: bool,
    ) -> Result<Self, ReadError> {
        Self::check(&src, shape.len(), "a correlation")?;
        let c = if convolve {
            fl::Correlate::convolve(shape, weights, origin, mode)
        } else {
            fl::Correlate::new(shape, weights, origin, mode)
        }
        .map_err(ReadError::filter)?;
        Ok(Self::of(src, Kernel::Correlate(c), mode))
    }

    fn of(src: SourceRef, kernel: Kernel, mode: fl::Mode) -> Self {
        Self {
            src,
            kernel,
            mode,
            max_read: DEFAULT_MAX_READ,
        }
    }

    fn check(src: &SourceRef, ndim: usize, what: &str) -> Result<(), ReadError> {
        let n = src.shape().len();
        if ndim != n {
            return Err(ReadError::new(format!(
                "a {ndim}-d window for a {n}-d array"
            )));
        }
        match src.dtype() {
            DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64
            | DType::Float32
            | DType::Float64 => Ok(()),
            other => Err(ReadError::new(format!(
                "{what} of {} is not supported",
                other.name()
            ))),
        }
    }

    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    /// The box a region reads from the source.
    fn input_box(&self, start: &[u64], stop: &[u64]) -> Box_ {
        let shape = self.src.shape();
        let n = shape.len();
        let (mut lo, mut hi) = (vec![0u64; n], vec![0u64; n]);
        for a in 0..n {
            let (before, after) = self.kernel.reach(a);
            let lo_p = start[a] as i64 - before as i64;
            let hi_p = stop[a] as i64 + after as i64;
            let (l, h) = mapped_range(self.mode, shape[a] as i64, lo_p, hi_p);
            lo[a] = l as u64;
            hi[a] = h as u64;
        }
        (lo, hi)
    }

    fn part<T: Select>(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (bs, be) = self.input_box(start, &stop);
        let full = self.src.shape();
        let data = read_vec::<T>(self.src.as_ref(), &bs, &be, threads)?;
        let shape: Vec<usize> = (0..n).map(|a| (be[a] - bs[a]) as usize).collect();
        let spans: Vec<fl::Span> = (0..n)
            .map(|a| fl::Span {
                len: full[a] as usize,
                start: bs[a] as usize,
                out_start: start[a] as usize,
                out_len: r[a] as usize,
            })
            .collect();
        let mut res = vec![T::default(); r.iter().product::<u64>() as usize];
        let parallel = threads > 1;
        let mut run = || self.kernel.run(parallel, &data, &shape, &spans, &mut res);
        let done = if parallel {
            install(threads, &mut run)?
        } else {
            run()
        };
        done.map_err(ReadError::filter)?;
        drop(data);
        let item = std::mem::size_of::<T>();
        let mut strides = vec![item; n];
        for k in (0..n.saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * r[k + 1] as usize;
        }
        // SAFETY: plain numeric elements, viewed as their bytes
        let bytes =
            unsafe { std::slice::from_raw_parts(res.as_ptr() as *const u8, res.len() * item) };
        out.copy_from(r, bytes, &strides, item, None);
        Ok(())
    }

    /// One piece at its fullest: the read box with what reading it costs, or the read
    /// box, the kernel's buffers (its `bytes`) and the result.
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (bs, be) = self.input_box(start, &stop);
        let item = self.src.dtype().size() as u64;
        let in_len: Vec<usize> = (0..n).map(|a| (be[a] - bs[a]) as usize).collect();
        let out_len: Vec<usize> = r.iter().map(|&v| v as usize).collect();
        let raw = in_len.iter().map(|&v| v as u64).product::<u64>() * item;
        let reading = raw + self.src.cost(&bs, &be, threads)?;
        let (shared, each) = self.kernel.bytes(&in_len, &out_len, item as usize);
        let region = r.iter().product::<u64>() * item;
        let computing = raw + shared + threads.max(1) as u64 * each + region;
        Ok(reading.max(computing) + SMALL)
    }

    /// Input bytes of a piece (its read box twice), the measure for splitting a large
    /// request.
    fn box_bytes(&self, start: &[u64], r: &[u64]) -> u64 {
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (bs, be) = self.input_box(start, &stop);
        (0..n).map(|k| be[k] - bs[k]).product::<u64>() * self.src.dtype().size() as u64 * 2
    }
}

impl Source for Window {
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        true
    }
    fn shape(&self) -> &[u64] {
        self.src.shape()
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        self.src.grid()
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = region(self.src.shape(), start, stop)?;
        out.check(&r, self.dtype())?;
        let bytes = |s: &[u64], r: &[u64]| self.box_bytes(s, r);
        let run = |s: &[u64], r: &[u64], o: &Out| match self.src.dtype() {
            DType::Int8 => self.part::<i8>(s, r, o, threads),
            DType::Int16 => self.part::<i16>(s, r, o, threads),
            DType::Int32 => self.part::<i32>(s, r, o, threads),
            DType::Int64 => self.part::<i64>(s, r, o, threads),
            DType::UInt8 => self.part::<u8>(s, r, o, threads),
            DType::UInt16 => self.part::<u16>(s, r, o, threads),
            DType::UInt32 => self.part::<u32>(s, r, o, threads),
            DType::UInt64 => self.part::<u64>(s, r, o, threads),
            DType::Float32 => self.part::<f32>(s, r, o, threads),
            DType::Float64 => self.part::<f64>(s, r, o, threads),
            other => Err(ReadError::new(format!("filtering {}", other.name()))),
        };
        split(start, &r, out, self.max_read, &bytes, &run)
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(self.src.shape(), start, stop)?;
        let bytes = |s: &[u64], r: &[u64]| self.box_bytes(s, r);
        let cost = |s: &[u64], r: &[u64]| self.part_cost(s, r, threads);
        split_cost(start, &r, self.max_read, &bytes, &cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::read;
    use std::sync::Arc;

    /// An in-memory f32 source for the tests.
    struct Mem {
        shape: Vec<u64>,
        data: Vec<f32>,
    }

    impl Source for Mem {
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            DType::Float32
        }
        fn read_into(
            &self,
            start: &[u64],
            stop: &[u64],
            out: &Out,
            _threads: usize,
        ) -> Result<(), ReadError> {
            let r = region(&self.shape, start, stop)?;
            let n = r.len();
            let mut strides = vec![4usize; n];
            for k in (0..n - 1).rev() {
                strides[k] = strides[k + 1] * self.shape[k + 1] as usize;
            }
            let off: usize = (0..n).map(|k| start[k] as usize * strides[k]).sum();
            // SAFETY: f32s as bytes
            let bytes = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
            };
            out.copy_from(&r, &bytes[off..], &strides, 4, None);
            Ok(())
        }
        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(0)
        }
    }

    fn as_f32(b: &[u8]) -> &[f32] {
        // SAFETY: the reads gave f32s
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
    }

    #[test]
    fn any_region_equals_the_whole_filtered_array() {
        let shape = [7usize, 9, 11];
        let mut data: Vec<f32> = (0..693).map(|v| ((v * 7919) % 23) as f32 - 11.0).collect();
        data[100] = f32::NAN;
        data[300] = -0.0;
        let src: SourceRef = Arc::new(Mem {
            shape: shape.iter().map(|&s| s as u64).collect(),
            data: data.clone(),
        });
        let modes = [
            fl::Mode::Reflect,
            fl::Mode::Mirror,
            fl::Mode::Nearest,
            fl::Mode::Wrap,
            fl::Mode::Constant(2.5),
        ];
        let fp = Footprint {
            shape: vec![3, 4, 5],
            mask: None,
            origin: vec![0, -1, 1],
        };
        for mode in modes {
            let mut want = vec![0f32; data.len()];
            fl::rank_filter_on(false, &data, &shape, fp.clone(), 25, mode, &mut want).unwrap();
            let view = Window::rank(src.clone(), fp.clone(), 25, mode).unwrap();
            for (s, e) in [
                ([0u64, 0, 0], [7u64, 9, 11]),
                ([0, 0, 0], [1, 1, 1]),
                ([6, 8, 10], [7, 9, 11]),
                ([2, 3, 4], [5, 6, 9]),
            ] {
                for threads in [1, 3] {
                    let got = read(&view, &s, &e, threads).unwrap();
                    let got = as_f32(&got);
                    let mut q = 0;
                    for i in s[0]..e[0] {
                        for j in s[1]..e[1] {
                            for l in s[2]..e[2] {
                                let w = want[((i * 9 + j) * 11 + l) as usize];
                                assert_eq!(got[q].to_bits(), w.to_bits(), "{mode:?} {s:?}");
                                q += 1;
                            }
                        }
                    }
                }
            }
            // split reads give the same values
            let split = Window::rank(src.clone(), fp.clone(), 25, mode)
                .unwrap()
                .with_max_read(800);
            let got = read(&split, &[0, 0, 0], &[7, 9, 11], 2).unwrap();
            let same = as_f32(&got)
                .iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{mode:?} split");
        }
    }

    #[test]
    fn a_correlation_region_equals_the_whole() {
        let shape = [7usize, 9, 11];
        let data: Vec<f32> = (0..693).map(|v| ((v * 7919) % 23) as f32 - 11.5).collect();
        let src: SourceRef = Arc::new(Mem {
            shape: shape.iter().map(|&s| s as u64).collect(),
            data: data.clone(),
        });
        let w: Vec<f64> = (0..24).map(|i| ((i * 5) % 7) as f64 / 3.0 - 1.0).collect();
        for convolve in [false, true] {
            for mode in [fl::Mode::Reflect, fl::Mode::Wrap, fl::Mode::Constant(1.5)] {
                let mut want = vec![0f32; data.len()];
                fl::correlate_on(
                    false,
                    &data,
                    &shape,
                    vec![2, 3, 4],
                    &w,
                    vec![0, 1, -1],
                    mode,
                    convolve,
                    &mut want,
                )
                .unwrap();
                let view = Window::correlate(
                    src.clone(),
                    vec![2, 3, 4],
                    &w,
                    vec![0, 1, -1],
                    mode,
                    convolve,
                )
                .unwrap()
                .with_max_read(900);
                for (s, e) in [([0u64, 0, 0], [7u64, 9, 11]), ([2, 3, 4], [5, 6, 9])] {
                    let got = read(&view, &s, &e, 3).unwrap();
                    let got = as_f32(&got);
                    let mut q = 0;
                    for i in s[0]..e[0] {
                        for j in s[1]..e[1] {
                            for l in s[2]..e[2] {
                                let wv = want[((i * 9 + j) * 11 + l) as usize];
                                assert_eq!(got[q].to_bits(), wv.to_bits(), "{mode:?} {s:?}");
                                q += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}
