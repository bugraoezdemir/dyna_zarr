//! Filters as lazy views (Workplan WP15, task DR7): chains of 1-D passes from
//! `dyna-zarr-filters` (correlations, uniform windows, sliding minima and maxima), with
//! scipy.ndimage's values; one chain, or several each applied to the source and combined
//! (scipy's generic_laplace and generic_gradient_magnitude).
//!
//! A region is read as its input box: the region grown by every pass's reach, cut back at
//! the array's edges, plus whatever positions the mode maps the reach past an edge to
//! (wrap mode reaches the far end of the axis). A chain's passes then shrink the box as
//! they go: pass `i` computes only what the later passes need. So any region gives the
//! values of the whole filtered array, as scipy computes them, axis by axis in the
//! element type.

use dyna_zarr_filters as fl;
use fl::Element;

use crate::downsample::{split, split_cost, DEFAULT_MAX_READ};
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::install;
use crate::source::{read_vec, region, Source, SourceRef, SMALL};
use crate::ReadError;

pub use fl::{Combine, Line as FilterLine, Mode as FilterMode, Pass};

/// A box: start and stop per axis.
type Box_ = (Vec<u64>, Vec<u64>);

/// `chains` of passes, each applied in order to `src`; several are combined by `combine`.
pub struct Separable {
    src: SourceRef,
    chains: Vec<Vec<Pass>>,
    combine: Option<Combine>,
    max_read: u64,
}

impl Separable {
    /// One chain of `passes`.
    pub fn new(src: SourceRef, passes: Vec<Pass>) -> Result<Self, ReadError> {
        Self::terms(src, vec![passes], None)
    }

    /// Several chains, each applied to `src`, combined by `combine` (needed for more
    /// than one chain).
    pub fn terms(
        src: SourceRef,
        chains: Vec<Vec<Pass>>,
        combine: Option<Combine>,
    ) -> Result<Self, ReadError> {
        let n = src.shape().len();
        if chains.is_empty() {
            return Err(ReadError::new(
                "a filter needs at least one chain".to_string(),
            ));
        }
        if chains.len() > 1 && combine.is_none() {
            return Err(ReadError::new(
                "several chains need a way to combine them".to_string(),
            ));
        }
        for p in chains.iter().flatten() {
            if p.axis >= n {
                return Err(ReadError::new(format!(
                    "a pass along axis {} of a {n}-d array",
                    p.axis
                )));
            }
            let taps = p.line.taps();
            let o = p.line.origin();
            if taps == 0 || o < -((taps / 2) as isize) || o > ((taps - 1) / 2) as isize {
                return Err(ReadError::new(format!(
                    "a pass of size {taps} with origin {o}"
                )));
            }
            if let FilterLine::Correlate { weights, .. } = &p.line {
                if weights.iter().any(|w| !w.is_finite()) {
                    return Err(ReadError::new("a pass needs finite weights".to_string()));
                }
            }
        }
        let float = matches!(src.dtype(), DType::Float32 | DType::Float64);
        if combine == Some(Combine::Hypot) && !float {
            return Err(ReadError::new(format!(
                "a gradient magnitude of {} is not supported (floating-point only)",
                src.dtype().name()
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
            | DType::Float64 => {}
            other => {
                return Err(ReadError::new(format!(
                    "filtering {} is not supported (scipy.ndimage does not filter it either)",
                    other.name()
                )))
            }
        }
        Ok(Self {
            src,
            chains,
            combine,
            max_read: DEFAULT_MAX_READ,
        })
    }

    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    /// What every pass of `chain` reads: `boxes[i]` is pass i's input, the last is the
    /// region.
    fn boxes(&self, chain: &[Pass], start: &[u64], stop: &[u64]) -> Vec<Box_> {
        let shape = self.src.shape();
        let mut cur: Box_ = (start.to_vec(), stop.to_vec());
        let mut out = vec![cur.clone()];
        for p in chain.iter().rev() {
            let a = p.axis;
            let n = shape[a] as i64;
            let (before, after) = p.reach();
            let lo_p = cur.0[a] as i64 - before as i64;
            let hi_p = cur.1[a] as i64 + after as i64;
            let (lo, hi) = mapped_range(p.mode, n, lo_p, hi_p);
            cur.0[a] = lo as u64;
            cur.1[a] = hi as u64;
            out.push(cur.clone());
        }
        out.reverse();
        out
    }

    /// Every chain's boxes, and the box read from the source (covering every chain's
    /// first box).
    fn plan(&self, start: &[u64], stop: &[u64]) -> (Vec<Vec<Box_>>, Box_) {
        let all: Vec<Vec<Box_>> = self
            .chains
            .iter()
            .map(|c| self.boxes(c, start, stop))
            .collect();
        let mut union = all[0][0].clone();
        for b in &all[1..] {
            for k in 0..union.0.len() {
                union.0[k] = union.0[k].min(b[0].0[k]);
                union.1[k] = union.1[k].max(b[0].1[k]);
            }
        }
        (all, union)
    }

    fn part<T: Element + Default>(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (plans, union) = self.plan(start, &stop);
        let full = self.src.shape();
        let parallel = threads > 1;
        let single = self.chains.len() == 1;
        let mut read = Some(read_vec::<T>(
            self.src.as_ref(),
            &union.0,
            &union.1,
            threads,
        )?);
        let mut acc: Option<Vec<T>> = None;
        for (chain, boxes) in self.chains.iter().zip(&plans) {
            // the chain's input: the read box itself (one chain), or the part it needs
            let mut cur: Vec<T> = if single {
                read.take().unwrap_or_default()
            } else {
                crop(read.as_deref().unwrap_or(&[]), &union, &boxes[0])
            };
            for (i, p) in chain.iter().enumerate() {
                let (b, next_box) = (&boxes[i], &boxes[i + 1]);
                let a = p.axis;
                let shape: Vec<usize> = (0..n).map(|k| (b.1[k] - b.0[k]) as usize).collect();
                let out_len = (next_box.1[a] - next_box.0[a]) as usize;
                let span = fl::Span {
                    len: full[a] as usize,
                    start: b.0[a] as usize,
                    out_start: next_box.0[a] as usize,
                    out_len,
                };
                let count = cur.len() / shape[a].max(1) * out_len;
                let mut next = vec![T::default(); count];
                let mut run = || p.run_on(parallel, &cur, &shape, span, &mut next);
                let done = if parallel {
                    install(threads, &mut run)?
                } else {
                    run()
                };
                done.map_err(ReadError::filter)?;
                cur = next;
            }
            // `cur` is this chain's region
            match (&mut acc, self.combine) {
                (None, Some(c)) => {
                    c.start(&mut cur);
                    acc = Some(cur);
                }
                (None, None) => acc = Some(cur),
                (Some(a), Some(c)) => c.add(a, &cur),
                (Some(_), None) => {
                    return Err(ReadError::new(
                        "several chains need a way to combine them".to_string(),
                    ))
                }
            }
        }
        let mut res = acc.unwrap_or_default();
        if let Some(c) = self.combine {
            c.finish(&mut res).map_err(ReadError::filter)?;
        }
        // `res` is the region, C-ordered
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

    /// One piece at its fullest: the read box with what reading it costs; then for each
    /// chain, beside what several chains keep (the read box and the result so far), its
    /// input and each pass's input and output boxes and buffers (`Pass::bytes`: a row
    /// table, and per thread a line or a block's scratch).
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (plans, union) = self.plan(start, &stop);
        let item = self.src.dtype().size() as u64;
        let size = |b: &Box_| (0..n).map(|k| b.1[k] - b.0[k]).product::<u64>() * item;
        let raw = size(&union);
        let mut most = raw + self.src.cost(&union.0, &union.1, threads)?;
        let region = r.iter().product::<u64>() * item;
        let kept = if self.chains.len() == 1 {
            0
        } else {
            raw + region
        };
        for (chain, boxes) in self.chains.iter().zip(&plans) {
            most = most.max(kept + size(&boxes[0]));
            for (i, p) in chain.iter().enumerate() {
                let a = p.axis;
                let out_len = (boxes[i + 1].1[a] - boxes[i + 1].0[a]) as usize;
                let (shared, each) = p.bytes(out_len);
                let tables = shared + threads.max(1) as u64 * each;
                most = most.max(kept + size(&boxes[i]) + size(&boxes[i + 1]) + tables);
            }
        }
        Ok(most + SMALL)
    }

    /// Input bytes of a piece (its read box twice), the measure for splitting a large
    /// request.
    fn box_bytes(&self, start: &[u64], r: &[u64]) -> u64 {
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (_, b) = self.plan(start, &stop);
        (0..n).map(|k| b.1[k] - b.0[k]).product::<u64>() * self.src.dtype().size() as u64 * 2
    }
}

/// The positions of an axis of length `n` that positions `lo_p .. hi_p` read: those
/// inside the axis, and those the mode maps the ones past an edge to (wrap reaches the
/// far end).
pub(crate) fn mapped_range(mode: FilterMode, n: i64, lo_p: i64, hi_p: i64) -> (i64, i64) {
    let (mut lo, mut hi) = (lo_p.max(0), hi_p.min(n));
    for q in (lo_p..0.min(hi_p)).chain(n.max(lo_p)..hi_p) {
        if lo == 0 && hi == n {
            break;
        }
        if let Some(g) = mode.index(q, n) {
            lo = lo.min(g);
            hi = hi.max(g + 1);
        }
    }
    (lo, hi)
}

/// The part `want` of a C-ordered block `data` that holds the box `have`.
fn crop<T: Copy>(data: &[T], have: &Box_, want: &Box_) -> Vec<T> {
    let n = have.0.len();
    let hs: Vec<usize> = (0..n).map(|k| (have.1[k] - have.0[k]) as usize).collect();
    let ws: Vec<usize> = (0..n).map(|k| (want.1[k] - want.0[k]) as usize).collect();
    let off: Vec<usize> = (0..n).map(|k| (want.0[k] - have.0[k]) as usize).collect();
    let count: usize = ws.iter().product();
    let mut out = Vec::with_capacity(count);
    if count == 0 || n == 0 {
        return out;
    }
    let inner = ws[n - 1];
    let mut idx = vec![0usize; n - 1];
    for _ in 0..count / inner {
        let mut at = 0usize;
        for k in 0..n - 1 {
            at = at * hs[k] + idx[k] + off[k];
        }
        at = at * hs[n - 1] + off[n - 1];
        out.extend_from_slice(&data[at..at + inner]);
        for k in (0..n - 1).rev() {
            idx[k] += 1;
            if idx[k] < ws[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

impl Source for Separable {
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

    /// An in-memory f64 source for the tests.
    struct Mem {
        shape: Vec<u64>,
        data: Vec<f64>,
    }

    impl Source for Mem {
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            DType::Float64
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
            let mut strides = vec![8usize; n];
            for k in (0..n - 1).rev() {
                strides[k] = strides[k + 1] * self.shape[k + 1] as usize;
            }
            let off: usize = (0..n).map(|k| start[k] as usize * strides[k]).sum();
            // SAFETY: f64s as bytes
            let bytes = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 8)
            };
            out.copy_from(&r, &bytes[off..], &strides, 8, None);
            Ok(())
        }
        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(0)
        }
    }

    const SHAPE: [usize; 3] = [7, 9, 11];

    fn source() -> (SourceRef, Vec<f64>) {
        let data: Vec<f64> = (0..693).map(|v| ((v * 7919) % 101) as f64 - 50.0).collect();
        let src: SourceRef = Arc::new(Mem {
            shape: SHAPE.iter().map(|&s| s as u64).collect(),
            data: data.clone(),
        });
        (src, data)
    }

    /// Every region read alone equals that region of `want` (the whole filtered array).
    fn regions_equal(view: &Separable, want: &[f64], what: &str) {
        for (s, e) in [
            ([0, 0, 0], [7, 9, 11]),
            ([0, 0, 0], [1, 1, 1]),
            ([6, 8, 10], [7, 9, 11]),
            ([2, 3, 4], [5, 6, 9]),
            ([0, 4, 9], [3, 9, 11]),
        ] {
            for threads in [1, 3] {
                let got = read(view, &s, &e, threads).unwrap();
                // SAFETY: the read gave f64s
                let got: &[f64] = unsafe {
                    std::slice::from_raw_parts(got.as_ptr() as *const f64, got.len() / 8)
                };
                let mut k = 0;
                for i in s[0]..e[0] {
                    for j in s[1]..e[1] {
                        for l in s[2]..e[2] {
                            let w = want[((i * 9 + j) * 11 + l) as usize];
                            assert_eq!(got[k], w, "{what} {s:?}..{e:?} at {i},{j},{l}");
                            k += 1;
                        }
                    }
                }
            }
        }
    }

    const MODES: [FilterMode; 5] = [
        FilterMode::Reflect,
        FilterMode::Mirror,
        FilterMode::Nearest,
        FilterMode::Wrap,
        FilterMode::Constant(2.5),
    ];

    #[test]
    fn any_region_equals_the_whole_filtered_array() {
        let (src, data) = source();
        for mode in MODES {
            let passes = vec![
                Pass::correlate(0, vec![0.25, 0.5, 0.25], mode),
                Pass::correlate(1, vec![0.1, 0.2, 0.3, 0.4], mode),
                Pass::correlate(2, fl::gaussian_weights(2.0, 0, 8), mode),
            ];
            let mut want = vec![0f64; data.len()];
            fl::separable(&data, &SHAPE, &passes, &mut want).unwrap();
            let view = Separable::new(src.clone(), passes).unwrap();
            regions_equal(&view, &want, &format!("{mode:?}"));
        }
    }

    #[test]
    fn running_passes_too() {
        // a uniform window, then a maximum with an origin, then a minimum
        let (src, data) = source();
        for mode in MODES {
            let passes = vec![
                Pass {
                    axis: 0,
                    line: FilterLine::Uniform { size: 3, origin: 0 },
                    mode,
                },
                Pass {
                    axis: 1,
                    line: FilterLine::MinMax {
                        size: 4,
                        origin: -1,
                        minimum: false,
                    },
                    mode,
                },
                Pass {
                    axis: 2,
                    line: FilterLine::MinMax {
                        size: 5,
                        origin: 0,
                        minimum: true,
                    },
                    mode,
                },
            ];
            let mut want = vec![0f64; data.len()];
            fl::separable(&data, &SHAPE, &passes, &mut want).unwrap();
            let view = Separable::new(src.clone(), passes).unwrap();
            regions_equal(&view, &want, &format!("{mode:?}"));
        }
    }

    #[test]
    fn combined_chains_too() {
        // a laplace (sum of one pass per axis) and a gradient magnitude (hypot of
        // Gaussian chains with one derivative each)
        let (src, data) = source();
        for mode in MODES {
            let lap: Vec<Vec<Pass>> = (0..3)
                .map(|a| vec![Pass::correlate(a, vec![1.0, -2.0, 1.0], mode)])
                .collect();
            let grad: Vec<Vec<Pass>> = (0..3)
                .map(|d| {
                    (0..3)
                        .map(|a| {
                            let order = usize::from(a == d);
                            Pass::correlate(a, fl::gaussian_weights(1.0, order, 4), mode)
                        })
                        .collect()
                })
                .collect();
            for (chains, combine) in [(lap, Combine::Sum), (grad, Combine::Hypot)] {
                let mut want = vec![0f64; data.len()];
                fl::terms(&data, &SHAPE, &chains, combine, &mut want).unwrap();
                let view = Separable::terms(src.clone(), chains, Some(combine)).unwrap();
                regions_equal(&view, &want, &format!("{combine:?} {mode:?}"));
            }
        }
    }

    #[test]
    fn a_split_read_gives_the_same_values() {
        let shape = [20usize, 30];
        let data: Vec<f64> = (0..600).map(|v| ((v * 31) % 17) as f64).collect();
        let src: SourceRef = Arc::new(Mem {
            shape: vec![20, 30],
            data: data.clone(),
        });
        let passes = vec![
            Pass::correlate(0, fl::gaussian_weights(1.5, 0, 6), FilterMode::Mirror),
            Pass::correlate(1, fl::gaussian_weights(1.5, 1, 6), FilterMode::Mirror),
        ];
        let mut want = vec![0f64; data.len()];
        fl::separable(&data, &shape, &passes, &mut want).unwrap();
        let view = Separable::new(src, passes).unwrap().with_max_read(200);
        let got = read(&view, &[0, 0], &[20, 30], 2).unwrap();
        // SAFETY: the read gave f64s
        let got: &[f64] =
            unsafe { std::slice::from_raw_parts(got.as_ptr() as *const f64, got.len() / 8) };
        assert_eq!(got, &want[..]);
    }

    #[test]
    fn wrap_reads_the_far_end() {
        let src: SourceRef = Arc::new(Mem {
            shape: vec![100],
            data: vec![0.0; 100],
        });
        let view = Separable::new(
            src,
            vec![Pass::correlate(0, vec![1.0; 5], FilterMode::Wrap)],
        )
        .unwrap();
        assert_eq!(view.plan(&[0], &[3]).1, (vec![0], vec![100]));
        assert_eq!(view.plan(&[40], &[43]).1, (vec![38], vec![45]));
    }

    #[test]
    fn crop_takes_the_inner_box() {
        let data: Vec<u32> = (0..60).collect();
        let have = (vec![10, 20, 30], vec![13, 24, 35]);
        let want = (vec![11, 21, 32], vec![13, 23, 34]);
        // (i, j, k) of the 3 x 4 x 5 block is i*20 + j*5 + k
        assert_eq!(
            crop(&data, &have, &want),
            vec![27, 28, 32, 33, 47, 48, 52, 53]
        );
    }
}
