//! scipy.ndimage's correlate and convolve with an n-D kernel (`NI_Correlate`), with
//! scipy's values.
//!
//! scipy drops the weights whose magnitude is at most DBL_EPSILON (NaN weights too),
//! then sums, per output, `0.0 + w0 * v0 + w1 * v1 + ...` in the kernel's C order, in
//! double, with `cval` itself (not cast to the type) where a position is outside the
//! array in constant mode, and casts the sum to the output type (unsigned types through
//! `v > -1 ? (type)v : -(type)(-v)`). Here every output keeps that order of additions;
//! the parallelism is across outputs: a tile of neighbouring outputs along the last axis
//! adds one weight's term at a time, which the compiler vectorises.
//!
//! convolve is a correlation with the kernel reversed and the origins negated (one lower
//! for an even size), as scipy builds it ([`Correlate::convolve`]).

use crate::madd;
use crate::window::{Footprint, Window};
use crate::{Element, FilterError, Mode, Span};

/// Outputs a tile sums together.
const LANES: usize = 256;

/// A correlation with an n-D kernel: its non-zero weights in C order and where they are.
#[derive(Debug, Clone)]
pub struct Correlate {
    footprint: Footprint,
    weights: Vec<f64>,
    positions: Vec<Vec<usize>>,
    mode: Mode,
}

impl Correlate {
    /// scipy.ndimage.correlate's kernel: `weights` of `shape` (C order), shifted by
    /// `origin` per axis.
    pub fn new(
        shape: Vec<usize>,
        weights: &[f64],
        origin: Vec<isize>,
        mode: Mode,
    ) -> Result<Self, FilterError> {
        if weights.len() != shape.iter().product::<usize>() {
            return Err(FilterError::new(format!(
                "{} weights for a kernel of shape {shape:?}",
                weights.len()
            )));
        }
        // scipy: fabs(w) > DBL_EPSILON (false for NaN)
        let mask: Vec<bool> = weights.iter().map(|w| w.abs() > f64::EPSILON).collect();
        let kept: Vec<f64> = weights
            .iter()
            .zip(&mask)
            .filter(|(_, &k)| k)
            .map(|(w, _)| *w)
            .collect();
        let footprint = Footprint {
            shape,
            mask: Some(mask),
            origin,
        };
        footprint.check_shape()?;
        let positions = footprint.positions();
        Ok(Self {
            footprint,
            weights: kept,
            positions,
            mode,
        })
    }

    /// scipy.ndimage.convolve's kernel: the weights reversed along every axis and the
    /// origins negated, one lower along an axis of even size.
    pub fn convolve(
        shape: Vec<usize>,
        weights: &[f64],
        origin: Vec<isize>,
        mode: Mode,
    ) -> Result<Self, FilterError> {
        let flipped: Vec<f64> = weights.iter().rev().copied().collect();
        let origin = origin
            .iter()
            .zip(&shape)
            .map(|(&o, &s)| if s % 2 == 0 { -o - 1 } else { -o })
            .collect();
        Self::new(shape, &flipped, origin, mode)
    }

    pub fn footprint(&self) -> &Footprint {
        &self.footprint
    }

    /// How far the kernel reaches before and after an output along `axis`.
    pub fn reach(&self, axis: usize) -> (usize, usize) {
        self.footprint.reach(axis)
    }

    /// Heap bytes of a run computing `out_len` outputs per axis: (shared, per thread).
    /// Shared: the box the outputs read, as float64, and the axes' maps. Per thread: a
    /// tile's sums. For callers that budget memory.
    pub fn bytes(&self, out_len: &[usize]) -> (u64, u64) {
        let n = out_len.len();
        let ext: u64 = (0..n)
            .map(|a| (out_len[a] + self.footprint.shape[a] - 1) as u64)
            .product();
        let maps: u64 = (0..n)
            .map(|a| (out_len[a] + self.footprint.shape[a]) as u64 * 16)
            .sum();
        (ext * 8 + maps, LANES as u64 * 8)
    }

    /// The correlation of a block of `shape` holding, per axis, the part `spans[a]`
    /// describes (as for `correlate1d`), into `output` (the spans' outputs, C order).
    /// Lines in parallel on the current rayon pool when `parallel`.
    pub fn run_on<T: Element, U: Element + Default>(
        &self,
        parallel: bool,
        input: &[T],
        shape: &[usize],
        spans: &[Span],
        output: &mut [U],
    ) -> Result<(), FilterError> {
        let Some(win) = Window::new(
            &self.footprint,
            self.mode,
            input.len(),
            shape,
            spans,
            output.len(),
        )?
        else {
            return Ok(());
        };
        if self.weights.is_empty() {
            // no weight above DBL_EPSILON: every sum is scipy's starting 0.0
            output.fill(U::from_f64_safe(0.0));
            return Ok(());
        }
        // the box in double, as scipy reads each element; cval as it is
        let boxed: Vec<f64> = win.extend(input, shape, |v| v.to_f64(), self.mode.cval());
        let terms: Vec<(f64, usize)> = self
            .weights
            .iter()
            .copied()
            .zip(win.offsets(&self.positions))
            .collect();
        let last = *win.out_len.last().unwrap_or(&1);
        win.lines(
            parallel,
            output,
            || vec![0f64; LANES],
            |acc, o, dst| {
                let base = win.line_base(o);
                let mut t0 = 0;
                while t0 < last {
                    let m = LANES.min(last - t0);
                    sum_tile(&boxed[base + t0..], &terms, &mut acc[..m]);
                    for (d, v) in dst[t0..t0 + m].iter_mut().zip(acc.iter()) {
                        *d = U::from_f64_safe(*v);
                    }
                    t0 += m;
                }
            },
        );
        Ok(())
    }
}

/// `acc[l] = 0.0 + w0 * at[off0 + l] + w1 * at[off1 + l] + ...`, term after term (each
/// `+ w * x` as scipy's compiled loop does it: [`madd`]).
#[inline(never)]
fn sum_tile(at: &[f64], terms: &[(f64, usize)], acc: &mut [f64]) {
    let m = acc.len();
    acc.fill(0.0);
    for &(w, off) in terms {
        let row = &at[off..off + m];
        for (a, x) in acc.iter_mut().zip(row) {
            *a = madd(*a, w, *x);
        }
    }
}

/// scipy.ndimage.correlate on a whole array (`convolve`: scipy.ndimage.convolve), lines
/// in parallel on the current rayon pool when `parallel`.
#[allow(clippy::too_many_arguments)]
pub fn correlate_on<T: Element, U: Element + Default>(
    parallel: bool,
    input: &[T],
    shape: &[usize],
    kernel_shape: Vec<usize>,
    weights: &[f64],
    origin: Vec<isize>,
    mode: Mode,
    convolve: bool,
    output: &mut [U],
) -> Result<(), FilterError> {
    if kernel_shape.len() != shape.len() {
        return Err(FilterError::new(format!(
            "a {}-d kernel for a {}-d array",
            kernel_shape.len(),
            shape.len()
        )));
    }
    let c = if convolve {
        Correlate::convolve(kernel_shape, weights, origin, mode)?
    } else {
        Correlate::new(kernel_shape, weights, origin, mode)?
    };
    let spans: Vec<Span> = shape.iter().map(|&l| Span::whole(l)).collect();
    c.run_on(parallel, input, shape, &spans, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correlation by its definition: every window summed in scipy's order.
    fn direct(
        data: &[f64],
        shape: &[usize],
        kshape: &[usize],
        w: &[f64],
        origin: &[isize],
        mode: Mode,
    ) -> Vec<f64> {
        let n = shape.len();
        let total: usize = shape.iter().product();
        let fp = Footprint {
            shape: kshape.to_vec(),
            mask: None,
            origin: origin.to_vec(),
        };
        let pos = fp.positions();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; n];
        for _ in 0..total {
            let mut acc = 0.0;
            for (p, &wk) in pos.iter().zip(w) {
                if wk.abs() <= f64::EPSILON || wk.is_nan() {
                    continue;
                }
                let mut flat = Some(0usize);
                for a in 0..n {
                    let (before, _) = fp.reach(a);
                    let q = idx[a] as i64 + p[a] as i64 - before as i64;
                    flat = match (flat, mode.index(q, shape[a] as i64)) {
                        (Some(f), Some(g)) => Some(f * shape[a] + g as usize),
                        _ => None,
                    };
                }
                acc = madd(acc, wk, flat.map_or(mode.cval(), |f| data[f]));
            }
            out.push(acc);
            for a in (0..n).rev() {
                idx[a] += 1;
                if idx[a] < shape[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        out
    }

    const MODES: [Mode; 5] = [
        Mode::Reflect,
        Mode::Mirror,
        Mode::Nearest,
        Mode::Wrap,
        Mode::Constant(2.5),
    ];

    #[test]
    fn equals_the_definition() {
        let shape = [5usize, 6, 300];
        let data: Vec<f64> = (0..9000)
            .map(|v| ((v * 7919) % 31) as f64 - 15.25)
            .collect();
        let kernels: [(Vec<usize>, Vec<isize>); 3] = [
            (vec![3, 3, 3], vec![0, 0, 0]),
            (vec![2, 1, 4], vec![0, 0, -1]),
            (vec![1, 5, 2], vec![0, 2, 0]),
        ];
        for (k, origin) in &kernels {
            let size: usize = k.iter().product();
            let mut w: Vec<f64> = (0..size)
                .map(|i| ((i * 37) % 11) as f64 / 7.0 - 0.6)
                .collect();
            w[size / 2] = 0.0; // dropped, as scipy drops it
            for mode in MODES {
                let want = direct(&data, &shape, k, &w, origin, mode);
                for parallel in [false, true] {
                    let mut got = vec![0f64; data.len()];
                    correlate_on(
                        parallel,
                        &data,
                        &shape,
                        k.clone(),
                        &w,
                        origin.clone(),
                        mode,
                        false,
                        &mut got,
                    )
                    .unwrap();
                    assert_eq!(got, want, "{k:?} {mode:?}");
                }
            }
        }
    }

    #[test]
    fn convolve_is_the_reversed_correlation() {
        let shape = [7usize, 9];
        let data: Vec<f64> = (0..63).map(|v| ((v * 13) % 17) as f64).collect();
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // even shape along the last axis: origin -0 - 1
        let mut conv = vec![0f64; 63];
        correlate_on(
            false,
            &data,
            &shape,
            vec![2, 4],
            &w,
            vec![0, 0],
            Mode::Mirror,
            true,
            &mut conv,
        )
        .unwrap();
        let rev: Vec<f64> = w.iter().rev().copied().collect();
        let want = direct(&data, &shape, &[2, 4], &rev, &[-1, -1], Mode::Mirror);
        assert_eq!(conv, want);
    }

    #[test]
    fn a_part_of_the_array_equals_that_part_of_the_whole() {
        let shape = [6usize, 9, 20];
        let data: Vec<i32> = (0..1080).map(|v| (v * 37) % 101 - 50).collect();
        let w: Vec<f64> = (0..60).map(|i| (i % 7) as f64 - 3.0).collect();
        for mode in MODES {
            let c = Correlate::new(vec![3, 4, 5], &w, vec![0, -1, 1], mode).unwrap();
            let spans: Vec<Span> = shape.iter().map(|&l| Span::whole(l)).collect();
            let mut whole = vec![0i32; 1080];
            c.run_on(false, &data, &shape, &spans, &mut whole).unwrap();
            let (bs, be) = match mode {
                Mode::Wrap => ([0usize, 0, 0], [6usize, 9, 20]),
                _ => ([0, 0, 1], [6, 9, 20]),
            };
            // the kernel reaches 3 back along the last axis (origin 1): outputs from 4
            let (os, oe) = ([1usize, 2, 4], [4usize, 9, 17]);
            let bshape: Vec<usize> = (0..3).map(|a| be[a] - bs[a]).collect();
            let mut block = Vec::new();
            for i in bs[0]..be[0] {
                for j in bs[1]..be[1] {
                    for l in bs[2]..be[2] {
                        block.push(data[(i * 9 + j) * 20 + l]);
                    }
                }
            }
            let spans: Vec<Span> = (0..3)
                .map(|a| Span {
                    len: shape[a],
                    start: bs[a],
                    out_start: os[a],
                    out_len: oe[a] - os[a],
                })
                .collect();
            let mut got = vec![0i32; 3 * 7 * 13];
            c.run_on(true, &block, &bshape, &spans, &mut got).unwrap();
            let mut q = 0;
            for i in os[0]..oe[0] {
                for j in os[1]..oe[1] {
                    for l in os[2]..oe[2] {
                        assert_eq!(got[q], whole[(i * 9 + j) * 20 + l], "{mode:?}");
                        q += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn no_weight_left_gives_zeros() {
        let data = [1u8, 2, 3, 4];
        let mut out = [9u8; 4];
        let w = [0.0, 1e-17, f64::NAN];
        correlate_on(
            false,
            &data,
            &[4],
            vec![3],
            &w,
            vec![0],
            Mode::Reflect,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [0; 4]);
    }

    #[test]
    fn unsigned_outputs_take_scipys_safe_cast() {
        assert_eq!(u8::from_f64_safe(-200.0), 56);
        assert_eq!(u8::from_f64_safe(-0.5), 0);
        // uint64 below -2^63: -(uint64)(1e19), not the conversion's indefinite value
        assert_eq!(
            u64::from_f64_safe(-1e19),
            10_000_000_000_000_000_000u64.wrapping_neg()
        );
        assert_eq!(i16::from_f64_safe(-40000.0), i16::from_f64(-40000.0));
    }
}
