//! One 1-D correlation pass along an axis of a block: scipy.ndimage.correlate1d
//! (`NI_Correlate1D`), the building block of every separable filter.

use rayon::prelude::*;

use crate::madd;
use crate::{Element, FilterError, Mode};

/// Which part of an axis a pass reads and writes, in the whole array's positions.
///
/// The block holds positions `start .. start + shape[axis]` of an axis `len` long, and
/// the pass computes positions `out_start .. out_start + out_len`. Every position the
/// kernel reaches, after the mode maps it back inside `0 .. len`, must be in the block:
/// so a block padded by the kernel's reach where the array has data gives the same
/// values as filtering the whole array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub len: usize,
    pub start: usize,
    pub out_start: usize,
    pub out_len: usize,
}

impl Span {
    /// The whole axis in, the whole axis out.
    pub fn whole(len: usize) -> Self {
        Self {
            len,
            start: 0,
            out_start: 0,
            out_len: len,
        }
    }
}

/// Lanes of this many elements are summed together (a pass along any axis but the last
/// sums whole rows of the block at once, element by element in scipy's order).
pub(crate) const LANE: usize = 256;

/// Output rows a task of a pass along any axis but the last computes together (the input
/// rows they share are converted to f64 once).
const BLOCK: usize = 16;

/// Heap bytes of one correlation pass computing `out_len` positions with `taps` weights:
/// (shared, per thread). Shared: the extended line's row table. Per thread: a line and its
/// sums (along the last axis) or a block's scratch (along another), whichever is larger.
/// For callers that budget memory.
pub fn pass_bytes(out_len: usize, taps: usize) -> (u64, u64) {
    let ext = (out_len + taps.max(1) - 1) as u64;
    let shared = ext * std::mem::size_of::<Option<usize>>() as u64;
    let line = 2 * ext * 8;
    let block = ((BLOCK + taps) * LANE * 8) as u64;
    (shared, line.max(block))
}

/// What every pass along an axis needs: the block's layout and the extended line's rows.
pub(crate) struct Layout {
    pub(crate) outer: usize,
    pub(crate) in_len: usize,
    pub(crate) inner: usize,
    pub(crate) out_len: usize,
    /// out_len + taps - 1: the extended line's length
    pub(crate) ext: usize,
    /// for every entry b of the extended line (position out_start + b - (taps / 2 + origin)),
    /// its row in the block, or None for the fill (constant mode)
    pub(crate) rows: Vec<Option<usize>>,
}

/// Check a pass's arguments and build its [`Layout`]; None when the output is empty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layout(
    input_len: usize,
    shape: &[usize],
    axis: usize,
    taps: usize,
    origin: isize,
    mode: Mode,
    span: Span,
    output_len: usize,
) -> Result<Option<Layout>, FilterError> {
    let nd = shape.len();
    if axis >= nd {
        return Err(FilterError::new(format!(
            "axis {axis} of a {nd}-d block of shape {shape:?}"
        )));
    }
    if taps == 0 {
        return Err(FilterError::new("a filter of size 0"));
    }
    let (size1, size2) = (taps / 2, taps - taps / 2 - 1);
    if origin < -(size1 as isize) || origin > size2 as isize {
        return Err(FilterError::new(format!(
            "origin {origin} for a size of {taps}: between -{size1} and {size2}"
        )));
    }
    if input_len != shape.iter().product::<usize>() {
        return Err(FilterError::new(format!(
            "{input_len} elements for shape {shape:?}"
        )));
    }
    let in_len = shape[axis];
    if span.start + in_len > span.len || span.out_start + span.out_len > span.len {
        return Err(FilterError::new(format!(
            "{span:?} for a block {in_len} long on axis {axis}"
        )));
    }
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let out_len = span.out_len;
    if output_len != outer * out_len * inner {
        return Err(FilterError::new(format!(
            "output of {output_len} elements; the pass writes {}",
            outer * out_len * inner
        )));
    }
    if output_len == 0 {
        return Ok(None);
    }
    // The extended line, as scipy's line buffer: entry b is position
    // out_start + b - (size1 + origin).
    let ext = out_len + taps - 1;
    let first = span.out_start as i64 - size1 as i64 - origin as i64;
    let rows = (0..ext)
        .map(|b| match mode.index(first + b as i64, span.len as i64) {
            None => Ok(None),
            Some(g) => {
                let r = g - span.start as i64;
                if (0..in_len as i64).contains(&r) {
                    Ok(Some(r as usize))
                } else {
                    Err(FilterError::new(format!(
                        "position {g} of axis {axis} is not in the block                          ({} .. {}): read the filter's reach first",
                        span.start,
                        span.start + in_len
                    )))
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(Layout {
        outer,
        in_len,
        inner,
        out_len,
        ext,
        rows,
    }))
}

/// scipy's symmetry test of the weights: 1 symmetric, -1 antisymmetric, 0 neither.
/// Only odd sizes count, and "equal" means within f64's epsilon, as `NI_Correlate1D`.
fn symmetry(w: &[f64]) -> i8 {
    let n = w.len();
    if n.is_multiple_of(2) {
        return 0;
    }
    let c = n / 2;
    if (1..=c).all(|i| (w[c + i] - w[c - i]).abs() <= f64::EPSILON) {
        1
    } else if (1..=c).all(|i| (w[c + i] + w[c - i]).abs() <= f64::EPSILON) {
        -1
    } else {
        0
    }
}

/// Correlate `input` (C order, `shape`) with `weights` along `axis` into `output`
/// (`shape` with `span.out_len` on `axis`), as scipy.ndimage.correlate1d: the line is
/// read as f64, extended by `mode`, summed in scipy's order and cast to `U`.
///
/// `origin` shifts the kernel as scipy's (`-(n / 2) <= origin <= (n - 1) / 2` for `n`
/// weights). Lines run in parallel when called inside a rayon pool (on its threads),
/// one after another otherwise.
#[allow(clippy::too_many_arguments)]
pub fn correlate1d<T: Element, U: Element>(
    input: &[T],
    shape: &[usize],
    axis: usize,
    weights: &[f64],
    origin: isize,
    mode: Mode,
    span: Span,
    output: &mut [U],
) -> Result<(), FilterError> {
    // outside a pool rayon would use its global pool of every core: one thread then
    let parallel = rayon::current_thread_index().is_some();
    correlate1d_on(
        parallel, input, shape, axis, weights, origin, mode, span, output,
    )
}

/// [`correlate1d`], with lines in parallel on the current rayon pool when `parallel`,
/// else one after another on this thread (also inside a pool: a caller that planned one
/// thread for this pass keeps it to one).
#[allow(clippy::too_many_arguments)]
pub fn correlate1d_on<T: Element, U: Element>(
    parallel: bool,
    input: &[T],
    shape: &[usize],
    axis: usize,
    weights: &[f64],
    origin: isize,
    mode: Mode,
    span: Span,
    output: &mut [U],
) -> Result<(), FilterError> {
    let fs = weights.len();
    let Some(lay) = layout(
        input.len(),
        shape,
        axis,
        fs,
        origin,
        mode,
        span,
        output.len(),
    )?
    else {
        return Ok(());
    };
    let (in_len, inner, out_len, ext, rows) =
        (lay.in_len, lay.inner, lay.out_len, lay.ext, lay.rows);
    let (size1, size2) = (fs / 2, fs - fs / 2 - 1);
    let k = Kernel {
        w: weights,
        size1,
        size2,
        sym: symmetry(weights),
        cval: mode.cval(),
    };
    if inner == 1 {
        // along the last axis: one contiguous line at a time, extended into a buffer
        // (and the sums of the whole line at once, term by term)
        let line = |(buf, acc): &mut (Vec<f64>, Vec<f64>), (o, dst): (usize, &mut [U])| {
            let line = &input[o * in_len..(o + 1) * in_len];
            for (b, r) in buf.iter_mut().zip(&rows) {
                *b = r.map_or(k.cval, |r| line[r].to_f64());
            }
            k.line(buf, acc);
            for (d, v) in dst.iter_mut().zip(acc.iter()) {
                *d = U::from_f64(*v);
            }
        };
        let buffers = || (vec![0f64; ext], vec![0f64; out_len]);
        if parallel {
            output
                .par_chunks_mut(out_len)
                .enumerate()
                .for_each_init(buffers, line);
        } else {
            let mut bufs = buffers();
            output
                .chunks_mut(out_len)
                .enumerate()
                .for_each(|x| line(&mut bufs, x));
        }
    } else {
        // along another axis: blocks of BLOCK output rows (outer o, positions q ..), lane
        // by lane. The input rows a block needs are read as f64 once into a scratch of
        // (BLOCK + n - 1) x LANE; every output element then sums them in scipy's order.
        let block = |scratch: &mut Vec<f64>, o: usize, qb: usize, dst: &mut [U]| {
            let (q0, qn) = (qb * BLOCK, dst.len() / inner);
            let need = qn + fs - 1;
            scratch.resize(need * LANE, 0.0);
            let base = o * in_len * inner;
            for l0 in (0..inner).step_by(LANE) {
                let m = LANE.min(inner - l0);
                for (rr, r) in rows[q0..q0 + need].iter().enumerate() {
                    let srow = &mut scratch[rr * LANE..rr * LANE + m];
                    match r {
                        Some(r) => {
                            let src = &input[base + r * inner + l0..][..m];
                            for (d, v) in srow.iter_mut().zip(src) {
                                *d = v.to_f64();
                            }
                        }
                        None => srow.fill(k.cval),
                    }
                }
                for qi in 0..qn {
                    k.lanes(scratch, qi + size1, m, &mut dst[qi * inner + l0..][..m]);
                }
            }
        };
        let slab = BLOCK * inner;
        if parallel {
            output
                .par_chunks_mut(out_len * inner)
                .enumerate()
                .for_each(|(o, dst)| {
                    dst.par_chunks_mut(slab)
                        .enumerate()
                        .for_each_init(Vec::new, |s, (qb, d)| block(s, o, qb, d))
                });
        } else {
            let mut scratch = Vec::new();
            for (o, dst) in output.chunks_mut(out_len * inner).enumerate() {
                for (qb, d) in dst.chunks_mut(slab).enumerate() {
                    block(&mut scratch, o, qb, d);
                }
            }
        }
    }
    Ok(())
}

struct Kernel<'a> {
    w: &'a [f64],
    size1: usize,
    size2: usize,
    sym: i8,
    cval: f64,
}

impl Kernel<'_> {
    /// Every output of one line from its extended line `buf` into `acc` (one value per
    /// output): the three loops of `NI_Correlate1D`, taken term by term across the whole
    /// line, so each output sees scipy's operations in scipy's order.
    #[inline]
    fn line(&self, buf: &[f64], acc: &mut [f64]) {
        let (w, s1, s2) = (self.w, self.size1, self.size2);
        let m = acc.len();
        match self.sym {
            1 | -1 => {
                for (a, b) in acc.iter_mut().zip(&buf[s1..s1 + m]) {
                    *a = b * w[s1];
                }
                for j in (1..=s1).rev() {
                    let wj = w[s1 - j];
                    let (lo, hi) = (&buf[s1 - j..s1 - j + m], &buf[s1 + j..s1 + j + m]);
                    if self.sym == 1 {
                        for ((a, l), h) in acc.iter_mut().zip(lo).zip(hi) {
                            *a = madd(*a, l + h, wj);
                        }
                    } else {
                        for ((a, l), h) in acc.iter_mut().zip(lo).zip(hi) {
                            *a = madd(*a, l - h, wj);
                        }
                    }
                }
            }
            _ => {
                let last = s1 + s2;
                for (a, b) in acc.iter_mut().zip(&buf[last..last + m]) {
                    *a = b * w[last];
                }
                // jj = t - size1 runs from -size1 to size2 - 1: buffer position ll + t
                for (t, &wt) in w[..last].iter().enumerate() {
                    for (a, b) in acc.iter_mut().zip(&buf[t..t + m]) {
                        *a = madd(*a, *b, wt);
                    }
                }
            }
        }
    }

    /// The same sums for `out.len()` elements of a row at once, from `scratch`: rows of
    /// LANE f64s, the extended positions of one block; `c` is the output's centre row.
    #[inline]
    fn lanes<U: Element>(&self, scratch: &[f64], c: usize, m: usize, out: &mut [U]) {
        let (w, s1, s2) = (self.w, self.size1, self.size2);
        let row = |b: usize| &scratch[b * LANE..b * LANE + m];
        let mut acc = [0f64; LANE];
        let acc = &mut acc[..m];
        match self.sym {
            1 | -1 => {
                for (a, v) in acc.iter_mut().zip(row(c)) {
                    *a = v * w[s1];
                }
                for j in (1..=s1).rev() {
                    let wj = w[s1 - j];
                    let (lo, hi) = (row(c - j), row(c + j));
                    if self.sym == 1 {
                        for ((a, l), h) in acc.iter_mut().zip(lo).zip(hi) {
                            *a = madd(*a, l + h, wj);
                        }
                    } else {
                        for ((a, l), h) in acc.iter_mut().zip(lo).zip(hi) {
                            *a = madd(*a, l - h, wj);
                        }
                    }
                }
            }
            _ => {
                for (a, v) in acc.iter_mut().zip(row(c + s2)) {
                    *a = v * w[s1 + s2];
                }
                for (t, &wt) in w[..s1 + s2].iter().enumerate() {
                    for (a, v) in acc.iter_mut().zip(row(c + t - s1)) {
                        *a = madd(*a, *v, wt);
                    }
                }
            }
        }
        for (d, v) in out.iter_mut().zip(acc.iter()) {
            *d = U::from_f64(*v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole<T: Element, U: Element + Default>(
        x: &[T],
        shape: &[usize],
        axis: usize,
        w: &[f64],
        mode: Mode,
    ) -> Vec<U> {
        let mut out = vec![U::default(); x.len()];
        correlate1d(
            x,
            shape,
            axis,
            w,
            0,
            mode,
            Span::whole(shape[axis]),
            &mut out,
        )
        .unwrap();
        out
    }

    #[test]
    fn matches_scipy_on_a_line() {
        // scipy.ndimage.correlate1d(np.arange(6.0), w, mode=...)
        let x: Vec<f64> = (0..6).map(f64::from).collect();
        let w = [1.0, 2.0, 4.0];
        let got: Vec<f64> = whole(&x, &[6], 0, &w, Mode::Reflect);
        assert_eq!(got, [4.0, 10.0, 17.0, 24.0, 31.0, 34.0]);
        let got: Vec<f64> = whole(&x, &[6], 0, &w, Mode::Constant(-1.0));
        assert_eq!(got, [3.0, 10.0, 17.0, 24.0, 31.0, 10.0]);
        let got: Vec<f64> = whole(&x, &[6], 0, &w, Mode::Wrap);
        assert_eq!(got, [9.0, 10.0, 17.0, 24.0, 31.0, 14.0]);
    }

    #[test]
    fn integers_truncate_and_wrap() {
        // a Sobel derivative [-1, 0, 1] on uint8: 200 - 0 = 200; 0 - 255 wraps
        let x: Vec<u8> = vec![0, 0, 200, 255, 0, 0];
        let got: Vec<u8> = whole(&x, &[6], 0, &[-1.0, 0.0, 1.0], Mode::Reflect);
        assert_eq!(got, [0, 200, 255, (-200i32) as u8, (-255i32) as u8, 0]);
    }

    #[test]
    fn a_region_equals_the_whole_array_cut() {
        // 2-D, along each axis: compute the whole, then rows 3..7 from a padded box
        let shape = [10, 5];
        let x: Vec<f64> = (0..50).map(|v| ((v * 37) % 11) as f64).collect();
        let w = [0.1, 0.2, 0.4, 0.2, 0.1];
        for mode in [
            Mode::Reflect,
            Mode::Mirror,
            Mode::Nearest,
            Mode::Constant(3.0),
        ] {
            let all: Vec<f64> = whole(&x, &shape, 0, &w, mode);
            // the box rows 1 .. 9 (3 - 2 .. 7 + 2)
            let boxed = &x[5..45];
            let mut part = vec![0f64; 4 * 5];
            let span = Span {
                len: 10,
                start: 1,
                out_start: 3,
                out_len: 4,
            };
            correlate1d(boxed, &[8, 5], 0, &w, 0, mode, span, &mut part).unwrap();
            assert_eq!(part, all[15..35], "{mode:?}");
        }
        // at the edge: rows 0..2 from rows 0..4
        let all: Vec<f64> = whole(&x, &shape, 0, &w, Mode::Mirror);
        let mut part = vec![0f64; 10];
        let span = Span {
            len: 10,
            start: 0,
            out_start: 0,
            out_len: 2,
        };
        correlate1d(&x[..20], &[4, 5], 0, &w, 0, Mode::Mirror, span, &mut part).unwrap();
        assert_eq!(part, all[..10]);
    }

    #[test]
    fn a_box_too_small_is_an_error() {
        let x = vec![0f64; 4];
        let mut out = vec![0f64; 2];
        let span = Span {
            len: 10,
            start: 3,
            out_start: 4,
            out_len: 2,
        };
        let e = correlate1d(&x, &[4], 0, &[1.0; 5], 0, Mode::Reflect, span, &mut out);
        assert!(e.unwrap_err().0.contains("not in the block"));
    }

    #[test]
    fn symmetry_as_scipy() {
        assert_eq!(symmetry(&[1.0, 2.0, 1.0]), 1);
        assert_eq!(symmetry(&[-1.0, 0.0, 1.0]), -1);
        assert_eq!(symmetry(&[1.0, 2.0]), 0);
        assert_eq!(symmetry(&[1.0, 2.0, 3.0]), 0);
    }
}
