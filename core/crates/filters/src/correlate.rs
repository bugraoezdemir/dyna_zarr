//! One 1-D correlation pass along an axis of a block: scipy.ndimage.correlate1d
//! (`NI_Correlate1D`), the building block of every separable filter.

use rayon::prelude::*;

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
const LANE: usize = 256;

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
    let nd = shape.len();
    if axis >= nd {
        return Err(FilterError::new(format!(
            "axis {axis} of a {nd}-d block of shape {shape:?}"
        )));
    }
    let fs = weights.len();
    if fs == 0 {
        return Err(FilterError::new("no weights"));
    }
    let (size1, size2) = (fs / 2, fs - fs / 2 - 1);
    if origin < -(size1 as isize) || origin > size2 as isize {
        return Err(FilterError::new(format!(
            "origin {origin} for {fs} weights: between -{size1} and {size2}"
        )));
    }
    if input.len() != shape.iter().product::<usize>() {
        return Err(FilterError::new(format!(
            "{} elements for shape {shape:?}",
            input.len()
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
    if output.len() != outer * out_len * inner {
        return Err(FilterError::new(format!(
            "output of {} elements; the pass writes {}",
            output.len(),
            outer * out_len * inner
        )));
    }
    if output.is_empty() {
        return Ok(());
    }
    // The extended line, as scipy's line buffer: entry b is position
    // out_start + b - (size1 + origin); `rows[b]` is its row in the block (None: cval).
    let ext = out_len + fs - 1;
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
                        "position {g} of axis {axis} is not in the block \
                         ({} .. {}): read the kernel's reach first",
                        span.start,
                        span.start + in_len
                    )))
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let k = Kernel {
        w: weights,
        size1,
        size2,
        sym: symmetry(weights),
        cval: mode.cval(),
    };
    // In parallel only inside a rayon pool the caller entered (its threads); outside one,
    // on this thread: rayon would otherwise use its global pool of every core.
    let parallel = rayon::current_thread_index().is_some();
    if inner == 1 {
        // along the last axis: one contiguous line at a time, extended into a buffer
        let line = |buf: &mut Vec<f64>, (o, dst): (usize, &mut [U])| {
            let line = &input[o * in_len..(o + 1) * in_len];
            for (b, r) in rows.iter().enumerate() {
                buf[b] = r.map_or(k.cval, |r| line[r].to_f64());
            }
            for (ll, d) in dst.iter_mut().enumerate() {
                *d = U::from_f64(k.at(buf, ll + size1));
            }
        };
        if parallel {
            output
                .par_chunks_mut(out_len)
                .enumerate()
                .for_each_init(|| vec![0f64; ext], line);
        } else {
            let mut buf = vec![0f64; ext];
            output
                .chunks_mut(out_len)
                .enumerate()
                .for_each(|x| line(&mut buf, x));
        }
    } else {
        // along another axis: each output row (outer o, position q) sums whole input
        // rows, lane by lane; every element sees scipy's operations in scipy's order
        let slab = |(slab, dst): (usize, &mut [U])| {
            let (o, q) = (slab / out_len, slab % out_len);
            let base = o * in_len * inner;
            let row = |b: usize| rows[b].map(|r| &input[base + r * inner..][..inner]);
            for (l, out) in dst.chunks_mut(LANE).enumerate() {
                k.lanes(&row, q + size1, l * LANE, out);
            }
        };
        if parallel {
            output.par_chunks_mut(inner).enumerate().for_each(slab);
        } else {
            output.chunks_mut(inner).enumerate().for_each(slab);
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
    /// One output value from the extended line `buf`, centred at `c`: the three loops of
    /// `NI_Correlate1D`, term by term.
    #[inline]
    fn at(&self, buf: &[f64], c: usize) -> f64 {
        let (w, s1, s2) = (self.w, self.size1, self.size2);
        match self.sym {
            1 => {
                let mut v = buf[c] * w[s1];
                for j in (1..=s1).rev() {
                    v += (buf[c - j] + buf[c + j]) * w[s1 - j];
                }
                v
            }
            -1 => {
                let mut v = buf[c] * w[s1];
                for j in (1..=s1).rev() {
                    v += (buf[c - j] - buf[c + j]) * w[s1 - j];
                }
                v
            }
            _ => {
                let mut v = buf[c + s2] * w[s1 + s2];
                for t in 0..s1 + s2 {
                    // jj = t - s1 runs from -size1 to size2 - 1
                    v += buf[c + t - s1] * w[t];
                }
                v
            }
        }
    }

    /// The same sums for `out.len()` elements of a row at once: `row(b)` is the input row
    /// at extended position b (None: the fill), read from element `at` on.
    #[inline]
    fn lanes<'r, T: Element, U: Element>(
        &self,
        row: &impl Fn(usize) -> Option<&'r [T]>,
        c: usize,
        at: usize,
        out: &mut [U],
    ) {
        let m = out.len();
        let (w, s1, s2) = (self.w, self.size1, self.size2);
        let mut acc = [0f64; LANE];
        let mut a = [0f64; LANE];
        let mut b = [0f64; LANE];
        let load = |b_: usize, dst: &mut [f64; LANE]| match row(b_) {
            Some(r) => {
                for (d, v) in dst[..m].iter_mut().zip(&r[at..at + m]) {
                    *d = v.to_f64();
                }
            }
            None => dst[..m].fill(self.cval),
        };
        match self.sym {
            1 | -1 => {
                load(c, &mut a);
                for i in 0..m {
                    acc[i] = a[i] * w[s1];
                }
                for j in (1..=s1).rev() {
                    load(c - j, &mut a);
                    load(c + j, &mut b);
                    let wj = w[s1 - j];
                    if self.sym == 1 {
                        for i in 0..m {
                            acc[i] += (a[i] + b[i]) * wj;
                        }
                    } else {
                        for i in 0..m {
                            acc[i] += (a[i] - b[i]) * wj;
                        }
                    }
                }
            }
            _ => {
                load(c + s2, &mut a);
                for i in 0..m {
                    acc[i] = a[i] * w[s1 + s2];
                }
                for (t, &wt) in w[..s1 + s2].iter().enumerate() {
                    load(c + t - s1, &mut a);
                    for i in 0..m {
                        acc[i] += a[i] * wt;
                    }
                }
            }
        }
        for (d, v) in out.iter_mut().zip(&acc[..m]) {
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
