//! Passes whose state runs along the line: scipy.ndimage's uniform filter (a running sum,
//! `NI_UniformFilter1D`) and its minimum / maximum filters (Harter's sliding minimum,
//! `NI_MinOrMaxFilter1D`). Both follow scipy's loops step by step, so every value is
//! scipy's.
//!
//! The running sum's rounding depends on where the line starts; a whole line here starts
//! where scipy's does (the extended line of the positions computed). The sliding minimum
//! depends only on the values inside each window (older values are dropped first).

use rayon::prelude::*;

use crate::correlate::{layout, Layout, LANE};
use crate::{Element, FilterError, Mode, Span};

/// The output, written by tasks that own disjoint elements of it.
#[derive(Clone, Copy)]
struct Shared<U>(*mut U, usize);

// SAFETY: the tasks below write disjoint elements (each owns a block of lanes of one
// outer index), and the slice outlives them (the call borrows it mutably throughout)
unsafe impl<U: Send> Send for Shared<U> {}
unsafe impl<U: Send> Sync for Shared<U> {}

impl<U> Shared<U> {
    /// # Safety
    /// No other task writes element `i`, and `i < len`.
    #[inline]
    unsafe fn write(self, i: usize, v: U) {
        debug_assert!(i < self.1);
        unsafe { self.0.add(i).write(v) }
    }
}

/// What one line-walking kernel does: from the extended line `buf` (f64) to `out` (one
/// value per output position).
trait Walk: Sync {
    /// Lines a task walks together along an axis other than the last: wide for a kernel
    /// that runs its lanes in step (vectorised), narrow for one that walks them one by
    /// one (its block of lines then stays in the cache).
    const WIDTH: usize;
    fn walk(&self, buf: &[f64], out: &mut [f64]);
    /// The same for `m` lanes at once; `buf` holds the extended line's rows of WIDTH
    /// values, `out` the output rows of WIDTH values. The default walks lane by lane.
    fn walk_lanes(&self, buf: &[f64], m: usize, ext: usize, out_len: usize, out: &mut [f64]) {
        let w = Self::WIDTH;
        let mut line = vec![0f64; ext];
        let mut res = vec![0f64; out_len];
        for i in 0..m {
            for (b, v) in line.iter_mut().enumerate() {
                *v = buf[b * w + i];
            }
            self.walk(&line, &mut res);
            for (q, v) in res.iter().enumerate() {
                out[q * w + i] = *v;
            }
        }
    }
}

/// Run `walk` over every line of a pass (shared driver of the kernels below).
#[allow(clippy::too_many_arguments)]
fn run<T: Element, U: Element, W: Walk>(
    parallel: bool,
    walk: &W,
    input: &[T],
    lay: Layout,
    cval: f64,
    output: &mut [U],
) {
    let Layout {
        outer,
        in_len,
        inner,
        out_len,
        ext,
        rows,
    } = lay;
    if inner == 1 {
        // along the last axis: one contiguous line at a time
        let line = |(buf, res): &mut (Vec<f64>, Vec<f64>), (o, dst): (usize, &mut [U])| {
            let line = &input[o * in_len..(o + 1) * in_len];
            for (b, r) in buf.iter_mut().zip(&rows) {
                *b = r.map_or(cval, |r| line[r].to_f64());
            }
            walk.walk(buf, res);
            for (d, v) in dst.iter_mut().zip(res.iter()) {
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
            let mut b = buffers();
            output
                .chunks_mut(out_len)
                .enumerate()
                .for_each(|x| line(&mut b, x));
        }
        return;
    }
    // along another axis: a task walks WIDTH lines of one outer index together, over the
    // whole extended line, and writes its own lanes of every output row
    let width = W::WIDTH;
    let chunks = inner.div_ceil(width);
    let len = output.len();
    let out = Shared(output.as_mut_ptr(), len);
    let task = |(buf, res): &mut (Vec<f64>, Vec<f64>), t: usize| {
        let (o, l0) = (t / chunks, (t % chunks) * width);
        let m = width.min(inner - l0);
        let base = o * in_len * inner;
        for (b, r) in rows.iter().enumerate() {
            let row = &mut buf[b * width..b * width + m];
            match r {
                Some(r) => {
                    let src = &input[base + r * inner + l0..][..m];
                    for (d, v) in row.iter_mut().zip(src) {
                        *d = v.to_f64();
                    }
                }
                None => row.fill(cval),
            }
        }
        walk.walk_lanes(buf, m, ext, out_len, res);
        let obase = o * out_len * inner;
        for q in 0..out_len {
            for i in 0..m {
                // SAFETY: element (o, q, l0 + i) belongs to this task alone
                unsafe { out.write(obase + q * inner + l0 + i, U::from_f64(res[q * width + i])) };
            }
        }
    };
    let buffers = || (vec![0f64; ext * width], vec![0f64; out_len * width]);
    if parallel {
        (0..outer * chunks)
            .into_par_iter()
            .for_each_init(buffers, task);
    } else {
        let mut b = buffers();
        (0..outer * chunks).for_each(|t| task(&mut b, t));
    }
}

/// Heap bytes of a running pass computing `out_len` positions with a window of `size`:
/// (shared, per thread), as [`crate::pass_bytes`] for correlation. Per thread: a block of
/// LANE extended lines and their outputs, plus a line for the minimum's walk.
pub fn running_pass_bytes(out_len: usize, size: usize) -> (u64, u64) {
    let ext = (out_len + size.max(1) - 1) as u64;
    let shared = ext * std::mem::size_of::<Option<usize>>() as u64;
    let lanes = (ext + out_len as u64) * LANE as u64 * 8;
    let line = (2 * ext + out_len as u64) * 8 + size as u64 * 16;
    (shared, lanes + line)
}

// ------------------------------------------------------------------------- uniform

struct Uniform {
    size: usize,
}

impl Walk for Uniform {
    const WIDTH: usize = LANE;

    /// `NI_UniformFilter1D`: the first window summed from 0.0, then each step adds the
    /// difference of the value entering and the value leaving; every output the sum
    /// divided by the size.
    fn walk(&self, buf: &[f64], out: &mut [f64]) {
        let fs = self.size;
        let mut tmp = 0.0;
        for v in &buf[..fs] {
            tmp += v;
        }
        let d = fs as f64;
        out[0] = tmp / d;
        for ll in 1..out.len() {
            tmp += buf[ll + fs - 1] - buf[ll - 1];
            out[ll] = tmp / d;
        }
    }

    /// The same sums for `m` lanes side by side (each lane's operations in scipy's order).
    fn walk_lanes(&self, buf: &[f64], m: usize, _ext: usize, out_len: usize, out: &mut [f64]) {
        let fs = self.size;
        let d = fs as f64;
        let row = |b: usize| &buf[b * LANE..b * LANE + m];
        let mut tmp = [0f64; LANE];
        let tmp = &mut tmp[..m];
        for b in 0..fs {
            for (t, v) in tmp.iter_mut().zip(row(b)) {
                *t += v;
            }
        }
        for (o, t) in out[..m].iter_mut().zip(tmp.iter()) {
            *o = t / d;
        }
        for ll in 1..out_len {
            let (enter, leave) = (row(ll + fs - 1), row(ll - 1));
            for ((t, e), l) in tmp.iter_mut().zip(enter).zip(leave) {
                *t += e - l;
            }
            for (o, t) in out[ll * LANE..ll * LANE + m].iter_mut().zip(tmp.iter()) {
                *o = t / d;
            }
        }
    }
}

/// scipy.ndimage.uniform_filter1d along `axis` (a window of `size`, shifted by `origin`),
/// as [`crate::correlate1d_on`] for correlation: lines in parallel on the current rayon
/// pool when `parallel`.
#[allow(clippy::too_many_arguments)]
pub fn uniform1d_on<T: Element, U: Element>(
    parallel: bool,
    input: &[T],
    shape: &[usize],
    axis: usize,
    size: usize,
    origin: isize,
    mode: Mode,
    span: Span,
    output: &mut [U],
) -> Result<(), FilterError> {
    let lay = layout(
        input.len(),
        shape,
        axis,
        size,
        origin,
        mode,
        span,
        output.len(),
    )?;
    if let Some(lay) = lay {
        run(parallel, &Uniform { size }, input, lay, mode.cval(), output);
    }
    Ok(())
}

// ------------------------------------------------------------------- minimum / maximum

struct MinMax {
    size: usize,
    minimum: bool,
}

impl Walk for MinMax {
    const WIDTH: usize = 16;

    /// The lanes of a block in place: lane i's line at buf[b * WIDTH + i], its outputs
    /// at out[q * WIDTH + i] (no copy of each line in and out).
    fn walk_lanes(&self, buf: &[f64], m: usize, _ext: usize, out_len: usize, out: &mut [f64]) {
        let (fs, w) = (self.size, Self::WIDTH);
        if fs == 1 {
            for q in 0..out_len {
                out[q * w..q * w + m].copy_from_slice(&buf[q * w..q * w + m]);
            }
            return;
        }
        RING.with(|cell| {
            let mut ring = cell.borrow_mut();
            ring.clear();
            ring.resize(fs, (0f64, 0usize));
            for i in 0..m {
                let (b, o) = (&buf[i..], &mut out[i..]);
                if self.minimum {
                    slide::<true>(fs, out_len, (b, w), (o, w), &mut ring);
                } else {
                    slide::<false>(fs, out_len, (b, w), (o, w), &mut ring);
                }
            }
        });
    }

    /// `NI_MinOrMaxFilter1D` (Richard Harter's sliding minimum): a ring of (value, death)
    /// pairs, front = the current minimum; a value at least as small replaces the front
    /// and empties the ring, a larger one pops the larger-or-equal tail and is appended.
    fn walk(&self, buf: &[f64], out: &mut [f64]) {
        let fs = self.size;
        if fs == 1 {
            out.copy_from_slice(&buf[..out.len()]);
            return;
        }
        // the ring is the thread's own, reused line after line (one allocation per thread)
        RING.with(|cell| {
            let mut ring = cell.borrow_mut();
            ring.clear();
            ring.resize(fs, (0f64, 0usize));
            let n = out.len();
            if self.minimum {
                slide::<true>(fs, n, (buf, 1), (out, 1), &mut ring);
            } else {
                slide::<false>(fs, n, (buf, 1), (out, 1), &mut ring);
            }
        });
    }
}

thread_local! {
    /// Each thread's ring of (value, death) pairs for the sliding minimum.
    static RING: std::cell::RefCell<Vec<(f64, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Harter's loop over one extended line with a ring of `fs` pairs, compiled once for the
/// minimum and once for the maximum (no test of which inside the loop). The comparisons
/// are scipy's, NaN included. The line's values are at `buf[ll * bstep]` and the `n`
/// outputs go to `out[q * ostep]` (steps above 1: one lane of a block, in place).
#[inline]
fn slide<const MIN: bool>(
    fs: usize,
    n: usize,
    (buf, bstep): (&[f64], usize),
    (out, ostep): (&mut [f64], usize),
    ring: &mut [(f64, usize)],
) {
    // `a` comes before `b` in the order the filter keeps (as small / as large or more)
    let before = |a: f64, b: f64| if MIN { a <= b } else { a >= b };
    let ring = &mut ring[..fs];
    assert!(
        fs >= 2 && n >= 1 && buf.len() > (fs + n - 2) * bstep && out.len() > (n - 1) * ostep,
        "a sliding window of {fs} over {n} outputs"
    );
    let inc = |p: usize| if p + 1 >= fs { 0 } else { p + 1 };
    let dec = |p: usize| if p == 0 { fs - 1 } else { p - 1 };
    let (mut front, mut last) = (0usize, 0usize);
    ring[0] = (buf[0], fs);
    let rp = ring.as_mut_ptr();
    // SAFETY: `front` and `last` only take values 0 .. fs (inc / dec wrap there) and the
    // ring holds fs pairs; ll <= fs + n - 2 and ll + 1 - fs <= n - 1, in bounds of their
    // slices at their steps (asserted)
    unsafe {
        for ll in 1..fs + n - 1 {
            let val = *buf.get_unchecked(ll * bstep);
            if (*rp.add(front)).1 == ll {
                front = inc(front);
            }
            if before(val, (*rp.add(front)).0) {
                *rp.add(front) = (val, ll + fs);
                last = front;
            } else {
                while before(val, (*rp.add(last)).0) {
                    last = dec(last);
                }
                last = inc(last);
                *rp.add(last) = (val, ll + fs);
            }
            if ll >= fs - 1 {
                *out.get_unchecked_mut((ll + 1 - fs) * ostep) = (*rp.add(front)).0;
            }
        }
    }
}

/// scipy.ndimage.minimum_filter1d (`minimum`) or maximum_filter1d along `axis`, as
/// [`uniform1d_on`].
#[allow(clippy::too_many_arguments)]
pub fn min_max1d_on<T: Element, U: Element>(
    parallel: bool,
    input: &[T],
    shape: &[usize],
    axis: usize,
    size: usize,
    origin: isize,
    minimum: bool,
    mode: Mode,
    span: Span,
    output: &mut [U],
) -> Result<(), FilterError> {
    let lay = layout(
        input.len(),
        shape,
        axis,
        size,
        origin,
        mode,
        span,
        output.len(),
    )?;
    if let Some(lay) = lay {
        run(
            parallel,
            &MinMax { size, minimum },
            input,
            lay,
            mode.cval(),
            output,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line<W: Walk>(w: &W, buf: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0f64; n];
        w.walk(buf, &mut out);
        out
    }

    #[test]
    fn uniform_is_a_running_mean() {
        let buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(line(&Uniform { size: 3 }, &buf, 4), [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn minimum_and_maximum_slide() {
        let buf = [5.0, 1.0, 4.0, 2.0, 8.0, 3.0, 3.0];
        let mn = MinMax {
            size: 3,
            minimum: true,
        };
        let mx = MinMax {
            size: 3,
            minimum: false,
        };
        assert_eq!(line(&mn, &buf, 5), [1.0, 1.0, 2.0, 2.0, 3.0]);
        assert_eq!(line(&mx, &buf, 5), [5.0, 4.0, 8.0, 8.0, 8.0]);
    }

    #[test]
    fn nan_follows_harters_comparisons() {
        // a NaN never replaces the front nor pops anything; once 5 dies the NaN is the front
        // until it dies too (scipy: minimum_filter1d([5, nan, 3, 4, 6], 2) after the first)
        let buf = [5.0, f64::NAN, 3.0, 4.0, 6.0];
        let out = line(
            &MinMax {
                size: 2,
                minimum: true,
            },
            &buf,
            4,
        );
        assert_eq!(out[0], 5.0);
        assert!(out[1].is_nan());
        assert_eq!(out[2..], [3.0, 4.0]);
    }

    #[test]
    fn lanes_equal_lines() {
        // the lane walk of the uniform filter gives each lane's line walk
        let (ext, n, m) = (9, 7, 3);
        let mut buf = vec![0f64; ext * LANE];
        let lines: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..ext)
                    .map(|b| ((b * 7 + i * 3) % 5) as f64 * 0.1)
                    .collect()
            })
            .collect();
        for (i, l) in lines.iter().enumerate() {
            for (b, v) in l.iter().enumerate() {
                buf[b * LANE + i] = *v;
            }
        }
        let u = Uniform { size: 3 };
        let mut out = vec![0f64; n * LANE];
        u.walk_lanes(&buf, m, ext, n, &mut out);
        for (i, l) in lines.iter().enumerate() {
            let want = line(&u, l, n);
            for q in 0..n {
                assert_eq!(out[q * LANE + i], want[q]);
            }
        }
    }
}
