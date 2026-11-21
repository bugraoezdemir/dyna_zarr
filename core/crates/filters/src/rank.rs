//! scipy.ndimage's n-D rank filter (`NI_RankFilter`, which rank_filter, median_filter
//! and percentile_filter use for any array of two or more dimensions), with scipy's
//! values.
//!
//! scipy gathers each window into a double buffer, in the footprint's C order, and picks
//! the element of the wanted rank with a Hoare quickselect (`NI_Select`). The value of a
//! given rank does not depend on how it is found, as long as equal values are equal
//! bits: so where a window holds no NaN and no -0.0, any exact selection gives scipy's
//! value. Here a selection network does it, on many neighbouring outputs at once (a
//! Batcher sorting network cut down to the comparisons the wanted rank depends on: no
//! branches, so the compiler vectorises it). A tile of outputs whose windows touch a NaN
//! or a -0.0 runs scipy's quickselect itself, on the same buffer in the same order, so
//! those outputs equal scipy's too (a NaN makes the result depend on the window's order).
//!
//! scipy's `_rank_filter` sends rank 0 and the last rank to minimum_filter and
//! maximum_filter, and 1-D arrays to another routine (`_rank_filter_1d`); the caller
//! does that dispatch. This module is the n-D routine only.

use rayon::prelude::*;

use crate::{Element, FilterError, Mode, Span};

/// An element the rank filter selects among, with a branch-free minimum and maximum.
pub trait Select: Element + Default + PartialOrd {
    /// The smaller of two values (any of them when equal; never used on NaN or -0.0).
    fn lo(a: Self, b: Self) -> Self;
    /// The larger of two values.
    fn hi(a: Self, b: Self) -> Self;
    /// NaN or -0.0: a value whose window must take scipy's quickselect.
    fn bad(self) -> bool;
    /// Whether scipy's double buffer rounds the type's values (64-bit integers): the
    /// result is then the selected value as a double, cast back.
    const ROUNDS: bool;
    /// Whether the type has NaN or -0.0 at all.
    const FLOAT: bool;
}

macro_rules! select_int {
    ($($t:ty, $rounds:expr;)*) => {$(
        impl Select for $t {
            #[inline(always)]
            fn lo(a: Self, b: Self) -> Self {
                a.min(b)
            }
            #[inline(always)]
            fn hi(a: Self, b: Self) -> Self {
                a.max(b)
            }
            #[inline(always)]
            fn bad(self) -> bool {
                false
            }
            const ROUNDS: bool = $rounds;
            const FLOAT: bool = false;
        }
    )*};
}

select_int! {
    u8, false; i8, false; u16, false; i16, false; u32, false; i32, false;
    u64, true; i64, true;
}

macro_rules! select_float {
    ($($t:ty;)*) => {$(
        impl Select for $t {
            // written as compares and selects (not f32::min), which compile to minps /
            // maxps; NaN never reaches them
            #[inline(always)]
            fn lo(a: Self, b: Self) -> Self {
                if b < a { b } else { a }
            }
            #[inline(always)]
            fn hi(a: Self, b: Self) -> Self {
                if a < b { b } else { a }
            }
            #[inline(always)]
            fn bad(self) -> bool {
                self.is_nan() || self.to_bits() == (-0.0 as $t).to_bits()
            }
            const ROUNDS: bool = false;
            const FLOAT: bool = true;
        }
    )*};
}

select_float! { f32; f64; }

/// Which positions around an output the filter reads: a box of `shape` (C order), all of
/// it or the elements `mask` marks, shifted by `origin` per axis as scipy shifts it.
#[derive(Debug, Clone, PartialEq)]
pub struct Footprint {
    pub shape: Vec<usize>,
    pub mask: Option<Vec<bool>>,
    pub origin: Vec<isize>,
}

impl Footprint {
    /// A whole box of `shape`, centred.
    pub fn sized(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            mask: None,
            origin: vec![0; shape.len()],
        }
    }

    /// The number of positions read (scipy's `filter_size`).
    pub fn size(&self) -> usize {
        match &self.mask {
            Some(m) => m.iter().filter(|&&b| b).count(),
            None => self.shape.iter().product(),
        }
    }

    /// How far the footprint reaches before and after an output along `axis`.
    pub fn reach(&self, axis: usize) -> (usize, usize) {
        // within 0 .. shape (checked when the filter is made)
        let before = ((self.shape[axis] / 2) as isize + self.origin[axis]) as usize;
        (before, self.shape[axis] - 1 - before)
    }

    /// Positions read, in the footprint's C order (scipy's buffer order), as offsets in
    /// the box.
    fn positions(&self) -> Vec<Vec<usize>> {
        let n = self.shape.len();
        let total: usize = self.shape.iter().product();
        let mut out = Vec::new();
        let mut idx = vec![0usize; n];
        for f in 0..total {
            if self.mask.as_ref().is_none_or(|m| m[f]) {
                out.push(idx.clone());
            }
            for k in (0..n).rev() {
                idx[k] += 1;
                if idx[k] < self.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        out
    }

    fn check(&self) -> Result<(), FilterError> {
        let n = self.shape.len();
        if self.origin.len() != n {
            return Err(FilterError::new(format!(
                "{} origins for a {n}-d footprint",
                self.origin.len()
            )));
        }
        if self.shape.contains(&0) {
            return Err(FilterError::new(format!(
                "a footprint of shape {:?}",
                self.shape
            )));
        }
        if let Some(m) = &self.mask {
            if m.len() != self.shape.iter().product::<usize>() {
                return Err(FilterError::new(format!(
                    "a mask of {} for a footprint of shape {:?}",
                    m.len(),
                    self.shape
                )));
            }
        }
        if self.size() == 0 {
            return Err(FilterError::new("a footprint with no element"));
        }
        for (a, (&s, &o)) in self.shape.iter().zip(&self.origin).enumerate() {
            let c = (s / 2) as isize + o;
            if c < 0 || c >= s as isize {
                return Err(FilterError::new(format!(
                    "origin {o} for a footprint of {s} along axis {a} (scipy: invalid origin)"
                )));
            }
        }
        Ok(())
    }
}

/// Batcher's merge-exchange sorting network for `n` wires (Knuth, TAOCP 5.2.2,
/// algorithm M), as compare-exchange pairs (low wire first), kept to those the output on
/// wire `k` depends on.
fn selection_network(n: usize, k: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    if n >= 2 {
        let t = usize::BITS - (n - 1).leading_zeros();
        let mut p = 1usize << (t - 1);
        while p > 0 {
            let (mut q, mut r, mut d) = (1usize << (t - 1), 0usize, p);
            loop {
                for i in 0..n - d {
                    if i & p == r {
                        pairs.push((i as u32, (i + d) as u32));
                    }
                }
                if q == p {
                    break;
                }
                d = q - p;
                q >>= 1;
                r = p;
            }
            p >>= 1;
        }
    }
    // backwards: a comparison matters when either of its wires does
    let mut needed = vec![false; n];
    if k < n {
        needed[k] = true;
    }
    let mut kept: Vec<(u32, u32)> = pairs
        .into_iter()
        .rev()
        .filter(|&(a, b)| {
            let keep = needed[a as usize] || needed[b as usize];
            if keep {
                needed[a as usize] = true;
                needed[b as usize] = true;
            }
            keep
        })
        .collect();
    kept.reverse();
    kept
}

/// scipy's `NI_Select`: the element of rank `rank` among `buffer[min ..= max]`, by Hoare
/// partitioning around the first element, reordering the buffer as scipy does (so NaN
/// gives scipy's result too).
// `!(a > x)` is scipy's `while (a > x)` turned round, and is not `a <= x` for NaN
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn ni_select(buffer: &mut [f64], mut min: usize, mut max: usize, mut rank: usize) -> f64 {
    loop {
        if min == max {
            return buffer[min];
        }
        let x = buffer[min];
        let (mut ii, mut jj) = (min as isize - 1, max as isize + 1);
        loop {
            loop {
                jj -= 1;
                if !(buffer[jj as usize] > x) {
                    break;
                }
            }
            loop {
                ii += 1;
                if !(buffer[ii as usize] < x) {
                    break;
                }
            }
            if ii < jj {
                buffer.swap(ii as usize, jj as usize);
            } else {
                break;
            }
        }
        let jj = jj as usize;
        let left = jj - min + 1;
        if rank < left {
            max = jj;
        } else {
            min = jj + 1;
            rank -= left;
        }
    }
}

/// A rank filter: the element of rank `rank` among a footprint's positions around each
/// output, the array extended at its edges by `mode`.
#[derive(Debug, Clone)]
pub struct RankFilter {
    footprint: Footprint,
    rank: usize,
    mode: Mode,
    /// footprint positions in scipy's order, as offsets in the box
    positions: Vec<Vec<usize>>,
    network: Vec<(u32, u32)>,
}

/// Bytes a tile of lanes aims at (its rows stay in the first-level cache).
const TILE_BYTES: usize = 32 * 1024;

impl RankFilter {
    pub fn new(footprint: Footprint, rank: usize, mode: Mode) -> Result<Self, FilterError> {
        footprint.check()?;
        let size = footprint.size();
        if rank >= size {
            return Err(FilterError::new(format!(
                "rank {rank} for a footprint of {size} (scipy: rank not within filter footprint size)"
            )));
        }
        let positions = footprint.positions();
        let network = selection_network(size, rank);
        Ok(Self {
            footprint,
            rank,
            mode,
            positions,
            network,
        })
    }

    pub fn footprint(&self) -> &Footprint {
        &self.footprint
    }

    /// How far the filter reaches before and after an output along `axis`.
    pub fn reach(&self, axis: usize) -> (usize, usize) {
        self.footprint.reach(axis)
    }

    /// Outputs of a tile along the last axis, for elements of `item` bytes.
    fn lanes(&self, item: usize, out_last: usize) -> usize {
        let k = self.positions.len().max(1);
        let lanes = (TILE_BYTES / (k * item)).clamp(8, 256) / 8 * 8;
        lanes.min(out_last.max(1))
    }

    /// Heap bytes of a run computing `out_len` outputs per axis from a block of
    /// `in_len`, for elements of `item` bytes: (shared, per thread). Shared: the box
    /// extended at the edges (when the block needs extending) and the network. Per
    /// thread: a tile's rows and scipy's buffer. For callers that budget memory.
    pub fn bytes(&self, in_len: &[usize], out_len: &[usize], item: usize) -> (u64, u64) {
        let n = out_len.len();
        let ext: u64 = (0..n)
            .map(|a| (out_len[a] + self.footprint.shape[a] - 1) as u64)
            .product();
        let block: u64 = in_len.iter().map(|&l| l as u64).product();
        let padded = if ext == block { 0 } else { ext * item as u64 };
        let maps: u64 = (0..n)
            .map(|a| (out_len[a] + self.footprint.shape[a]) as u64 * 16)
            .sum();
        let shared = padded + maps + self.network.len() as u64 * 8;
        let k = self.positions.len() as u64;
        let lanes = self.lanes(item, out_len.last().copied().unwrap_or(1)) as u64;
        (shared, k * lanes * item as u64 + k * 8)
    }

    /// The filter on a block of `shape` holding, per axis, the part `spans[a]` describes
    /// (as for `correlate1d`), into `output` (the spans' outputs, C order). Lines in
    /// parallel on the current rayon pool when `parallel`.
    pub fn run_on<T: Select>(
        &self,
        parallel: bool,
        input: &[T],
        shape: &[usize],
        spans: &[Span],
        output: &mut [T],
    ) -> Result<(), FilterError> {
        let n = self.footprint.shape.len();
        if n == 0 || shape.len() != n || spans.len() != n {
            return Err(FilterError::new(format!(
                "a {}-d footprint on a {}-d block with {} spans",
                n,
                shape.len(),
                spans.len()
            )));
        }
        if input.len() != shape.iter().product::<usize>() {
            return Err(FilterError::new(format!(
                "{} elements for a block of shape {shape:?}",
                input.len()
            )));
        }
        let out_len: Vec<usize> = spans.iter().map(|s| s.out_len).collect();
        if output.len() != out_len.iter().product::<usize>() {
            return Err(FilterError::new(format!(
                "{} outputs for {out_len:?}",
                output.len()
            )));
        }
        if output.is_empty() {
            return Ok(());
        }
        // per axis: the block index each extended position reads (None: the constant)
        let mut maps: Vec<Vec<Option<usize>>> = Vec::with_capacity(n);
        for a in 0..n {
            let s = spans[a];
            let (before, _) = self.reach(a);
            let ext = s.out_len + self.footprint.shape[a] - 1;
            if s.start + shape[a] > s.len || s.out_start + s.out_len > s.len {
                return Err(FilterError::new(format!(
                    "a span {s:?} for a block of {} along axis {a}",
                    shape[a]
                )));
            }
            let mut m = Vec::with_capacity(ext);
            for i in 0..ext {
                let p = s.out_start as i64 - before as i64 + i as i64;
                match self.mode.index(p, s.len as i64) {
                    None => m.push(None),
                    Some(g) => {
                        let g = g as usize;
                        if g < s.start || g >= s.start + shape[a] {
                            return Err(FilterError::new(format!(
                                "position {p} along axis {a} reads {g}, outside the block \
                                 ({} .. {})",
                                s.start,
                                s.start + shape[a]
                            )));
                        }
                        m.push(Some(g - s.start));
                    }
                }
            }
            maps.push(m);
        }
        let ext: Vec<usize> = maps.iter().map(|m| m.len()).collect();
        let same = (0..n)
            .all(|a| ext[a] == shape[a] && maps[a].iter().enumerate().all(|(i, &g)| g == Some(i)));
        let padded: Vec<T>;
        let boxed: &[T] = if same {
            input
        } else {
            padded = extend(input, shape, &maps, T::from_f64(self.mode.cval()));
            &padded
        };
        let any_bad = T::FLOAT && boxed.iter().any(|v| v.bad());
        // offsets of the footprint's positions in the extended box
        let mut strides = vec![1usize; n];
        for a in (0..n - 1).rev() {
            strides[a] = strides[a + 1] * ext[a + 1];
        }
        let offsets: Vec<usize> = self
            .positions
            .iter()
            .map(|p| p.iter().zip(&strides).map(|(i, s)| i * s).sum())
            .collect();
        let last = out_len[n - 1];
        let lanes = self.lanes(std::mem::size_of::<T>(), last);
        let k = offsets.len();
        let line = |(rows, buf): &mut (Vec<T>, Vec<f64>), (o, dst): (usize, &mut [T])| {
            // the line's start in the box: its outer coordinates
            let mut rem = o;
            let mut base = 0usize;
            for a in (0..n - 1).rev() {
                base += (rem % out_len[a]) * strides[a];
                rem /= out_len[a];
            }
            let mut t0 = 0;
            while t0 < last {
                let m = lanes.min(last - t0);
                self.tile(
                    &boxed[base + t0..],
                    &offsets,
                    m,
                    lanes,
                    any_bad,
                    rows,
                    buf,
                    &mut dst[t0..t0 + m],
                );
                t0 += m;
            }
        };
        let scratch = || (vec![T::default(); k * lanes], vec![0f64; k]);
        if parallel {
            output
                .par_chunks_mut(last)
                .enumerate()
                .for_each_init(scratch, line);
        } else {
            let mut s = scratch();
            output
                .chunks_mut(last)
                .enumerate()
                .for_each(|x| line(&mut s, x));
        }
        Ok(())
    }

    /// `m` outputs from `at` on: row w of `rows` (stride `lanes`) holds footprint
    /// position w for every output, then the network (or, where a NaN or -0.0 is near,
    /// scipy's quickselect per output) picks the rank.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn tile<T: Select>(
        &self,
        at: &[T],
        offsets: &[usize],
        m: usize,
        lanes: usize,
        any_bad: bool,
        rows: &mut [T],
        buf: &mut [f64],
        dst: &mut [T],
    ) {
        for (w, &off) in offsets.iter().enumerate() {
            rows[w * lanes..w * lanes + m].copy_from_slice(&at[off..off + m]);
        }
        let k = offsets.len();
        let bad = any_bad && (0..k).any(|w| rows[w * lanes..w * lanes + m].iter().any(|v| v.bad()));
        if bad {
            for (l, d) in dst.iter_mut().enumerate() {
                for (w, b) in buf.iter_mut().enumerate() {
                    *b = rows[w * lanes + l].to_f64();
                }
                *d = T::from_f64(ni_select(buf, 0, k - 1, self.rank));
            }
            return;
        }
        for &(a, b) in &self.network {
            let (a, b) = (a as usize, b as usize);
            let (low, high) = rows.split_at_mut(b * lanes);
            let ra = &mut low[a * lanes..a * lanes + m];
            let rb = &mut high[..m];
            for (x, y) in ra.iter_mut().zip(rb.iter_mut()) {
                let (u, v) = (*x, *y);
                *x = T::lo(u, v);
                *y = T::hi(u, v);
            }
        }
        let picked = &rows[self.rank * lanes..self.rank * lanes + m];
        if T::ROUNDS {
            for (d, v) in dst.iter_mut().zip(picked) {
                *d = T::from_f64(v.to_f64());
            }
        } else {
            dst.copy_from_slice(picked);
        }
    }
}

/// The block extended to the box `maps` describes: element i of axis a reads block index
/// `maps[a][i]`, or `cval` where any axis has None.
fn extend<T: Copy>(input: &[T], shape: &[usize], maps: &[Vec<Option<usize>>], cval: T) -> Vec<T> {
    let n = shape.len();
    let ext: Vec<usize> = maps.iter().map(|m| m.len()).collect();
    let total: usize = ext.iter().product();
    let mut out = Vec::with_capacity(total);
    let last = ext[n - 1];
    let in_last = shape[n - 1];
    let mut idx = vec![0usize; n - 1];
    for _ in 0..total / last {
        let mut src = Some(0usize);
        for a in 0..n - 1 {
            src = match (src, maps[a][idx[a]]) {
                (Some(s), Some(g)) => Some(s * shape[a] + g),
                _ => None,
            };
        }
        match src {
            None => out.extend(std::iter::repeat_n(cval, last)),
            Some(s) => {
                let row = &input[s * in_last..(s + 1) * in_last];
                out.extend(maps[n - 1].iter().map(|g| g.map_or(cval, |g| row[g])));
            }
        }
        for a in (0..n - 1).rev() {
            idx[a] += 1;
            if idx[a] < ext[a] {
                break;
            }
            idx[a] = 0;
        }
    }
    out
}

/// scipy.ndimage's n-D rank filter on a whole array (`NI_RankFilter`), lines in
/// parallel on the current rayon pool when `parallel`.
pub fn rank_filter_on<T: Select>(
    parallel: bool,
    input: &[T],
    shape: &[usize],
    footprint: Footprint,
    rank: usize,
    mode: Mode,
    output: &mut [T],
) -> Result<(), FilterError> {
    if footprint.shape.len() != shape.len() {
        return Err(FilterError::new(format!(
            "a {}-d footprint for a {}-d array",
            footprint.shape.len(),
            shape.len()
        )));
    }
    let f = RankFilter::new(footprint, rank, mode)?;
    let spans: Vec<Span> = shape.iter().map(|&l| Span::whole(l)).collect();
    f.run_on(parallel, input, shape, &spans, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_network_selects_every_rank() {
        // the 0-1 principle: a comparison network selects rank k of every input when it
        // does so for every input of zeros and ones
        for n in 1..=14usize {
            for k in 0..n {
                let net = selection_network(n, k);
                for bits in 0u32..1 << n {
                    let mut v: Vec<u8> = (0..n).map(|i| ((bits >> i) & 1) as u8).collect();
                    let mut sorted = v.clone();
                    sorted.sort();
                    for &(a, b) in &net {
                        let (x, y) = (v[a as usize], v[b as usize]);
                        v[a as usize] = x.min(y);
                        v[b as usize] = x.max(y);
                    }
                    assert_eq!(v[k], sorted[k], "n {n} k {k} bits {bits:b}");
                }
            }
        }
    }

    #[test]
    fn larger_networks_select_too() {
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % 1000
        };
        for n in [25usize, 27, 49, 125, 343] {
            for k in [0, 1, n / 2, n - 2, n - 1] {
                let net = selection_network(n, k);
                for _ in 0..20 {
                    let mut v: Vec<u64> = (0..n).map(|_| next()).collect();
                    let mut sorted = v.clone();
                    sorted.sort();
                    for &(a, b) in &net {
                        let (x, y) = (v[a as usize], v[b as usize]);
                        v[a as usize] = x.min(y);
                        v[b as usize] = x.max(y);
                    }
                    assert_eq!(v[k], sorted[k], "n {n} k {k}");
                }
            }
        }
    }

    #[test]
    fn quickselect_finds_the_rank() {
        let v = [5.0, 1.0, 4.0, 1.0, 9.0, 2.0, 6.0];
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (k, want) in sorted.iter().enumerate() {
            let mut b = v.to_vec();
            assert_eq!(ni_select(&mut b, 0, v.len() - 1, k), *want);
        }
        // a NaN pivot: scipy's order of swaps decides (traced by hand from NI_Select:
        // [NaN 1 2] -> [2 1 NaN], then [2 1] -> [1 2])
        let mut b = vec![f64::NAN, 1.0, 2.0];
        assert_eq!(ni_select(&mut b, 0, 2, 0), 1.0);
    }

    /// The rank filter by brute force: every window gathered and sorted.
    fn brute(data: &[f64], shape: &[usize], fp: &Footprint, rank: usize, mode: Mode) -> Vec<f64> {
        let n = shape.len();
        let total: usize = shape.iter().product();
        let pos = fp.positions();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; n];
        for _ in 0..total {
            let mut w: Vec<f64> = pos
                .iter()
                .map(|p| {
                    let mut flat = Some(0usize);
                    for a in 0..n {
                        let (before, _) = fp.reach(a);
                        let q = idx[a] as i64 + p[a] as i64 - before as i64;
                        flat = match (flat, mode.index(q, shape[a] as i64)) {
                            (Some(f), Some(g)) => Some(f * shape[a] + g as usize),
                            _ => None,
                        };
                    }
                    flat.map_or(mode.cval(), |f| data[f])
                })
                .collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.push(w[rank]);
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
    fn equals_brute_force() {
        let shape = [5usize, 6, 7];
        let data: Vec<f64> = (0..210).map(|v| ((v * 7919) % 31) as f64 - 15.0).collect();
        let fps = [
            Footprint::sized(&[3, 3, 3]),
            Footprint::sized(&[1, 4, 2]),
            Footprint {
                shape: vec![2, 3, 3],
                mask: Some((0..18).map(|i| i % 3 != 1).collect()),
                origin: vec![0, -1, 1],
            },
            Footprint::sized(&[7, 1, 9]),
        ];
        for fp in &fps {
            for mode in MODES {
                for rank in [1, fp.size() / 2, fp.size() - 2] {
                    let want = brute(&data, &shape, fp, rank, mode);
                    for parallel in [false, true] {
                        let mut got = vec![0f64; data.len()];
                        rank_filter_on(parallel, &data, &shape, fp.clone(), rank, mode, &mut got)
                            .unwrap();
                        assert_eq!(got, want, "{fp:?} {mode:?} rank {rank}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_part_of_the_array_equals_that_part_of_the_whole() {
        let shape = [6usize, 9, 20];
        let data: Vec<i32> = (0..1080).map(|v| (v * 37) % 101 - 50).collect();
        let fp = Footprint::sized(&[3, 4, 5]);
        for mode in MODES {
            let mut whole = vec![0i32; data.len()];
            rank_filter_on(false, &data, &shape, fp.clone(), 30, mode, &mut whole).unwrap();
            // outputs 1..4, 2..9, 3..17 from the block 0..6, 0..9, 1..20 (wrap needs
            // the whole axis, so it reads all of it)
            let (bs, be) = match mode {
                Mode::Wrap => ([0usize, 0, 0], [6usize, 9, 20]),
                _ => ([0, 0, 1], [6, 9, 20]),
            };
            let (os, oe) = ([1usize, 2, 3], [4usize, 9, 17]);
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
            let f = RankFilter::new(fp.clone(), 30, mode).unwrap();
            let mut got = vec![0i32; 3 * 7 * 14];
            f.run_on(true, &block, &bshape, &spans, &mut got).unwrap();
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
    fn nan_tiles_follow_scipy_buffer_order() {
        // with a NaN in the window the result is NI_Select's on the footprint-ordered
        // buffer; check against that directly
        let shape = [3usize, 40];
        let mut data: Vec<f64> = (0..120).map(|v| ((v * 13) % 17) as f64).collect();
        data[45] = f64::NAN;
        data[70] = -0.0;
        let fp = Footprint::sized(&[3, 3]);
        let mut got = vec![0f64; 120];
        rank_filter_on(false, &data, &shape, fp.clone(), 4, Mode::Reflect, &mut got).unwrap();
        for i in 0..3i64 {
            for j in 0..40i64 {
                let mut buf: Vec<f64> = fp
                    .positions()
                    .iter()
                    .map(|p| {
                        let a = Mode::Reflect.index(i + p[0] as i64 - 1, 3).unwrap();
                        let b = Mode::Reflect.index(j + p[1] as i64 - 1, 40).unwrap();
                        data[(a * 40 + b) as usize]
                    })
                    .collect();
                let want = ni_select(&mut buf, 0, 8, 4);
                let g = got[(i * 40 + j) as usize];
                assert!(g.to_bits() == want.to_bits(), "at {i},{j}: {g} vs {want}");
            }
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        let fp = Footprint::sized(&[3, 3]);
        assert!(RankFilter::new(fp.clone(), 9, Mode::Reflect).is_err());
        let off = Footprint {
            origin: vec![2, 0],
            ..fp
        };
        assert!(RankFilter::new(off, 4, Mode::Reflect).is_err());
    }
}
